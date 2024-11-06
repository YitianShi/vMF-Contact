from typing import Any, Dict, Tuple, cast
from ..datasets import DataModule
import einops

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F
from ..nn import BayesianLoss, UncertaintyEstimator, vmfContact
from ..metrics import AUSC
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
import random

Batch = Tuple[torch.Tensor, torch.Tensor]
loss_terms_orientation = {
    "baseline": 1,
    "approach": 1e-1,
    "grasp_width": 10,
    "graspness": 0.1,
}


class vmfContactLightningModule(pl.LightningModule):
    def __init__(
        self,
        args,
        ground_truth_gaussian_spread=0.01,
        label_smoothing=0.0,
        symmetric_baseline_loss=False,
        compute_loss_at_all_layers=False,
        flow_gmm_components=50,
        cp_loss="l1_loss",
        cp_loss_coeff=1.0,
        cp_offset_loss_coeff=1000.0,
        flow_loss_coeff=1e-4,
        debug=False,
    ) -> None:

        super().__init__()
        # loss parameters
        

        self.save_hyperparameters()

        self.args = args
        self.prob_baseline = args.prob_baseline
        self.ground_truth_gaussian_spread = ground_truth_gaussian_spread
        self.label_smoothing = label_smoothing
        self.symmetric_baseline_loss = symmetric_baseline_loss
        self.compute_loss_at_all_layers = compute_loss_at_all_layers

        self.cp_loss = cp_loss
        self.cp_loss_coeff = cp_loss_coeff
        self.cp_offset_loss_coeff = cp_offset_loss_coeff
        self.reconstruction_loss_coeff = 100.
        self.flow_loss_coeff = flow_loss_coeff

        self.debug = debug
        self.batch_size = args.batch_size
        self.automatic_optimization = False

        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate = args.learning_rate
        self.learning_rate_flow = args.learning_rate_flow
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.model = vmfContact(
            args=args,
            image_size=args.image_size,
            embedding_dim=args.embedding_dim,
            scale=args.scale,
            prob_baseline=args.prob_baseline,
            pcd_with_rgb=args.pcd_with_rgb,
        )

        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_features=args.hidden_feat_flow,
            embedding_dim=args.embedding_dim,
            flow_layers=args.flow_layers,
            certainty_budget=args.certainty_budget,
            prob_baseline=args.prob_baseline,
            gmm_components=flow_gmm_components,
        )

        self.loss = BayesianLoss(args.entropy_weight)

        # We have continuous output
        self.ausc = AUSC(loss_terms_orientation.keys(), baseline_unc=self.prob_baseline)

        self.reconstruction_loss = ChamferDistanceL1()

    def training_step(self, batch, _batch_idx: int):

        self.forward_and_loss(batch)

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size)

        self.log(
            "bsl",
            self.losses["baseline"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "app",
            self.losses["approach"],
            prog_bar=True,
            batch_size=self.batch_size,
        )
        if "flow_loss" in self.losses.keys():
            self.log(
                "flow",
                self.losses["flow_loss"],
                prog_bar=True,
                batch_size=self.batch_size,
            )
        if "reconstruction_loss" in self.losses.keys():
            self.log(
                "recon",
                self.losses["reconstruction_loss"],
                prog_bar=True,
                batch_size=self.batch_size,
            )
            
        loss = sum(list(self.losses.values()))
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/lr",
            self.lr_schedulers().get_last_lr()[0],
            sync_dist=True,
            prog_bar=True,
        )


        if (_batch_idx+1) % self.gradient_accumulation_steps == 0:
            opt = self.optimizers()
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()  

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        sch.step(self.current_epoch)

    def on_validation_start(self) -> None:
        self.uncertainty_estimator.train()
    
    
    def validate(self, dataloader: torch.utils.data.DataLoader):
        for batch_idx, batch in dataloader:
            self.validation_step(batch, batch_idx)
    
    @torch.no_grad()
    def validation_step(self, batch, _batch_idx):
        self.uncertainty_estimator.flow.eval()
        
        pred = self.forward_and_loss(batch)
        loss = sum(list(self.losses.values()))

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "val/loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size
        )
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.ausc.compute(), prog_bar=True)
        self.ausc.reset()
        self.model.train()

    def test_step(self, batch, _batch_idx):
        self.uncertainty_estimator.flow.eval()
        
        pred = self.forward_and_loss(batch, val=True)
        loss = sum(list(self.losses.values()))

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "val/loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size
        )
        if self.training:
            self.lr_schedulers().step(loss)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_grouped_parameters_flow = {
                "params": [],
                "weight_decay": 0,
                "lr": self.learning_rate_flow,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
            }

        for _, param in self.uncertainty_estimator.named_parameters():
            optimizer_grouped_parameters_flow["params"].append(param)

        optimizer = build_optimizer_from_cfg(self.model, 
                                              lr=self.learning_rate, 
                                              **self.args.point_backbone_cfgs.optimizer)
        scheduler = build_scheduler_from_cfg(self.args.point_backbone_cfgs, 
                                              optimizer)
        
        optimizer.add_param_group(optimizer_grouped_parameters_flow)

        config = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

        return config

    def forward_and_loss(self, batch) -> Dict[str, torch.Tensor]:
        """
        pred:
        # Grasps (B, num_quries, 8 + bin_num)
            "contact_point":                                (B, num_quries, 3),
                final contact points

            "baseline":                         (B, num_quries, 4),
                baseline

            "bin_score":                                    (B, num_quries, bin_num),
                bin_score

            "graspness":                                      (B, num_quries, 1),
                graspness

        # Auxiliary outputs used to compute the loss or for visualization

            "position_pyramid":                             [(B, num_quries, 3)],
                query points from each level

            "contact_pcd_masks_pyramid":
                for visualization of similarity between
                the contact point and query points

        # Return intermediate results

            "visible_rgb_features_pyramid":                 [(B, n_samples_per_level, n_features)],
                visible_rgb_features_pyramid,

            "visible_pcd_pyramid":
                visible_pcd_pyramid,

            "cp_features":                               [(B, num_quries, n_features)],
                cp_features,

        """
        # torch.autograd.set_detect_anomaly(True)

        input_batch, gt_batch = batch
        pcd = input_batch["pcd"]
        normals = input_batch["normals"]
        self.batch_size = pcd.shape[0]

        if self.prob_baseline == "post" and self.training:
            self.uncertainty_estimator.update_lipschitz()

        # List of ground-truth contact points positions
        self.gt_pos_bt = list(gt_sample["gt_pt"] for gt_sample in gt_batch)
        pred = self.model(pcd, self.gt_pos_bt, debug=self.debug)

        self.losses = {}

        # Compute the loss
        self.position_assignment_to_gt = pred["matched_ind"]
        self._compute_cp_loss(pred)
        self._compute_orientation_loss(pred, gt_batch, pcd=pcd[..., :3], normals=normals)
        self._compute_reconstruction_loss(pred, input_batch["pcd_gt"])

        for k,v in self.losses.items():
            # check inf in loss
            if torch.isinf(v).any():
                print(f"Loss {k} is inf")
                self._compute_orientation_loss(pred, gt_batch, pcd=pcd[..., :3], normals=normals)

        if self.prob_baseline == "post":
           self._flow_loss(pred)

        return pred

    def _compute_reconstruction_loss(self, pred, pcds):
        pred_pcd_bt = pred["reconstructed_pcds"]
        self.losses["reconstruction_loss"] = self.reconstruction_loss(pred_pcd_bt, pcds)
        if self.debug and not self.training:
            for pred_pcd, pcd in zip(pred_pcd_bt, pcds):
                self._vis_pcd(pred_pcd.view(-1, 3), pcd)
        self.losses["reconstruction_loss"] *= self.reconstruction_loss_coeff / self.batch_size 
        
    
    def _compute_cp_loss(self, pred):
        # Find the closest ground-truth contact point for each predicted contact point in each level
        self.losses["cp_loss"] = 0
        cp_pos_bt = pred["position"]
        
        # Calculate the l1 loss between the matched contact points and the ground truth
        for cp_pos, gt_pt, pair_ind in zip(
            cp_pos_bt, self.gt_pos_bt, self.position_assignment_to_gt
        ):
            gt_pt = gt_pt[pair_ind[0]]
            cp_pos = cp_pos[pair_ind[1]]
            self.losses["cp_loss"] = self.losses["cp_loss"] + F.huber_loss(
                cp_pos, gt_pt
            )
            # self.ausc.update(cp_pos.detach(), gt_pt, "cp")

        self.losses["cp_loss"] /= self.batch_size
        self.losses["cp_loss"] *= self.cp_loss_coeff

    def _compute_orientation_loss(self, pred, gt_batch, normals, pcd):
        for loss_term in loss_terms_orientation.keys():
            self.losses[loss_term] = 0

        # Get the ground-truths
        baseline_gt_bt = list(gt_sample["gt_baseline"] for gt_sample in gt_batch)
        approach_gt_bt = list(gt_sample["gt_approach"] for gt_sample in gt_batch)
        width_gt_bt = list(gt_sample["gt_width"] for gt_sample in gt_batch)
        graspness_gt_bt = list(gt_sample["gt_scores"] for gt_sample in gt_batch)

        # Get the predictions
        baselines_bt = pred["baseline"]
        bin_scores_bt = pred["bin_score"]
        features_bt = pred["cp_features"]
        width_bt = pred["grasp_width"]
        graspness_bt = pred["graspness"]

        # Get the normals
        sample_ids = pred["sample_ids"]

        for i in range(self.batch_size):
            pair_ind = self.position_assignment_to_gt[i]

            # Baseline vector loss
            if "baseline" in loss_terms_orientation.keys():
                baseline_gt = baseline_gt_bt[i][pair_ind[0]]
                baseline_params = baselines_bt[i][pair_ind[1]]
                baseline = (
                    baseline_params[..., :3]
                    if self.prob_baseline is not None
                    else baseline_params
                )
                baseline = torch.nn.functional.normalize(baseline, dim=-1)
                feature = features_bt[i][pair_ind[1]].unsqueeze(0)
                normal = normals[i][sample_ids[i]]
                if self.prob_baseline == "post":
                    cp = pred["contact_point"][i][pair_ind[1]]

                    baseline = baseline_params[..., :3] 
                    # Update the posterior of the baseline vector
                    baseline_post = self.uncertainty_estimator.posterior_update(
                        feature, baseline_params, prior=-normal
                    )
                    # Compute the loss
                    baseline_loss = self.loss.forward(baseline_post, baseline_gt)

                    # Update the AUSC metric
                    baseline = baseline_post.mu_post
                    kappa = baseline_post.kappa_likelihood
                    kappa_post = baseline_post.kappa_post
                    if not self.training:
                        self.ausc.update(baseline, baseline_gt, "baseline")
                        self.ausc.update(kappa, key="kappa_lh")
                        self.ausc.update(kappa_post, key="kappa_post")
                        
                elif self.prob_baseline == "lh":
                    # Compute the likelihood distribution of the baseline vector
                    baseline_lh = self.uncertainty_estimator.likelihood_update(
                        baseline_params
                    )
                    # Compute the negative log-likelihood
                    baseline_loss = baseline_lh.negative_log_likelihood(baseline_gt)

                    # Update the AUSC metric
                    baseline = baseline_lh.mu
                    kappa = baseline_lh.kappa
                    if not self.training:
                        self.ausc.update(baseline, baseline_gt, "baseline")
                        self.ausc.update(kappa, key = "kappa_lh")
                else:
                    # Compute the cosine similarity loss between the baseline vectors and the ground truth
                    baseline_loss = (
                        1 - F.cosine_similarity(baseline, baseline_gt, dim=-1).mean()
                    )
                    kappa = None

                    # Update the AUSC metric
                    if not self.training:
                        self.ausc.update(baseline, baseline_gt, "baseline")

                self.losses["baseline"] = (
                    self.losses["baseline"] + baseline_loss
                )

            if "approach" in loss_terms_orientation.keys():
                # Compute the approach vector loss
                approach_gt = approach_gt_bt[i][pair_ind[0]]
                bin_score = bin_scores_bt[i][pair_ind[1]]

                #approach = bin_score[:, :3]
                #approach = torch.nn.functional.normalize(approach, dim=-1)
                #approach_loss = 1 - F.cosine_similarity(approach, approach_gt, dim=-1).mean()
                bin_num = bin_score.shape[-1]
                bin_vectors = rotate_circle_to_batch_of_vectors(bin_num, baseline)  # rotate the bin circle to align with the baseline vector
                # Compute the cosine similarity between each bin vector with the only ground-truth approach vector as ground truth bin score
                bin_score_gt = torch.sum(bin_vectors * approach_gt.unsqueeze(1), dim=-1)
                
                # Compute the l1 loss between the bin score gt and the sigmoid bin score
                approach_loss = F.l1_loss(bin_score, bin_score_gt, reduction="mean")
                self.losses["approach"] = self.losses["approach"] + approach_loss

                approach = (bin_vectors * bin_score.unsqueeze(-1).detach().sigmoid()).sum(1)
                approach = torch.nn.functional.normalize(approach, dim=-1)

                # Update the AUSC metric
                if not self.training:
                    self.ausc.update(approach, approach_gt, "approach")

            # Compute the grasp width loss
            if "grasp_width" in loss_terms_orientation:
                width_gt = width_gt_bt[i][pair_ind[0]]
                width = width_bt[i][pair_ind[1]]
                width_loss = F.huber_loss(width, width_gt)
                self.losses["grasp_width"] = self.losses["grasp_width"] + width_loss
                if not self.training:
                    self.ausc.update(width, width_gt, "grasp_width")

            if "graspness" in loss_terms_orientation:
                # Compute the graspness loss
                graspness_gt = graspness_gt_bt[i][pair_ind[0]]

                graspness_gt_all = torch.zeros(
                    graspness_bt[i].size(0),
                    dtype=torch.float,
                    device=self.device,
                )
                graspness_gt_all[pair_ind[1]] = 1.

                weight_mask = torch.where(
                    graspness_gt_all > 0, 2., 1.,
                )

                graspness_loss = F.binary_cross_entropy_with_logits(
                    graspness_bt[i], graspness_gt_all, weight=weight_mask
                )

                self.losses["graspness"] = self.losses["graspness"] + graspness_loss
                self.ausc.update(graspness_bt[i], graspness_gt_all, "graspness")

            

            # Visualize the predicted contact points and the baseline vector
            if self.debug and not self.training:  
            # if self.losses["baseline"] > 1.8:   
                filter = torch.randint(0, pred["contact_point"][i][pair_ind[1]].shape[0], (100,), device=pred["contact_point"][i][pair_ind[1]].device)
                # group_point_pos = pred["group_point_pos"][-1][i]
                cp = pred["contact_point"][i][pair_ind[1]]
                
                
                #pt = o3d.geometry.PointCloud()
                #pt.points = o3d.utility.Vector3dVector(cp[filter].detach().cpu().numpy())
                #pt.normals = o3d.utility.Vector3dVector(normal[filter].cpu().numpy())
                #pt_2 = pcd[i]
                #pt_2 = o3d.geometry.PointCloud()
                #pt_2.points = o3d.utility.Vector3dVector(pcd[i].cpu().numpy())
                #o3d.visualization.draw_geometries([pt, pt_2], point_show_normal=True)

                # cp_vis1 = o3d.geometry.PointCloud()
                # cp_vis1.points = o3d.utility.Vector3dVector(cp.cpu().numpy())
                # cp_vis1.normals = o3d.utility.Vector3dVector(baseline.cpu().numpy())
                # cp_vis2 = o3d.geometry.PointCloud()
                # cp_vis2.points = o3d.utility.Vector3dVector(cp.cpu().numpy())
                # cp_vis2.normals = o3d.utility.Vector3dVector(-normal.cpu().numpy())
                # o3d.visualization.draw_geometries([cp_vis1, cp_vis2], point_show_normal=True)

                cp_gt = pred["anchor"][i]
                cp2 = cp + width_gt.unsqueeze(-1) * baseline
                cp2_gt = cp_gt + width_gt.unsqueeze(-1) * baseline_gt

                filter = torch.randint(0, cp.shape[0], (100,), device=cp.device)
                self.vis_grasps(
                    samples= pcd[i],
                    groups=None,
                    cp_gt=cp_gt,
                    cp2_gt=cp2_gt,
                    cp=cp,
                    cp2=cp2,
                    kappa=kappa,
                    approach_gt=approach_gt,
                    approach=approach,
                    #bin_vectors=bin_vectors * bin_score.sigmoid().unsqueeze(-1).detach(),
                    #bin_vectors_gt=bin_vectors * bin_score_gt.unsqueeze(-1).detach(),
                )

        for k, v in loss_terms_orientation.items():
            self.losses[k] *= v / self.batch_size

    
    def _vis_pcd(self, pred, gt):

        pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred)
        pcd.paint_uniform_color([0.1, 0.7, 0.1])

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        pcd_gt.paint_uniform_color([0.1, 0.1, 0.7])

        o3d.visualization.draw_geometries([pcd, pcd_gt])


    def _flow_loss(self, pred, groups=20):
        # torch.autograd.set_detect_anomaly(True)
        features = pred["cp_features"]

            # Compute loss
        features = features.view(-1, features.shape[-1]).detach()  # .to(self.model.flow_device)
        loss = self.uncertainty_estimator.forward_kld(features)

        # Make layers Lipschitz continuous
        self.losses["flow_loss"] = loss * self.flow_loss_coeff

    def vis_grasps(
        self,
        samples = None,
        groups=None,
        cp_gt=None,
        cp2_gt=None,
        cp=None,
        cp2=None,
        kappa=None,
        approach_gt=None,
        approach=None,
        bin_vectors=None,
        bin_vectors_gt=None,
        score=None,
    ):

        vis_list = []

        if approach is not None:
            approach = (approach.detach().cpu().numpy() if isinstance(approach, torch.Tensor) else approach)
        if approach_gt is not None:
            approach_gt = (approach_gt.detach().cpu().numpy() if isinstance(approach_gt, torch.Tensor) else approach_gt)
        if cp is not None:
            cp = cp.detach().cpu().numpy() if isinstance(cp, torch.Tensor) else cp
            cp2 = cp2.detach().cpu().numpy() if isinstance(cp2, torch.Tensor) else cp2
        if cp_gt is not None:
            cp_gt = cp_gt.detach().cpu().numpy() if isinstance(cp_gt, torch.Tensor) else cp_gt
            cp2_gt = cp2_gt.detach().cpu().numpy() if isinstance(cp2_gt, torch.Tensor) else cp2_gt
        if score is not None:
            score = score.cpu().numpy() if isinstance(score, torch.Tensor) else score
        if kappa is not None:
            kappa = kappa.detach().cpu().numpy() if isinstance(kappa, torch.Tensor) else kappa

        # Visualize the sampled points
        if samples is not None:
            samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(samples)
            vis_list.append(pcd)

        # Connect line between the cp_gt anchor and the cp
        if cp_gt is not None and cp is not None:
            for q, a in zip(cp, cp_gt):
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([a, q])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector(
                    np.tile([0.1, 0.1, 0.7], (1, 1))
                )
                vis_list.append(line)

        if groups is not None:
            rgb_groups = torch.rand((groups.shape[0], 3))
            groups = (
                groups.cpu().numpy() if isinstance(groups, torch.Tensor) else groups
            )
            rgb_groups = (
                rgb_groups.cpu().numpy()
                if isinstance(rgb_groups, torch.Tensor)
                else rgb_groups
            )
            pcds_groups = []
            for i in range(groups.shape[0]):
                pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(groups[i] + samples[i])
                pcd.points = o3d.utility.Vector3dVector(groups[i])
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(rgb_groups[i], (groups.shape[1], 1))
                )
                pcds_groups.append(pcd)

            vis_list += pcds_groups

        if cp is not None:
            vis_list += draw_grasps(cp, cp2, approach, bin_vectors, score, kappa)
        #if cp_gt is not None:
            #vis_list += draw_grasps(cp_gt, cp2_gt, approach_gt, bin_vectors_gt, score, None, color=[0.1, 0.7, 0.1])

        o3d.visualization.draw_geometries(vis_list)

    def inference(self, 
        pcd, 
        pcd_num = 20000, 
        sample_num=1,
        grasp_height_th=5e-3,
        grasp_width_th=0.07,
        graspness_th=0.1,
        pcd_from_prompt=None,
        ):
        pcd = torch.tensor(pcd, device=self.device, dtype=torch.float32)
        assert pcd.size(-1) == 3
        if pcd.dim() == 3:
            pcd = pcd.view(-1, 3)
        predictions = {}

        pcd = over_or_re_sample(pcd, pcd_num)

        with torch.no_grad():
            self.uncertainty_estimator.flow.eval()
            out = self.model(pcd)
            predictions["contact_point"] = out["contact_point"].squeeze(0)

            baseline_params = out["baseline"]
            if self.prob_baseline == "post":
                feature = out["cp_features"].squeeze(0)
                baseline_post = self.uncertainty_estimator.posterior_update(feature, baseline_params)

                baseline_vec = baseline_post.mu_post
                kappa = baseline_post.kappa_post.squeeze()
            
            else:
                baseline_vec = baseline_params[..., :3]
                kappa = baseline_params[..., -1].exp().squeeze()

            baseline_vec = torch.nn.functional.normalize(baseline_vec, dim=-1)
                
            predictions["baseline"] = baseline_vec.squeeze(0)
            predictions["kappa"] = kappa

            bin_score = out["bin_score"].squeeze(0)
            bin_vectors = rotate_circle_to_batch_of_vectors(
                        bin_score.shape[-1], baseline_vec.squeeze(0)
                    )
            approach = (bin_vectors * bin_score.sigmoid().unsqueeze(-1).detach()).sum(1)
            approach = torch.nn.functional.normalize(approach, dim=-1)
            predictions["approach"] = approach

            grasp_width = predictions["grasp_width"] = out["grasp_width"].squeeze(0)
            graspness = predictions["graspness"] = out["graspness"].squeeze(0).sigmoid()

        
        cp = predictions["contact_point"]
        cp2 = predictions["contact_point"] + predictions["grasp_width"].unsqueeze(-1) * predictions["baseline"]

        filter = (graspness > graspness_th) & \
                    (grasp_width < grasp_width_th) & \
                    (cp[..., -1] > grasp_height_th) & \
                    (cp2[..., -1] > grasp_height_th)

        if pcd_from_prompt is not None:
            pcd_from_prompt = torch.tensor(pcd_from_prompt, device=self.device, dtype=torch.float32)
            # calculate the distance between the contact points and the prompt points
            dist = torch.cdist(cp, pcd_from_prompt)
            # dist2 = torch.cdist(cp2, pcd_from_prompt)
            filter = filter & (dist.min(1).values < 0.01)
        
        if filter.sum() == 0:
            print("No valid grasp")
            return None

        # print(f"Number of grasps: {filter.sum()}")

        cp = cp[filter]
        cp2 = cp2[filter]
        mid_pt = (cp2 + cp) / 2
        
        baseline = predictions["baseline"][filter]
        kappa = predictions["kappa"][filter]
        approach = approach[filter]
        graspness = graspness[filter]

        self.vis_grasps(
            samples=pcd,
            cp=cp,
            cp2=cp2,
            kappa=kappa,
            approach=approach,
            score = graspness,
        )
        
        # rotation approach to 6d pose
        # approach is z, baseline is x
        poses = rotation_from_contact(baseline=baseline, approach=approach, translation=mid_pt)

        sample_num = min(sample_num, poses.size(0))
        # sort poses by graspness
        # print(torch.sort(kappa, descending=True)) # TODO: change to graspness
        poses_candidates = poses[torch.argsort(kappa, descending=True)][:sample_num] # TODO: change to graspness

        #randomly sample 1 poses
        sample_num = min(sample_num, poses_candidates.size(0))
        pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
        
        return pose_chosen.cpu().numpy()


def gram_schmidt(vectors):
    """
    Perform the Gram-Schmidt process on a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (n, d) where n is the number of vectors
                                and d is the dimension of each vector.

    Returns:
        torch.Tensor: A tensor of orthogonal vectors with the same shape as the input.
    """
    n, d = vectors.shape
    ortho_vectors = torch.zeros_like(vectors)

    for i in range(n):
        # Start with the current vector
        ortho_vectors[i] = vectors[i]

        # Subtract the projection of the current vector onto the previous orthogonal vectors
        for j in range(i):
            proj = torch.dot(ortho_vectors[j], vectors[i]) / torch.dot(
                ortho_vectors[j], ortho_vectors[j]
            )
            ortho_vectors[i] -= proj * ortho_vectors[j]

        # Normalize the vector to ensure it has unit length (if needed)
        ortho_vectors[i] = ortho_vectors[i] / torch.norm(ortho_vectors[i])

    return ortho_vectors


def rotate_circle_to_batch_of_vectors(bin_num, target_vectors):
    u_batch = perpendicular_highest_z(target_vectors)
    bin_vectors = generate_bin_vectors(target_vectors, u_batch, bin_num)
    return -bin_vectors


def perpendicular_highest_z(v):
    v = v / torch.norm(v, dim=1, keepdim=True)
    # Components of the input vector
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    u_x = -vx * vz
    u_y = -vy * vz
    u_z = 1 - torch.pow(vz, 2)
    u = torch.stack([u_x, u_y, u_z], dim=1)
    u = u / torch.norm(u, dim=1, keepdim=True)
    return u


def rotation_matrix(v, theta):
    # Normalize v to ensure it's a unit vector
    v = v / torch.norm(v, dim=1, keepdim=True)

    # Components of v
    vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]

    # Compute cos(theta) and sin(theta)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    one_minus_cos_theta = 1 - cos_theta

    # Rotation matrix components
    rotation = torch.zeros(v.size(0), 3, 3, device=v.device)

    rotation[:, 0, 0] = cos_theta.squeeze() + (vx * vx * one_minus_cos_theta).squeeze()
    rotation[:, 0, 1] = (vx * vy * one_minus_cos_theta - vz * sin_theta).squeeze()
    rotation[:, 0, 2] = (vx * vz * one_minus_cos_theta + vy * sin_theta).squeeze()

    rotation[:, 1, 0] = (vy * vx * one_minus_cos_theta + vz * sin_theta).squeeze()
    rotation[:, 1, 1] = cos_theta.squeeze() + (vy * vy * one_minus_cos_theta).squeeze()
    rotation[:, 1, 2] = (vy * vz * one_minus_cos_theta - vx * sin_theta).squeeze()

    rotation[:, 2, 0] = (vz * vx * one_minus_cos_theta - vy * sin_theta).squeeze()
    rotation[:, 2, 1] = (vz * vy * one_minus_cos_theta + vx * sin_theta).squeeze()
    rotation[:, 2, 2] = cos_theta.squeeze() + (vz * vz * one_minus_cos_theta).squeeze()

    return rotation


def generate_bin_vectors(v, u, num_points):
    num_points = (num_points + 1) // 2

    # Define the angles for 90-degree rotations
    theta1 = torch.tensor([torch.pi / 2]).repeat(v.size(0)).to(v.device)
    theta2 = torch.tensor([-torch.pi / 2]).repeat(v.size(0)).to(v.device)

    # Generate rotation matrices for +90 and -90 degrees
    rotation_matrix1 = rotation_matrix(v, theta1)
    rotation_matrix2 = rotation_matrix(v, theta2)

    # Rotate the starting vector by +90 and -90 degrees
    u_plus_90 = torch.matmul(rotation_matrix1, u.unsqueeze(2)).squeeze(2)
    u_minus_90 = torch.matmul(rotation_matrix2, u.unsqueeze(2)).squeeze(2)

    # Generate linspace for the 180-degree coverage
    theta_values = torch.linspace(0, 1, num_points).to(v.device).view(1, -1, 1)

    # Interpolate between u and u_plus_90
    vectors_pos = (1 - theta_values) * u.unsqueeze(
        1
    ) + theta_values * u_plus_90.unsqueeze(1)

    # Interpolate between u and u_minus_90
    vectors_neg = theta_values * u.unsqueeze(1) + (
        1 - theta_values
    ) * u_minus_90.unsqueeze(1)

    # Combine positive and negative rotations
    combined_vectors = torch.cat((vectors_neg, vectors_pos[:, 1:]), dim=1)

    combined_vectors = combined_vectors / torch.norm(
        combined_vectors, dim=2, keepdim=True
    )

    return combined_vectors


# Function to create a cylinder between two points
def create_cylinder_between_points(p1, p2, radius=0.05, color=[0.1, 0.1, 0.7]):
    # Calculate the direction and length of the cylinder
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction /= length

    # Create a cylinder mesh
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()

    # Rotate the cylinder to align with the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    # Translate the cylinder to start at point p1
    cylinder.translate((p1 + p2) / 2)

    # Paint the cylinder with the specified color
    cylinder.paint_uniform_color(color)

    return cylinder


def over_or_re_sample(pcd, num_points):
    c = pcd.shape[-1]
    # Determine the maximum size
    if pcd.shape[0] < num_points:
        # Oversample the point cloud
        pad_size = num_points - pcd.shape[0]
        pcd_rest_ind = torch.randint(0, pcd.shape[0], (pad_size,), device=pcd.device)
        pcd = torch.cat([pcd, pcd[pcd_rest_ind]], dim=0)
    else:
        # Resample the point cloud
        indices = torch.randint(0, pcd.shape[0], (num_points,), device=pcd.device)
        pcd = torch.gather(pcd, 0, indices.unsqueeze(-1).expand(-1, c))
    return pcd

def rotation_from_contact(baseline, approach, translation):
    # Example tensors for predictions["baseline"], approach, and translation
    x = baseline  # Baseline vector (B, 3)
    z = approach                 # Approach vector (B, 3)

    # Step 1: Normalize x and z to ensure they are unit vectors
    x_normalized = torch.nn.functional.normalize(x, dim=-1)
    z_normalized = torch.nn.functional.normalize(z, dim=-1)

    # Step 2: Compute the y vector (orthogonal to both x and z, pointing up)
    # First compute y as the cross product of z and x
    y = torch.cross(z_normalized, x_normalized)

    # Define the up direction (positive z-axis)
    up_direction = torch.tensor([0, 0, 1], dtype=x.dtype, device=x.device)

    # Ensure y is aligned with the up direction
    dot_product = torch.sum(y * up_direction, dim=-1, keepdim=True)  # dot product with up direction
    y = torch.where(dot_product < 0, -y, y)  # Flip y if it's pointing downward

    # Step 3: Normalize y to ensure it's a unit vector
    y_normalized = torch.nn.functional.normalize(y, dim=-1)

    # Step 4: Recompute x to ensure orthogonality (optional)
    x_normalized = torch.cross(y_normalized, z_normalized)

    # Step 5: Construct the rotation matrix
    # The rotation matrix is constructed by stacking the x, y, z vectors as columns.
    rotation_matrices = torch.stack([x_normalized, y_normalized, z_normalized], dim=-1)  # Shape (B, 3, 3)

    # Step 6: Construct the homogeneous transformation matrix
    # Create a (B, 4, 4) tensor to store the transformation matrix
    homogeneous_matrices = torch.zeros((x.shape[0], 4, 4), dtype=x.dtype, device=x.device)

    # Place the rotation matrix in the top-left 3x3 block
    homogeneous_matrices[:, :3, :3] = rotation_matrices

    # Place the translation vector in the top-right 3x1 block
    homogeneous_matrices[:, :3, 3] = translation

    # Set the bottom row to [0, 0, 0, 1] for each matrix
    homogeneous_matrices[:, 3, 3] = 1

    return homogeneous_matrices


def draw_grasps(cp, cp2, approach, bin_vectors=None, score=None, kappa=None,
                color=[0.7, 0.1, 0.1], graspline_width=5e-4, finger_length=0.025,
                arm_length=0.02, sphere_radius=2e-3):
    
    vis_list = []
    color_max = np.array([1, 1, 1])  # Light red (RGB)
    color_min = np.array([0, 0, 0])
    cp_half = (cp + cp2) / 2

    if cp is not None and cp2 is not None:
        for i, (q, a, app, half_q, half_a) in enumerate(zip(cp, cp2, approach, 
                                                           cp_half - approach * finger_length, 
                                                           cp_half - approach * (finger_length + arm_length))):
            # Determine color based on score
            color = color_max * score[i] + color_min * (1 - score[i]) if score is not None else color
            
            # Draw fingers and arm cylinders
            vis_list.extend([
                create_cylinder_between_points(a - app * finger_length, a, radius=graspline_width, color=color),
                create_cylinder_between_points(q - app * finger_length, q, radius=graspline_width, color=color),
                create_cylinder_between_points(q - app * finger_length, a - app * finger_length, radius=graspline_width, color=color),
                create_cylinder_between_points(half_q, half_a, radius=graspline_width, color=color)
            ])
            
            # Draw bin_vectors lines if provided
            if bin_vectors is not None:
                bin_vectors_np = bin_vectors.detach().cpu().numpy() if isinstance(bin_vectors, torch.Tensor) else bin_vectors
                for vec in bin_vectors_np[i]:
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([half_q, half_q + vec * 0.1])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.colors = o3d.utility.Vector3dVector([color])
                    vis_list.append(line)

            # Draw spheres if kappa is provided
            if kappa is not None:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * kappa[i] / 10)
                sphere.paint_uniform_color(color)
                sphere.translate(q - app * finger_length)
                vis_list.append(sphere)
    
    return vis_list

