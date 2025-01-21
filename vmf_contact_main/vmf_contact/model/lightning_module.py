from typing import Any, Dict, Tuple, cast

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from ..nn import BayesianLoss, UncertaintyEstimator, vmfContact
from ..metrics import AUSC
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from .utils import *
import threading
import random
import os

Batch = Tuple[torch.Tensor, torch.Tensor]
loss_terms_orientation = {
    "baseline": 1,
    "approach": 1,
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
        self.reconstruction_loss_coeff = 10.
        self.flow_loss_coeff = flow_loss_coeff

        self.debug = debug
        self.batch_size = args.batch_size
        self.automatic_optimization = False

        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate = args.learning_rate
        self.learning_rate_flow = args.learning_rate_flow
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.grasp_buffer = GraspBuffer(device=self.device)

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
        
        self.forward_and_loss(batch)
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
        self.forward_and_loss(batch, val=True)
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
        if self.debug:
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

                # Regression to the ground-truth approach vector
                #approach = bin_score[:, :3]
                #approach = torch.nn.functional.normalize(approach, dim=-1)
                #approach = gram_schmidt(approach, baseline)
                #approach_loss = 1 - F.cosine_similarity(approach, approach_gt, dim=-1).mean()
                
                # Classification to the ground-truth approach vector
                bin_num = bin_score.shape[-1]
                bin_vectors = rotate_circle_to_batch_of_vectors(bin_num, baseline)  # rotate the bin circle to align with the baseline vector
                # Compute the cosine similarity between each bin vector with the only ground-truth approach vector as ground truth bin score
                bin_score_gt = torch.sum(bin_vectors * approach_gt.unsqueeze(1), dim=-1)
                
                # Compute the l1 loss between the bin score gt and the sigmoid bin score
                approach_loss = F.l1_loss(bin_score, bin_score_gt, reduction="mean")                
                approach = torch.gather(bin_vectors, 1, bin_score.argmax(dim=-1, keepdim=True)[...,None].expand(-1, -1, 3)).squeeze(1)
                # approach = torch.nn.functional.normalize(approach, dim=-1)
                self.losses["approach"] = self.losses["approach"] + approach_loss
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
            if self.debug:  
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
                vis_list = vis_grasps(
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
                o3d.visualization.draw_geometries(vis_list)

        for k, v in loss_terms_orientation.items():
            self.losses[k] *= v / self.batch_size

    def _vis_pcd(self, pred, gt):

        pred = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
        gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred)
        pcd.paint_uniform_color([0.1, 0.7, 0.1])

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt)
        pcd_gt.paint_uniform_color([0.1, 0.1, 0.7])

        o3d.visualization.draw_geometries([pcd])

    def _flow_loss(self, pred, groups=20):
        # torch.autograd.set_detect_anomaly(True)
        paired_ind = self.position_assignment_to_gt
        loss = 0
        for i in range(self.batch_size):
            features = pred["cp_features"][i][paired_ind[i][1]]
            # Compute the flow loss
            loss += self.uncertainty_estimator.forward_kld(features)

        # Make layers Lipschitz continuous
        self.losses["flow_loss"] = loss / self.batch_size * self.flow_loss_coeff

    def inference(self, 
        pcd, 
        pcd_num = 20000,
        shift=0.0,
        resize=1.0, 
        sample_num=1,
        grasp_height_th=5e-3,
        grasp_width_th=0.1,
        graspness_th=0.3,
        pcd_from_prompt=None,
        convention="xzy",
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
            approach = torch.gather(bin_vectors, 1, bin_score.argmax(dim=-1, keepdim=True)[...,None].expand(-1, -1, 3)).squeeze(1)
            predictions["approach"] = approach

            predictions["grasp_width"] = out["grasp_width"].squeeze(0)
            predictions["graspness"] = out["graspness"].squeeze(0).sigmoid()


        # update the grasp buffer
        valid_grasp = self.grasp_buffer.update(pcd, 
                                 predictions,
                                 shift,
                                 resize,
                                 # threshold for filtering out invalid grasps 
                                 grasp_height_th, 
                                 grasp_width_th, 
                                 graspness_th, 
                                 pcd_from_prompt)
        if not valid_grasp:
            print("No valid grasp")
            return None
        self.grasp_buffer.vis_grasps(all=True)
        pose_chosen = self.grasp_buffer.get_pose_curr_best(convention=convention, sample_num=sample_num)
        
        print("Chosen pose", pose_chosen)
        return pose_chosen.cpu().numpy()
        