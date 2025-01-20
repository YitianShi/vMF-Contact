from typing import Any, Dict, Tuple, cast
<<<<<<< HEAD
from ..datasets import DataModule
from .utils import *
=======
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from ..nn import vmfContact, ConditionalUnet1D, UncertaintyEstimator
from openpoints.cpp.chamfer_dist import ChamferDistanceL1
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from .utils import *
import threading
import random
import os

Batch = Tuple[torch.Tensor, torch.Tensor]
loss_terms_orientation = {
    # "baseline": 1,
    # "approach": 1,
    # "grasp_width": 10,
    # "graspness": 0.1,
    "diffusion": 1,
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
        self.learning_rate_score = args.learning_rate_score
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

<<<<<<< HEAD
        self.encoder = vmfContact(
=======
        self.grasp_buffer = GraspBuffer(device=self.device)

        self.model = vmfContact(
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
            args=args,
            image_size=args.image_size,
            embedding_dim=args.embedding_dim,
            scale=args.scale,
            prob_baseline=args.prob_baseline,
            pcd_with_rgb=args.pcd_with_rgb,
            diffusion = True
        )

        self.score_model = ConditionalUnet1D(
            9,
            global_cond_dim=args.embedding_dim,
            diffusion_step_embed_dim=256,
            down_dims=256,
            cond_predict_scale=True
        )

        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_features=args.hidden_feat_flow,
            embedding_dim=args.embedding_dim,
            flow_layers=args.flow_layers,
            certainty_budget=args.certainty_budget,
            prob_baseline=args.prob_baseline,
            gmm_components=flow_gmm_components,
        )

        self.reconstruction_loss = ChamferDistanceL1()

    def training_step(self, batch, _batch_idx: int):

        self.forward_and_loss(batch)

        self.log(
<<<<<<< HEAD
            "diffusion",
            self.losses["diffusion"],
            prog_bar=True,
            batch_size=self.batch_size,
        )

=======
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
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
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
        self.encoder.train()

<<<<<<< HEAD
    def test_step(self, batch, _batch_idx):        
=======
    def test_step(self, batch, _batch_idx):
        self.uncertainty_estimator.flow.eval()        
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
        self.forward_and_loss(batch, val=True)
        loss = sum(list(self.losses.values()))

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size)
        self.log("val/loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size)

        if self.training:
            self.lr_schedulers().step(loss)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_grouped_parameters_score = {
                "params": [],
                "weight_decay": 0,
                "lr": self.learning_rate_score,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
            }

        for _, param in self.score_model.named_parameters():
            optimizer_grouped_parameters_score["params"].append(param)

        optimizer = build_optimizer_from_cfg(self.encoder, 
                                              lr=self.learning_rate, 
                                              **self.args.point_backbone_cfgs.optimizer)
        scheduler = build_scheduler_from_cfg(self.args.point_backbone_cfgs, 
                                              optimizer)
        optimizer.add_param_group(optimizer_grouped_parameters_score)
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

        # List of ground-truth contact points positions
        self.gt_pos_bt = list(gt_sample["gt_pt"] for gt_sample in gt_batch)
        pred = self.encoder(pcd, self.gt_pos_bt, debug=self.debug)

        self.losses = {}

        # Compute the loss
        self.position_assignment_to_gt = pred["matched_ind"]
        self._compute_cp_loss(pred)
        self._compute_orientation_loss(pred, gt_batch, pcd=pcd[..., :3], normals=normals)
<<<<<<< HEAD
        # self._compute_reconstruction_loss(pred, input_batch["pcd_gt"])
=======
        self._compute_reconstruction_loss(pred, input_batch["pcd_gt"])

        for k,v in self.losses.items():
            # check inf in loss
            if torch.isinf(v).any():
                print(f"Loss {k} is inf")
                self._compute_orientation_loss(pred, gt_batch, pcd=pcd[..., :3], normals=normals)

        if self.prob_baseline == "post":
           self._flow_loss(pred)
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718

        return pred        
            
    def _compute_reconstruction_loss(self, pred, pcds):
        pred_pcd_bt =  pred["reconstructed_pcds"]
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

    def _compute_orientation_loss(self, pred, gt_batch, normals, pcd, 
                                  diffusion_schedules =[[1.0, 0.15], [0.15, 0.01]],
                                  ang_mult = 1):

        
        for loss_term in loss_terms_orientation.keys():
            self.losses[loss_term] = 0

        # Get the predictions
        cp_features = pred["cp_features"]

        for i in range(self.batch_size):
            pair_ind = self.position_assignment_to_gt[i]

            gt_sample = gt_batch[i]
            cp = pred["contact_point"][i][pair_ind[1]]
            cp_gt = gt_sample["gt_pt"][pair_ind[0]]
            baseline_gt = gt_sample["gt_baseline"][pair_ind[0]]
            approach_gt = gt_sample["gt_approach"][pair_ind[0]]
            width_gt = gt_sample["gt_width"][pair_ind[0]]

            gt_poses = rotation_from_contact(
                baseline_gt,
                approach_gt,
                cp_gt,
                quat=True,
            )

<<<<<<< HEAD
            time_in = torch.empty(0, device=self.device)
            T_diffused = torch.empty(0,4, device=self.device)
            gt_ang_score = torch.empty(0,3, device=self.device)
            global_cond = torch.empty(0,self.encoder.embedding_dim, device=self.device)
            
            if self.training and not self.debug:

                for time_schedule in diffusion_schedules:
                    time = random_time(min_time=time_schedule[1], max_time=time_schedule[0], device=self.device) # Shape: (1,)
                    T_diffused_, delta_T, time_in_, gt_ang_score_, gt_ang_score_ref_ = self.score_model.diffuse_T_target(
                                                                                                        gt_poses[..., 3:], 
                                                                                                        time=time.to(self.device))
                    
                    T_diffused = torch.cat([T_diffused, T_diffused_], dim=0)
                    time_in = torch.cat([time_in, time_in_], dim=0)
                    gt_ang_score = torch.cat([gt_ang_score, gt_ang_score_], dim=0)
                    global_cond_ = cp_features[i][pair_ind[1]]
                    global_cond = torch.cat([global_cond, global_cond_], dim=0)

                ang_score = self.score_model(T_diffused, timestep=time_in, global_cond = global_cond)
                gt_ang_score = gt_ang_score * torch.sqrt(time[..., None]) * ang_mult
                self.losses["diffusion"] += F.mse_loss(ang_score, gt_ang_score)
=======
            # Visualize the predicted contact points and the baseline vector
            if self.debug:  
            # if self.losses["baseline"] > 1.8:   
                filter = torch.randint(0, pred["contact_point"][i][pair_ind[1]].shape[0], (100,), device=pred["contact_point"][i][pair_ind[1]].device)
                # group_point_pos = pred["group_point_pos"][-1][i]
                cp = pred["contact_point"][i][pair_ind[1]]
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
                
            else:
                init_seed = torch.tensor([0., 1., 0., 0.], device=self.device).repeat(gt_poses.size(0), 1)
                global_cond_ = cp_features[i][pair_ind[1]]
                poses = self.score_model.sample(init_seed, global_cond=global_cond_, diffusion_schedules=diffusion_schedules)
                baselines, approaches = contact_from_quaternion(poses)
                
                self.losses["diffusion"] += (1 - F.cosine_similarity(baselines[-1], baseline_gt, dim=-1).mean())
                self.losses["diffusion"] += (1 - F.cosine_similarity(approaches[-1], approach_gt, dim=-1).mean())

<<<<<<< HEAD
                if self.debug:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    os.makedirs("diff", exist_ok=True)
                    for t in range(baselines.size(0)):
                        t = -1
                        # Clear geometries from the previous frame
                        vis.clear_geometries()
                        
                        baseline, approach = baselines[t], approaches[t]
                        geometry = self.vis_grasps(
                            samples= pcd[i],
                            groups=None,
                            #cp_gt=cp_gt,
                            #cp2_gt=cp_gt + width_gt.unsqueeze(-1) * baseline_gt,
                            cp=cp,
                            cp2=cp + width_gt.unsqueeze(-1) * baseline,
                            #approach_gt=approach_gt,
                            approach=approach,
                            #bin_vectors=bin_vectors * bin_score.sigmoid().unsqueeze(-1).detach(),
                            #bin_vectors_gt=bin_vectors * bin_score_gt.unsqueeze(-1).detach(),
                        )
                        # Add the new geometry for the current frame
                        for geom in geometry:
                            vis.add_geometry(geom)
                        
                        # Update the visualizer
                        vis.poll_events()
                        vis.update_renderer()
                        
                        # Wait a bit to slow down the animation, adjust as needed
                        vis.capture_screen_image(f"diff/frame_{t:04d}.png", do_render=True)
=======
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
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718

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

<<<<<<< HEAD
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
        # Visualize the sampled points
        if samples is not None:
            samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(samples)
            vis_list.append(pcd)
        


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
        if cp_gt is not None:
            vis_list += draw_grasps(cp_gt, cp2_gt, approach_gt, bin_vectors_gt, score, None, color=[0.1, 0.7, 0.1])

        o3d.visualization.draw_geometries(vis_list)
        return vis_list

=======
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
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
            out = self.encoder(pcd)
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
<<<<<<< HEAD

        # print(f"Number of grasps: {filter.sum()}")

        cp = cp[filter]
        cp2 = cp2[filter]
        mid_pt = (cp2 + cp) / 2
        
        baseline = predictions["baseline"][filter]
        kappa = predictions["kappa"][filter]
        approach = approach[filter]
        graspness = graspness[filter]

        if True:
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
        poses = rotation_from_contact(baseline=baseline, approach=approach, translation=mid_pt, convention=convention)

        sample_num = min(sample_num, poses.size(0))
        # sort poses by graspness
        poses_candidates = poses[torch.argsort(kappa, descending=True)][:sample_num] # TODO: change to graspness

        #randomly sample 1 poses
        sample_num = min(sample_num, poses_candidates.size(0))
        pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
        
        print("Chosen pose", pose_chosen)
        return pose_chosen.cpu().numpy()
=======
        self.grasp_buffer.vis_grasps(all=True)
        pose_chosen = self.grasp_buffer.get_pose_curr_best(convention=convention, sample_num=sample_num)
        
        print("Chosen pose", pose_chosen)
        return pose_chosen.cpu().numpy()
        
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
