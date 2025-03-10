import numpy as np
import open3d as o3d
import torch
from typing import Union, Optional

#### Codes borrowed from pytorch3d ####


from typing import Optional, Union, Tuple, List
import math
import torch
import torch.nn.functional as F
import random
from vmf_contact.model.utils import *
from vmf_contact.nn.util import match
from .utils import group_and_sum
from sklearn.cluster import DBSCAN
import copy

Device = Union[str, torch.device]
INTEG = True
normal_o3d_vis = False

pcd_from_prompt_matching_th = 0.002
dist_th_pcd=0.01
dist_th_baseline = 0.86

class GraspBuffer:
    def __init__(self, device="cuda:0"):
        self.clear()
        self.dbscan = DBSCAN(eps=1.0, min_samples=1, metric='precomputed')

    def clear(self):
        self.buffer_dict = {
            "pcds": [],
            "baselines": [], 
            "approaches": [], 
            "cp": [], 
            "grasp_width": [],
            "kappa": [], 
            "graspness": [],
            "bin_score": []}
        self.buffer_size = 0
        self.baseline_fused = None
        self.cp_fused= None
        self.grasp_width_fused= None
        self.kappa_fused = None
        self.graspness_fused = None
        self.bin_score_fused = None

    def create_vis(self):
        if normal_o3d_vis:
            self.vis = None
        else:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
    
    def set_view(self):
        """Set a specific viewpoint."""
        ctr = self.vis.get_view_control()

        # Set camera parameters
        ctr.set_zoom(1.2)  # Zoom factor
        ctr.set_lookat(self.view_center)  # Look at center
        ctr.set_front([5, 0, 0])  # View direction
        ctr.set_up([0, 0, 1])  # Up vector
    
    def update_vis(self, vis_list):
        self.vis.clear_geometries()
        for geom in vis_list:
            self.vis.add_geometry(geom)
        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
        self.set_view()

    def vis_grasps(self, pcd_shift=None, interactive_vis=False, fused_pose=False, amplify_kappa=False):

        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to visualize")
            return

        pcd = self.buffer_dict["pcds"][-1]

        if fused_pose:
            baseline, approach, cp, grasp_width, kappa, graspness = self.get_grasp_fused()
        else:
            baseline, approach, cp, grasp_width, kappa, graspness = self.get_grasp_curr()
        cp2 = cp + grasp_width * baseline

        if amplify_kappa:
            kappa *= 80       
        
        vis_list = vis_grasps(
                    samples=pcd,
                    cp=cp,
                    cp2=cp2,
                    kappa=kappa,
                    approach=approach,
                    score = graspness,
                )
        if not hasattr(self, "vis") and pcd_shift is not None:
            self.create_vis()
            self.view_center = pcd_shift
        if interactive_vis or self.vis is None:
            o3d.visualization.draw_geometries(vis_list)
        else:
            self.update_vis(vis_list)

    def push_buffer(self, 
                    pcd,
                    out, 
                    pcd_shift=0., 
                    resize=1.0, 
                    graspness_th = 0.0,
                    grasp_height_th = -0.2, 
                    pcd_from_prompt=None,
                    prob_baseline="likelihood",
                    uncertainty_estimator=None,
                    integrate=False):
        predictions = {}
        predictions["contact_point"] = out["contact_point"].squeeze(0)
        baseline_params = out["baseline"]

        if len(pcd.shape) == 3:
            pcd = pcd.squeeze(0)
        
        if prob_baseline == "post" and uncertainty_estimator is not None:
            feature = out["cp_features"].squeeze(0)
            baseline_post = uncertainty_estimator.posterior_update(feature, baseline_params)
            baseline_vec = baseline_post.mu_post
            kappa = baseline_post.kappa_post.squeeze()
        
        else:
            baseline_vec = baseline_params[..., :3]
            kappa = baseline_params[..., -1].exp().squeeze()
        
        kappa = torch.clip(kappa, 0., 1000.) / 1000.

        baseline_vec = torch.nn.functional.normalize(baseline_vec, dim=-1)
            
        predictions["baseline"] = baseline_vec.squeeze(0)
        predictions["kappa"] = kappa

        bin_score = out["bin_score"].squeeze(0)
        bin_vectors = rotate_circle_to_batch_of_vectors(
                    bin_score.shape[-1], baseline_vec.squeeze(0)
                )
        approach = torch.gather(bin_vectors, 1, bin_score.argmax(dim=-1, keepdim=True)[...,None].expand(-1, -1, 3)).squeeze(1)
        predictions["approach"] = approach
        predictions["bin_score"] = bin_score
        predictions["grasp_width"] = out["grasp_width"].float().squeeze(0)
        predictions["graspness"] = out["graspness"].float().squeeze(0).sigmoid()

        # update the grasp buffer
        valid_grasp = self.update(pcd, 
                                predictions,
                                pcd_shift,
                                resize,
                                # threshold for filtering out invalid grasps
                                grasp_height_th = grasp_height_th, 
                                graspness_th = graspness_th, 
                                pcd_from_prompt = pcd_from_prompt,
                                integrate = integrate
                                )
        return valid_grasp

    def update(self, 
               pcds, 
               predictions,
               pcd_shift=0.0,
               resize=1.0, 
               grasp_height_th=-.2, 
               grasp_width_th=0.2, 
               graspness_th=0.0, 
               pcd_from_prompt=None,
               integrate=False):     
        
        if not isinstance(pcd_shift, torch.Tensor):
            pcd_shift = torch.tensor(pcd_shift, device=pcds.device, dtype=torch.float32)
        if not isinstance(resize, torch.Tensor):
            resize = torch.tensor(resize, device=pcds.device, dtype=torch.float32)
        
        cp = predictions["contact_point"]
        cp2 = predictions["contact_point"] + predictions["grasp_width"].unsqueeze(-1) * predictions["baseline"]
        grasp_width = predictions["grasp_width"].unsqueeze(-1)
        graspness = predictions["graspness"].unsqueeze(-1)
        approach = predictions["approach"]
        baseline = predictions["baseline"]
        kappa = predictions["kappa"].unsqueeze(-1)
        bin_score = predictions["bin_score"]

        filter = (graspness.squeeze(-1) > graspness_th) & \
                    (grasp_width.squeeze(-1) < grasp_width_th) & \
                    (cp[..., -1] > grasp_height_th) & \
                    (cp2[..., -1] > grasp_height_th)
        filter = filter.squeeze(-1)

        if pcd_from_prompt is not None:
            filter = filter & self.filter_grasps_by_pcd(cp, pcd_from_prompt)


        if filter.sum() == 0:
            return False

        # print(f"Number of grasps: {filter.sum()}")
        else:
            pcds = pcds * resize + pcd_shift
            cp = cp[filter] * resize + pcd_shift
            cp2 = cp2[filter] * resize + pcd_shift
            
            baseline = baseline[filter]
            kappa = kappa[filter]
            approach = approach[filter]
            graspness = graspness[filter]
            bin_score = bin_score[filter]
            grasp_width = grasp_width[filter]
            
            if integrate:
                self.integrate(cp, baseline, kappa, bin_score, grasp_width, graspness)
            
            self.buffer_dict["pcds"].append(pcds)  
            self.buffer_dict["baselines"].append(baseline)
            self.buffer_dict["approaches"].append(approach)
            self.buffer_dict["cp"].append(cp)
            self.buffer_dict["kappa"].append(kappa)
            self.buffer_dict["graspness"].append(graspness)
            self.buffer_dict["bin_score"].append(bin_score)
            self.buffer_dict["grasp_width"].append(grasp_width)

            self.buffer_size += 1
            return True
        
    def integrate(self, 
                  cp, 
                  baseline, 
                  kappa, 
                  bin_score, 
                  grasp_width, 
                  graspness):
        
        if self.baseline_fused is None:
            cp, grasp_width, baseline, bin_score, kappa, graspness = self.self_merge_grasps(
                cp, grasp_width, baseline, bin_score, kappa, graspness
            )
            # self fusion
            self.baseline_fused = baseline
            self.bin_score_fused = bin_score
            self.cp_fused= cp
            self.grasp_width_fused= grasp_width
            self.kappa_fused = kappa
            self.graspness_fused = graspness  
            return

        cp_dist = torch.cdist(self.cp_fused, cp)
        cp2 = cp + grasp_width * baseline
        cp2_dist = torch.cdist(self.cp_fused, cp2)

        baseline_dist = torch.mm(self.baseline_fused, baseline.T).abs() # cosine similarity
        # minmum distance between the current grasp and new discovered grasp over threshold will be integrated
        new_grasp_idx = (cp_dist > dist_th_pcd) & (cp2_dist > dist_th_pcd) & (baseline_dist < dist_th_baseline)
        # new grasps should be different from all the fused grasps
        new_grasp_idx = new_grasp_idx.all(dim=0)
        # match the grasps lower than the threshold with the old grasps, perform bayesian update
        matched_idx = ~new_grasp_idx

        if len(matched_idx) > 0:
            self.cross_merge_grasps(cp[matched_idx],
                                    grasp_width[matched_idx],
                                    baseline[matched_idx],
                                    bin_score[matched_idx],
                                    kappa[matched_idx],
                                    graspness[matched_idx])
                    
            # Merge similar new grasps
            cp_new, grasp_width_new, baseline_new, bin_score_new, kappa_new, graspness_new = self.self_merge_grasps(
                cp[new_grasp_idx], 
                grasp_width[new_grasp_idx], 
                baseline[new_grasp_idx], 
                bin_score[new_grasp_idx], 
                kappa[new_grasp_idx], 
                graspness[new_grasp_idx])
            
            # Integrate new grasp points
            self.cp_fused = torch.cat((self.cp_fused, cp_new), dim=0)
            self.grasp_width_fused = torch.cat((self.grasp_width_fused, grasp_width_new), dim=0)
            self.baseline_fused = torch.cat((self.baseline_fused, baseline_new), dim=0)
            self.bin_score_fused = torch.cat((self.bin_score_fused, bin_score_new), dim=0)
            self.kappa_fused = torch.cat((self.kappa_fused, kappa_new), dim=0)
            self.graspness_fused = torch.cat((self.graspness_fused, graspness_new), dim=0)

            # print(f"Total grasp number after fusion: {len(self.cp_fused)}")
    
    def self_merge_grasps(self, cp, grasp_width, baseline, bin_score, kappa, graspness):
        """
        Merge similar grasps based on Euclidean distance and cosine similarity using DBSCAN clustering.
        """
        num_grasps = cp.shape[0]

        cp2 = cp + grasp_width * baseline

        # Compute Euclidean distance between all grasps
        cp_dist = torch.cdist(cp, cp)  # Shape: (num_grasps, num_grasps)
        cp2_dist = torch.cdist(cp, cp2)  # Shape: (num_grasps, num_grasps)

        # Compute cosine similarity between approach vectors
        baseline_dist = torch.mm(baseline, baseline.T).abs()  # Shape: (num_grasps, num_grasps)

        # Identify redundant grasps (low distance and high cosine similarity)
        redundant_mask = ((cp_dist < dist_th_pcd) | (cp2_dist < dist_th_pcd)) & (baseline_dist > dist_th_baseline)
        
        # Get unique indices
        unique_indices = torch.arange(num_grasps)
        
        # Keep only unique grasps
        for i in range(num_grasps):
            if redundant_mask[i].any():
                similar_grasps = torch.where(redundant_mask[i])[0]
                # Merge similar grasps by taking the weighted average
                cp[i] = (cp[similar_grasps] * graspness[similar_grasps]).sum(dim=0) / graspness[similar_grasps].sum()
                grasp_width[i] = (grasp_width[similar_grasps] * graspness[similar_grasps]).sum(dim=0) / graspness[similar_grasps].sum()

                baseline[i] = (
                    baseline[similar_grasps] * kappa[similar_grasps] * graspness[similar_grasps]
                    ).sum(dim=0) / (
                        kappa[similar_grasps] * graspness[similar_grasps]
                        ).sum()

                bin_score[i] = (bin_score[similar_grasps] * graspness[similar_grasps]).sum(dim=0) / graspness[similar_grasps].sum()
                
                kappa[i] = (kappa[similar_grasps]).sum()
                
                graspness[i] = graspness[similar_grasps].mean()

                # Mark similar grasps as duplicates
                unique_indices[similar_grasps] = i

        # Select only unique grasps
        unique_mask = unique_indices == torch.arange(num_grasps)
        return cp[unique_mask], grasp_width[unique_mask], baseline[unique_mask], bin_score[unique_mask], kappa[unique_mask], graspness[unique_mask]  

    def cross_merge_grasps(self, cp, grasp_width, baseline, bin_score, kappa, graspness, decay=0.99):

        self.graspness_fused *= decay
        self.kappa_fused *= decay

        # Match fused and new contact points
        pair_ind = match(self.cp_fused, cp)[0]
        cp_fused_matched = pair_ind[0]
        cp_matched = pair_ind[1]
        
        # Fuse graspness
        graspness_sum, unique_indices, counts = group_and_sum(graspness, cp_matched, cp_fused_matched)
        graspness_mean = graspness_sum / counts
        w = graspness_sum / (self.graspness_fused[unique_indices] + graspness_sum)
        self.graspness_fused[unique_indices] = self.graspness_fused[unique_indices] + graspness_sum
        
        # Fuse contact points
        cp_all = torch.cat((cp, grasp_width), dim=1)
        cp_sum, unique_indices, counts = group_and_sum(cp_all, cp_matched, cp_fused_matched)
        cp_sum, grasp_width_sum = cp_sum.split(cp.size(-1), dim=-1)

        cp_mean = cp_sum / counts
        self.cp_fused[unique_indices] = self.cp_fused[unique_indices] * (1-w) + cp_mean * w
        grasp_width_sum = grasp_width_sum / counts
        self.grasp_width_fused[unique_indices] = self.grasp_width_fused[unique_indices] * (1-w) + grasp_width_sum * w

        # Perform Bayesian update on the fused baseline
        kappa_sum, unique_indices, counts = group_and_sum(kappa, cp_matched, cp_fused_matched)
        baseline_kappa_sum, unique_indices, counts = group_and_sum(baseline * kappa, cp_matched, cp_fused_matched)
        self.baseline_fused[unique_indices] = (
            self.baseline_fused[unique_indices] * self.kappa_fused[unique_indices] + baseline_kappa_sum
        ) / (self.kappa_fused[unique_indices] + kappa_sum)
        self.kappa_fused[unique_indices] = kappa_sum + self.kappa_fused[unique_indices]

        # Bayesian update on the fused approach using Dirichlet distribution
        bin_score_sum, unique_indices, counts = group_and_sum(bin_score, cp_matched, cp_fused_matched)
        self.bin_score_fused[unique_indices] = self.bin_score_fused[unique_indices] + bin_score_sum
    
    def approach_from_bin_score(self, bin_score, baseline):
        bin_num = bin_score.shape[-1]
        bin_vectors = rotate_circle_to_batch_of_vectors(bin_num, baseline)
        return torch.gather(bin_vectors, 1, bin_score.argmax(dim=-1, keepdim=True)[...,None].expand(-1, -1, 3)).squeeze(1)
    
    def get_pcds_all(self):
        return torch.cat(self.buffer_dict["pcds"], dim=0)
    
    def get_grasp_all(self):
        baselines = torch.cat(self.buffer_dict["baselines"], dim=0)
        approaches = torch.cat(self.buffer_dict["approaches"], dim=0)
        cp = torch.cat(self.buffer_dict["cp"], dim=0)
        grasp_width = torch.cat(self.buffer_dict[grasp_width], dim=0)
        kappa = torch.cat(self.buffer_dict["kappa"], dim=0)
        graspness = torch.cat(self.buffer_dict["graspness"], dim=0)
        return baselines, approaches, cp, grasp_width, kappa, graspness
    
    def get_pose_all(self, convention="xzy"):
        baselines, approaches, cp, grasp_width, kappa, graspness = self.get_grasp_all()
        cp2 = cp + grasp_width * baselines
        poses = rotation_from_contact(baseline=baselines, 
                                      approach=approaches, 
                                      translation=(cp+cp2)/2, 
                                      convention=convention)
        return poses, kappa, graspness
    
    def get_grasp_curr(self, pcd_from_prompt=None):
        baseline = self.buffer_dict["baselines"][-1]
        approach = self.buffer_dict["approaches"][-1]
        cp = self.buffer_dict["cp"][-1]
        grasp_width = self.buffer_dict["grasp_width"][-1]
        kappa = self.buffer_dict["kappa"][-1]
        graspness = self.buffer_dict["graspness"][-1]

        if pcd_from_prompt is not None:
            filter = self.filter_grasps_by_pcd(cp, pcd_from_prompt)
            baseline = baseline[filter]
            approach = approach[filter]
            cp = cp[filter]
            grasp_width = grasp_width[filter]
            kappa = kappa[filter]
            graspness = graspness[filter]

        return baseline, approach, cp, grasp_width, kappa, graspness
    
    def get_pose_curr(self, convention="xzy", pcd_from_prompt=None):
        baseline, approach, cp, grasp_width, kappa, graspness = self.get_grasp_curr(pcd_from_prompt)
        cp2 = cp + grasp_width * baseline
        poses = rotation_from_contact(baseline=baseline, 
                                      approach=approach, 
                                      translation=(cp+cp2)/2,
                                      convention=convention)
        return poses, kappa, graspness
    
    def get_pose_curr_best(self, 
                           convention="xzy", 
                           sort_by="graspness", 
                           sample_num=1,
                           pcd_from_prompt=None
                           ):

        return self.get_pose_best(convention, 
                                  sort_by, 
                                  sample_num, 
                                  pcd_from_prompt, 
                                  fused_pose=False
                                  )
    
    def get_grasp_fused(self, pcd_from_prompt=None):
        filter = self.graspness_fused > self.graspness_fused.max() * 0.2
        #filter = filter & (self.kappa_fused > self.kappa_fused.max() * 0.5)
        filter = filter.squeeze(-1)

        if pcd_from_prompt is not None:
            filter = filter & self.filter_grasps_by_pcd(self.cp_fused, pcd_from_prompt)

        approach_fused = self.approach_from_bin_score(self.bin_score_fused, self.baseline_fused)
        baseline = self.baseline_fused[filter]
        approach = approach_fused[filter]
        cp = self.cp_fused[filter]
        grasp_width = self.grasp_width_fused[filter]
        kappa = self.kappa_fused[filter]
        graspness = self.graspness_fused[filter]
        return baseline, approach, cp, grasp_width, kappa, graspness
    
    def get_pose_fused(self, convention="xzy", pcd_from_prompt=None):
        baseline, approach, cp, grasp_width, kappa, graspness = self.get_grasp_fused(pcd_from_prompt)
        cp2 = cp + grasp_width * baseline
        poses = rotation_from_contact(baseline=baseline, 
                                      approach=approach, 
                                      translation=(cp+cp2)/2,
                                      convention=convention)
        if len(poses) == 0:
            pcd_from_prompt_vis = o3d.geometry.PointCloud()
            pcd_from_prompt_vis.points = o3d.utility.Vector3dVector(pcd_from_prompt)
            pcd_from_prompt_vis.paint_uniform_color([0, 0, 1])
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(self.cp_fused.cpu().numpy())
            pcd_vis.paint_uniform_color([1, 0, 0])
            pcd_all_vis = o3d.geometry.PointCloud()
            pcd_all_vis.points = o3d.utility.Vector3dVector(self.get_pcds_all().cpu().numpy())
            pcd_all_vis.paint_uniform_color([0, 1, 0])
            # origine frame
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_from_prompt_vis, pcd_vis, pcd_all_vis, mesh_frame])
            print("No grasp found")
        
        return poses, kappa, graspness
    
    def get_pose_fused_best(self, 
                            convention="xzy", 
                            sort_by="graspness", 
                            sample_num=1,
                            pcd_from_prompt=None
                            ):

        return self.get_pose_best(convention, 
                                  sort_by, 
                                  sample_num, 
                                  pcd_from_prompt)
    
    def get_pose_best(self, 
                convention="xzy", 
                sort_by="graspness", 
                sample_num=1,
                pcd_from_prompt=None,
                fused_pose=True
                ):
        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to choose")
            return None
        
        if fused_pose:
            poses, kappa, graspness = self.get_pose_fused(convention, pcd_from_prompt)  
        else: 
            poses, kappa, graspness = self.get_pose_curr(convention, pcd_from_prompt)
        
        if len(poses) == 0:
            print("No grasp found")
            return None
        
        kappa = kappa.squeeze(-1)
        graspness = graspness.squeeze(-1)
        
        score = kappa if sort_by == "kappa" else graspness
        
        #sort poses by criterion
        sample_num = min(sample_num, poses.size(0))
                
        poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

        #randomly sample 1 poses
        pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
        return pose_chosen
        
    def filter_grasps_by_pcd(self, cp, pcd_from_prompt):
        # pcd_vis = o3d.geometry.PointCloud()
        # pcd_vis.points = o3d.utility.Vector3dVector(pcd_from_prompt)
        # o3d.visualization.draw_geometries([pcd_vis])
        
        if not isinstance(pcd_from_prompt, torch.Tensor):
            pcd_from_prompt = torch.tensor(pcd_from_prompt, device=cp.device, dtype=torch.float32)
        # calculate the distance between the contact points and the prompt points
        dist = torch.cdist(cp, pcd_from_prompt)
        print("Shortest distances between prompted pcd and grasps:",dist.min(1).values)
        filter = dist.min(1).values < pcd_from_prompt_matching_th
        return filter