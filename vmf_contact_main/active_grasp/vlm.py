import itertools
from numba import jit
import numpy as np

from .policy import MultiViewPolicy
from .timer import Timer
from .nbv import get_voxel_at, raycast
from .spatial import SpatialTransform, look_at, ViewHalfSphere
from .perception import UniformTSDFVolume 
from .bbox import AABBox
from vmf_contact_main.train import main_module, parse_args_from_yaml
from scipy.spatial.transform import Rotation
import os
from lang_sam import LangSAM

use_langsam = False

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3

class VLMPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.min_z_dist = .3
        self.max_views = 80
        self.min_gain = 10
        self.downsample = 10
        self.langsam_model = LangSAM() if use_langsam else None

    def activate(self, bbox, intrinsic, pcd_shift):
        self.intrinsic = intrinsic
        self.bbox = bbox
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.agent.grasp_buffer
        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.info = {}
        self.pcd_shift = pcd_shift
    
    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            
    def generate_view(self, img, depth, pose: SpatialTransform):
        # Extract the current pose
        current_position = pose.translation
        current_orientation:Rotation = pose.rotation

        # Convert quaternion to rotation matrix
        rotation_matrix = current_orientation.as_matrix()

        # TODO: add vlm inference
        # camera_pos_increment, gaze_point = self.vlm_agent(rgb, d)
        # camera_pos = camera_pos + camera_pos_increment * 0.1
        
        gaze_point_robot = [-0.74, 0.1, 0.031] # TODO: remove this line, this is a test for gazing at middle of the desk
        self.update_viewsphere(gaze_point_robot)
        
        # Compute the transformation from the current position to the gaze point
        up = np.r_[-1.0, 0.0, 0.0]
        transformation: SpatialTransform = look_at(current_position, gaze_point_robot, up)

        return transformation, 1000

    def update(self, img, depth, pcd_raw, pcd_from_prompt, x):
        
        # TODO: add criteria for grasp execution
        # Process the point cloud
        pcd = (pcd_raw - self.pcd_shift)
        pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        pcd = pcd[(pcd[:, 2] > 0.03) & (pcd[:, 2] < 0.45)]

        print("Processed point cloud: ", pcd.shape)

        # Process the prompt point cloud
        if self.langsam_model is not None:
            pcd_from_prompt = []
            for label in self.masked_pcd_dict:
                pcd_from_prompt.append(self.masked_pcd_dict[label])
                break
            pcd_from_prompt = np.concatenate(pcd_from_prompt, axis=0)
            pcd_from_prompt = (pcd_from_prompt - self.pcd_shift)
            print("Prompt point cloud: ", pcd_from_prompt.shape)
        else:
            pcd_from_prompt = None
            
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("grasp_prediction"):
                self.best_grasp = self.agent.inference(pcd, 
                                        pcd_from_prompt=pcd_from_prompt,
                                        shift=self.pcd_shift,
                                        graspness_th=0.6)
            with Timer("view_generation"):
                views, grasp_criterien = self.generate_view(img, depth, x)
            # with Timer("ig_computation"):
            #     gains = [self.ig_fn(v, self.downsample) for v in views]
            with Timer("cost_computation"):
                costs = [self.cost_fn(v) for v in views]
            # utilities = gains / np.sum(gains) - costs / np.sum(costs)
            # i = np.argmax(utilities)
            # nbv, gain = views[i], gains[i]
            nbv, gain = views, 1000

            print("NBV: ", nbv.translation, "Gain: ", gain)

            if gain < self.min_gain and len(self.views) > self.T:
                self.done = True
            
            self.x_d = nbv

    def best_grasp_prediction_is_stable(self):
        if self.best_grasp:
            t = (self.T_task_base * self.best_grasp.pose).translation
            i, j, k = (t / self.tsdf.voxel_size).astype(int)
            qs = self.qual_hist[:, i, j, k]
            if np.count_nonzero(qs) == self.T and np.mean(qs) > 0.9:
                return True
        return False

    def generate_views(self, q):
        thetas = np.deg2rad([15, 30])
        phis = np.arange(8) * np.deg2rad(45)
        view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            # TODO: check if the view is reachable
            view_candidates.append(view)
        return view_candidates

    def ig_fn(self, view, downsample):
        tsdf_grid, voxel_size = self.tsdf.get_grid(), self.tsdf.voxel_size
        tsdf_grid = -1.0 + 2.0 * tsdf_grid  # Open3D maps tsdf to [0,1]

        # Downsample the sensor resolution
        fx = self.intrinsic.fx / downsample
        fy = self.intrinsic.fy / downsample
        cx = self.intrinsic.cx / downsample
        cy = self.intrinsic.cy / downsample

        # Project bbox onto the image plane to get better bounds
        T_cam_base = view.inv()
        corners = np.array([T_cam_base.apply(p) for p in self.bbox.corners]).T
        u = (fx * corners[0] / corners[2] + cx).round().astype(int)
        v = (fy * corners[1] / corners[2] + cy).round().astype(int)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        t_min = 0.0  # self.min_z_dist
        t_max = corners[2].max()  # This bound might be a bit too short
        t_step = np.sqrt(3) * voxel_size  # Could be replaced with line rasterization

        # Cast rays from the camera view (we'll work in the task frame from now on)
        view = self.T_task_base * view
        ori, pos = view.rotation.as_matrix(), view.translation

        voxel_indices = raycast(
            voxel_size,
            tsdf_grid,
            ori,
            pos,
            fx,
            fy,
            cx,
            cy,
            u_min,
            u_max,
            v_min,
            v_max,
            t_min,
            t_max,
            t_step,
        )

        # Count rear side voxels within the bounding box
        indices = np.unique(voxel_indices, axis=0)
        bbox_min = self.T_task_base.apply(self.bbox.min) / voxel_size
        bbox_max = self.T_task_base.apply(self.bbox.max) / voxel_size
        mask = np.array([((i > bbox_min) & (i < bbox_max)).all() for i in indices])
        i, j, k = indices[mask].T
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > -1.0, tsdfs < 0.0).sum()
        
        return ig

    def cost_fn(self, view):
        return 1.0
