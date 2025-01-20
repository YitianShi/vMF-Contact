import numpy as np
from sensor_msgs.msg import CameraInfo
from pathlib import Path

from .perception import UniformTSDFVolume
from .spatial import *
from .timer import Timer
from rclpy.node import Node
from dataclasses import dataclass
from .networks import VGN
import open3d as o3d

class Policy():
    def __init__(self):
        self.load_parameters()
        self.shutdown = False

    def load_parameters(self):
        self.qual_thresh = .9

    def activate(self, bbox, view_sphere, intrinsic):
        self.intrinsic = intrinsic

        self.bbox = bbox
        self.view_sphere = view_sphere

        self.calibrate_task_frame()
        self.tsdf = UniformTSDFVolume(0.3, 40)
        model_path = "/home/yitian/GIGA/data/models/vgn_conv.pth" 

        self.vgn = VGN(Path(model_path))

        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.info = {}
    
    def calibrate_task_frame(self):
        xyz = np.r_[self.bbox.center[:2] - 0.15, self.bbox.min[2] - 0.05]
        self.T_base_task = SpatialTransform.from_translation(xyz)
        self.T_task_base = self.T_base_task.inv()

    def update(self, img, x, q):
        raise NotImplementedError

    def filter_grasps(self, out, q):
        grasps, qualities = select_local_maxima(
            self.tsdf.voxel_size,
            out,
            self.qual_thresh,
        )
        filtered_grasps, filtered_qualities = [], []
        for grasp, quality in zip(grasps, qualities):
            pose = self.T_base_task * grasp.pose
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if self.bbox.is_inside(tip):
                grasp.pose = pose
                filtered_grasps.append(grasp)
                filtered_qualities.append(quality)
        return filtered_grasps, filtered_qualities



def select_best_grasp(grasps, qualities):
    i = np.argmax(qualities)
    return grasps[i], qualities[i]


class SingleViewPolicy(Policy):
    def update(self, img, x, q):
        linear, _ = compute_error(self.x_d, x)
        if np.linalg.norm(linear) < 0.02:
            self.views.append(x)
            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)
            tsdf_grid = self.tsdf.get_grid()

            scene_cloud = self.tsdf.get_map_cloud()
            o3d.visualization.draw_geometries([scene_cloud])
            out = self.vgn(tsdf_grid)

            grasps, qualities = self.filter_grasps(out, q)

            if len(grasps) > 0:
                self.best_grasp, quality = select_best_grasp(grasps, qualities)

            self.done = True


class MultiViewPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.T = 12

    def activate(self, bbox, view_sphere, intrinsic):
        self.intrinsic = intrinsic
        super().activate(bbox, view_sphere, intrinsic)
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)

    def integrate(self, img, x, q):
        if not isinstance(x, SpatialTransform):
            x = SpatialTransform.from_matrix(x)
        self.views.append(x)

        with Timer("tsdf_integration"):
            self.tsdf.integrate(img, self.intrinsic, x.inv() * self.T_base_task)

        scene_cloud = self.tsdf.get_map_cloud()
        o3d.visualization.draw_geometries([scene_cloud])

        with Timer("grasp_prediction"):
            tsdf_grid = self.tsdf.get_grid()
            out = self.vgn.predict(tsdf_grid)

        t = (len(self.views) - 1) % self.T
        self.qual_hist[t, ...] = out.qual

        with Timer("grasp_selection"):
            grasps, qualities = self.filter_grasps(out, q)

        if len(grasps) > 0:
            self.best_grasp, quality = select_best_grasp(grasps, qualities)
        else:
            self.best_grasp = None

def compute_error(x_d, x):
    linear = x_d.translation - x.translation
    angular = (x_d.rotation * x.rotation.inv()).as_rotvec()
    return linear, angular


registry = {}


def register(id, cls):
    global registry
    registry[id] = cls


def make(id, *args, **kwargs):
    if id in registry:
        return registry[id](*args, **kwargs)
    else:
        raise ValueError("{} policy does not exist.".format(id))
