import threading
import open3d as o3d
import rclpy
import numpy as np
#import spatialmath as sm
from .utils_node import *

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.grasp import *
from vgn.perception import create_tsdf

from .inference_node_base import *

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.policy import make, registry
from vmf_contact_main.active_grasp.spatial import *

import argparse

# from vgn.utils import ros_utils
from pathlib import Path

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = 0.3
linear_vel = 0.1
angular_vel = 1
control_rate = 30
policy_rate = 4
qual_th = 0.8

class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class AIRNodeGIGA(AIRNode):

    def __init__(self):
        super().__init__()
        
        self.pcd_shift=np.array([-0.86, 0.1, 0.031])
        self.pcd_center = list_to_pose_stamped(self.pcd_shift.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", self.pcd_center)

        model_type = "vgn" 
        model_path = "/home/yitian/GIGA/data/models/vgn_packed.pt"
        self.origin_giga = list_to_pose_stamped([self.pcd_shift[0] - O_SIZE / 2,
                                                self.pcd_shift[1] - O_SIZE / 2,
                                                self.pcd_shift[2],
                                                0., 0., 0., 1.],
                                                "base_link")
        self.publish_new_frame("origin_giga", self.origin_giga) 

        if model_type == "vgn":
            self.agent = VGN(model_path, 
                                model_type=model_type, 
                                best=True, 
                                force_detection=True, 
                                qual_th=qual_th, 
                                out_th=0.1, 
                                visualize=True)
        elif model_type == "giga":
            self.agent = VGNImplicit(model_path, 
                                        model_type=model_type, 
                                        best=True, 
                                        force_detection=True, 
                                        qual_th=qual_th, 
                                        resolution=O_RESOLUTION, 
                                        voxel_size=O_VOXEL_SIZE, 
                                        out_th=0.1, 
                                        visualize=False)
                
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
  
    def handle_user_input(self):
        self.to_camera_ready_pose()
        self.get_camera_info()
        while True:
            self.send_goal(self.camera_ready_pose)
            self.open_gripper()
            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                if not identifier:
                    continue
                grasp = self.agent_inference(d) 
                if grasp is not None:
                    pose = self.process_grasp(grasp)
                    success = self.execute_grasp(pose, frame="origin_giga")
                    if success:
                        self.get_logger().info("Grasp successful.")
                else:
                    self.get_logger().info("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break
    
    def process_grasp(self, grasp):
        pose = list_to_pose(grasp.tolist())
        pose = pose_stamped_from_pose(pose, "base_link")
        pose = self.transform_pose_z(pose, z_offset=0.05) 
        return pose
            
    def agent_inference(self, depth_imgs):
        if self.last_depth_msg is None or self.camera_matrix is None:
            self.get_logger().info("Missing depth image or camera info.")
            return None
        
        extrinsics = self.get_extrinsics()
        depth_imgs = np.expand_dims(depth_imgs, axis=0)
        tsdf_volume = create_tsdf(O_SIZE, O_RESOLUTION, depth_imgs, self.intrinsics, extrinsics)

        # visualize the TSDF volume
        pcd = tsdf_volume.get_cloud()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, frame])

        state = State(tsdf=tsdf_volume)
        # Perform inference
        grasps, scores, toc, scene = self.agent(state)

        poses = []

        for i, (grasp, score) in enumerate(zip(grasps, scores)):
            pose = np.concatenate([grasp.pose.to_list()[4:],swap_z(grasp.pose.to_list()[:4])])
            poses.append(pose)
            self.publish_new_frame(f"grasp_{i}", list_to_pose_stamped(pose, "origin_giga"))
        scene.show()

        grasp_best = poses[np.argmax(scores)]

        self.get_logger().info(f"Best grasp pose: {grasp_best}")

        return grasp_best

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodeGIGA()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=registry.keys(), default="vlm")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser

if __name__ == "__main__":
    main()

