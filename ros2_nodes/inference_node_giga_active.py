import threading
import rclpy
import os
import numpy as np
#import spatialmath as sm
from .utils_node import *

from vgn.grasp import *

from .inference_node_base import *

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.policy import make, registry
from vmf_contact_main.active_grasp.bbox import AABBox
from vmf_contact_main.active_grasp.spatial import *
from vmf_contact_main.active_grasp.timer import Timer

import argparse

# from vgn.utils import ros_utils
from pathlib import Path

current_file_folder = os.path.dirname(os.path.abspath(__file__))

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = 0.3
linear_vel = 0.1
angular_vel = 1
control_rate = 30
policy_rate = 4
qual_th = 0.8

class AIRNodeGIGAActive(AIRNode):

    def __init__(self):
        super().__init__()
        
        self.pcd_shift=np.array([-0.86, 0.1, 0.031])
        self.pcd_center = list_to_pose_stamped(self.pcd_shift.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", self.pcd_center)

        lower = [self.pcd_center.pose.position.x - O_SIZE / 2, 
                    self.pcd_center.pose.position.y - O_SIZE / 2, 
                    self.pcd_center.pose.position.z]
        upper = [self.pcd_center.pose.position.x + O_SIZE / 2,
                self.pcd_center.pose.position.y + O_SIZE / 2,
                self.pcd_center.pose.position.z + 0.09]
        
        middle = (np.array(lower) + np.array(upper)) / 2
        self.box_center = list_to_pose_stamped(middle.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("box_center", self.box_center)

        self.bbox: AABBox = AABBox(lower, upper)
        
        # Active search setting
        parser = create_parser()
        args = parser.parse_args()
        self.policy = make(args.policy)
                
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
    
    def handle_user_input(self):

        self.get_camera_info()
        success = False
        while True:
            # Move to the camera ready pose
            self.send_goal(self.camera_ready_pose)
            self.open_gripper()

            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                # Initialize the search policy
                self.view_sphere = ViewHalfSphere(self.bbox, min_z_dist)
                self.policy.activate(self.bbox, self.view_sphere, self.intrinsics)
                
                # execute = threading.Thread(target=self.send_vel_cmd)
                # execute.start()
                # self.create_timer(1.0 / control_rate, self.send_vel_cmd)

                self.rate = self.create_rate(policy_rate)
                with Timer("Search time"):
                    while not self.policy.done:
                        (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                        if not identifier:
                            self.get_logger().info("No object detected, please try again.")
                            continue
                        extrinsic = self.get_extrinsics(inverse=True)[0]
                        self.policy.update(d, extrinsic, self.intrinsics)
                        print("Searching for grasp...")
                        # self.rate.sleep()
                        self.send_vel_cmd()
                    self.rate.sleep()
                    grasp = self.policy.best_grasp
                
                self.get_logger().info("Search policy done, start grasp execution.")
                if grasp is not None:
                    with Timer("Grasp execution"):
                        success = self.execute_grasp(grasp.pose)
                else:
                    self.get_logger().info("Aborted grasp execution.")
                    success = False
            elif user_input == "q":
                self.shutdown = True
                break
    
    def send_vel_cmd(self):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(self.policy.x_d, x, linear_vel=linear_vel, angular_vel=angular_vel)  

        if cmd is not None and not all([cmd[i] == 0 for i in range(6)]):

            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            print("Before move: ", pose_robot_2_camera.position.x, pose_robot_2_camera.position.y, pose_robot_2_camera.position.z)

            pose_robot_2_camera_next: Pose = pose_from_spacial_transform(self.policy.x_d)
            self.publish_new_frame("camera_target_view", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link"))
            # pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd)  
            print("After move: ", pose_robot_2_camera_next.position.x, pose_robot_2_camera_next.position.y, pose_robot_2_camera_next.position.z)     

            try:
                assert self.change_view(pose_robot_2_camera_next)
            except:
                self.get_logger().info("Failed to move the robot to the next view, keep searching...")
                return
            
        # send the velocity command to the robot

    def compute_velocity_cmd(self, x_d, x, linear_vel=0.05, angular_vel=1):
        r, theta, phi = cartesian_to_spherical(x.translation - self.view_sphere.center)
        e_t = x_d.translation - x.translation # translation error
        e_n = (x.translation - self.view_sphere.center) * (self.view_sphere.r - r) / r # pull the camera towards the sphere
        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n # weighted sum of the two errors
        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodeGIGAActive()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=registry.keys(), default="nbv")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser

if __name__ == "__main__":
    main()

