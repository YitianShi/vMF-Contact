import threading
import rclpy
import numpy as np
#import spatialmath as sm
from .utils_node import *

from .inference_node_base import *

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.policy import make, registry
from vmf_contact_main.active_grasp.bbox import AABBox
from vmf_contact_main.active_grasp.spatial import *
from vmf_contact_main.active_grasp.timer import Timer

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

class AIRNodeVLM(AIRNode):

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
            self.change_state_to_cartesian_ctl()
            self.to_camera_ready_pose()

            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                self.set_eelink("camera_color_optical_frame")
                # Initialize the search policy
                self.view_sphere = ViewHalfSphere(self.bbox, min_z_dist)
                self.policy.activate(self.bbox, self.intrinsics, self.pcd_shift)
                
                self.create_timer(1.0 / control_rate, self.send_vel_cmd)
                self.change_state_to_servo_ctl()
                self.rate = self.create_rate(policy_rate)

                with Timer("Search time"):
                    self.get_logger().info("Searching for grasp...")
            
                    while not self.policy.done:
                        (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                        if not identifier:
                            self.get_logger().info("No object detected, please try again.")
                            continue
                        self.policy.update(rgb, d, pcd, cam_pose)
                        self.rate.sleep()

                self.rate.sleep()
                grasp = self.policy.best_grasp
                self.change_state_to_cartesian_ctl()
                self.set_eelink("tcp")

                if grasp is not None:
                    self.get_logger().info("Search policy done, start grasp execution.")
                    with Timer("Grasp execution"):
                        success = self.execute_grasp(grasp.pose)
                else:
                    self.get_logger().info("Aborted grasp execution.")
                    success = False
            elif user_input == "q":
                self.shutdown = True
                self.change_state_to_cartesian_ctl()
                break

    def send_vel_cmd(self):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(self.policy.x_d, x, linear_vel=linear_vel, angular_vel=angular_vel) 

            # publish the next view
            pose_robot_2_camera_next: Pose = pose_from_spacial_transform(self.policy.x_d)
            self.publish_new_frame("camera_target_view", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link"))

            # publish the view velocity
            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd) 
            self.publish_new_frame(f"camera_view_velocity", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link")) 
        self.send_twist_cmd(cmd)

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
    
    def generate_trajectory(self):
        traj = []
        gaze_point_robot = [-0.74, 0.1, 0.031] # TODO: remove this line, this is a test for gazing at middle of the desk
        dist = .47 # np.linalg.norm(np.array(gaze_point_robot) - np.array(pos))
        azi_ele_groups = [
            [(-90, -50), (30, 155)], 
            [(-50, 0), (50, 135)],
            [(0, 50), (50, 135)],
            [(50, 90), (30, 155)]
            ]
        
        round = 0
        for azi_ele in azi_ele_groups:
            for azimuth in range(*azi_ele[0], 5):
                t = []            
                ele_range = range(*azi_ele[1]) if round % 2 == 0 else range(*azi_ele[1])[::-1]
                for elevation in ele_range:
                    pos = azi_to_pos(azimuth, elevation, dist)
                    pos =[pos[i] + gaze_point_robot[i] for i in range(3)]
                    quaternion = look_at_transformation(gaze_point_robot, pos)
                    t.append(pos + quaternion)
                round += 1
                traj.append(self.create_trajectory(t))

        for t in traj:
            while not self.movement_finished_flag.is_set():
                camera_data, is_data = self.process_point_cloud_and_rgbd(save_data=True)
                # if is_data:
                #     pcd = camera_data[0]
                #     grasp, grasp_criterien = self.agent_inference(pcd)
                pass
            print("Sending trajectory")
            self.movement_finished_flag.clear()
            self.send_goal_traj(t)            
            

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodeVLM()
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

