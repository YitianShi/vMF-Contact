import threading
import rclpy
import numpy as np
#import spatialmath as sm
from .utils_node import *

from .inference_node_base import *

from ros2_nodes.utils_camera import *
from active_grasp.policy import make, registry
from active_grasp.vlm_policy import VLMPolicy
from active_grasp.bbox import AABBox
from active_grasp.spatial import *
from active_grasp.timer import Timer

import argparse
from functools import partial

# from vgn.utils import ros_utils
from pathlib import Path

MOVING_BACK_TO_NBV = "moving_back_to_nbv"

# configurations
O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = .3
linear_vel = .05
angular_vel = 2
control_rate = 10
policy_rate = 4

TARGET_OBJECT = "tennis ball"

# baselines:
INITIAL_VIEW_ONLY = False
TOP_DOWN = False 

WITHOUT_NBV = TOP_DOWN or INITIAL_VIEW_ONLY

# WITHOUT_NBV = True
 
class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class AIRNodeVLM(AIRNode):

    def __init__(self):
        super().__init__()
        self.set_vel_acc(.2, .2)
        self.nbv_memory_pose = None
        if not TOP_DOWN:
            self.camera_ready_pose = list_to_pose_stamped([-0.370, -0.612, 1.224, 0.942, 0.007, -0.005, 0.336], "world")
        
        self.pcd_shift=np.array([-0.73, 0.1, 0.])
        pcd_center = list_to_pose_stamped(self.pcd_shift.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", pcd_center)
        lower = [pcd_center.pose.position.x - O_SIZE / 2, 
                    pcd_center.pose.position.y - O_SIZE / 2, 
                    pcd_center.pose.position.z]
        upper = [pcd_center.pose.position.x + O_SIZE / 2,
                pcd_center.pose.position.y + O_SIZE / 2,
                pcd_center.pose.position.z + 0.09]
        middle = (np.array(lower) + np.array(upper)) / 2
        self.box_center = list_to_pose_stamped(middle.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("box_center", self.box_center)
        self.bbox: AABBox = AABBox(lower, upper)
        self.view_sphere = ViewHalfSphere(self.bbox, min_z_dist)
        
        # Active search setting
        parser = create_parser()
        args = parser.parse_args()
        self.policy:VLMPolicy = make(args.policy, 
                                     target_object=TARGET_OBJECT, 
                                     pcd_center=self.pcd_shift,
                                     min_z_dist=min_z_dist)           
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()

    def memorize_nbv_pose(self):
        # store current position in memory
        self.set_eelink("camera_color_optical_frame")
        time.sleep(0.2)
        self.set_eelink("tcp")
        time.sleep(0.2)
        self.nbv_memory_pose = self.get_current_ee_pose()
        print(f"Memory: {self.nbv_memory_pose}")
    
    def handle_user_input(self):

        # intialize the camera and start capturing sensor data
        self.get_camera_info()
        # self.to_camera_ready_pose()
        self.fetch = threading.Thread(target=self.process_point_cloud_and_rgbd_node)
        self.fetch.start()

        # Initialize the search policy
        success = CLEAR = False
        self.create_timer(1.0 / 30, self.send_vel_cmd2)
        self.create_timer(1.0 / 8, self.grasp_inference)   
        # self.rate = self.create_rate(policy_rate)

        # start recording
        self.record_videos()

        while not CLEAR:
            self.set_eelink(SENSOR_FRAME)
            self.change_state_to_servo_ctl()
            with Timer("Search time"):
                self.get_logger().info("Searching for grasp...")
                self.policy.activate()
        
                while not self.policy.done:
                    if self.input_ready:
                        self.policy.update(self.img, 
                                           self.depth, 
                                           self.pcd, 
                                           self.cam_pose_robot, 
                                           self.azimuth, 
                                           self.elevation)
                    # self.rate.sleep()

            grasp = self.policy.best_grasp
            self.change_state_to_cartesian_ctl()
            time.sleep(.2)
            self.set_eelink("tcp")

            if grasp is not None:
                self.get_logger().info("Search policy done, start grasp execution.")

                self.memorize_nbv_pose()
                with Timer("Grasp execution"):
                    pose = self.process_grasp(grasp)
                    success = self.execute_grasp(pose)
            else:
                self.get_logger().info("Aborted grasp execution.")
                success = False
            
            if success:
                self.get_logger().info("Grasp successful.")
                CLEAR = all([word in self.policy.target_object_curr_label for word in TARGET_OBJECT.split(" ")])
                self.policy.scene_objects.pop(self.policy.target_object_curr.label, None)
            time.sleep(2)
            break

    def process_grasp(self, grasp):
        grasp = grasp.cpu().numpy()
        quat = quaternion_from_matrix(grasp)
        translation = translation_from_matrix(grasp)
        pose_chosen = np.concatenate([translation, quat])
        return list_to_pose(pose_chosen)
    
    def compute_velocity_cmd2(self, x, linear_vel=0.05, angular_vel=1):
        view_focus = self.view_sphere.center
        _, theta, phi = cartesian_to_spherical(x.translation - view_focus)

        # self.logger.info(f"e_t: {e_t}, e_n: {e_n}")

        linear = np.array([0., 0.1, 0.])

        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]
    
    def send_vel_cmd2(self):
        t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", SENSOR_FRAME, rclpy.time.Time()).transform
        x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
        cmd = self.compute_velocity_cmd2(x, linear_vel=linear_vel, angular_vel=angular_vel) 

        # publish the view velocity
        pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
        pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd) 
        self.publish_new_frame(f"camera_view_velocity", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link")) 
        # print(f"cmd: {cmd}")
        self.send_twist_cmd(cmd)
    
    def send_vel_cmd(self):
        if len(self.nbv_fields)==0 or self.policy.done:
            cmd = np.zeros(6)
        elif WITHOUT_NBV:
            cmd = np.zeros(6)
            self.get_logger().info("NBV reached, set velocity to zero")
            self.policy.nbv_reached = True
            self.policy.nbv_fields = []
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", SENSOR_FRAME, rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(x, linear_vel=linear_vel, angular_vel=angular_vel)
            # publish the view velocity
            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd) 
            self.publish_new_frame(f"camera_view_velocity", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link")) 
            # print(f"cmd move scale: {np.linalg.norm(cmd[:3])}")
            # print(f"cmd angle scale: {np.linalg.norm(cmd[3:])}")
            if np.linalg.norm(cmd[:3]) < 1e-3:
                self.get_logger().info("NBV reached, set velocity to zero")
                self.policy.nbv_reached = True
                self.policy.nbv_fields = []
        self.send_twist_cmd(cmd)

    def compute_velocity_cmd(self, x, linear_vel=0.05, angular_vel=1):
        view_focus = self.view_sphere.center
        _, theta, phi = cartesian_to_spherical(x.translation - self.policy.target_object_curr.center)

        e_t = self.policy.query_field_fusion_from_list(x.translation)
        
        r = np.linalg.norm(x.translation - self.pcd_shift)
        e_n = (x.translation - view_focus) * (self.view_sphere.r - r) / r # pull the camera towards the sphere

        # self.logger.info(f"e_t: {e_t}, e_n: {e_n}")

        linear = 1.0 * e_t + 2.0 * (r < self.view_sphere.r) * e_n # weighted sum of the two errors

        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]
    
    def grasp_inference(self, interactive_vis=False):
        #pcd, identifier = self.process_point_cloud_and_rgbd(pcd_only=True)
        if self.input_ready:
            self.policy.update_grasp(self.pcd, interactive_vis)
    

    def execute_grasp(self, grasp_pose:Pose, frame = "base_link"):

        self.change_state_to_cartesian_ctl()
        self.get_logger().info("Sending goal now...")

        # transform posestamped from base_link to world using t_world_2_base_link
        grasp_pose = pose_stamped_from_pose(grasp_pose, "base_link")
        t_world_2_base_link = self.tf_buffer.lookup_transform("world", frame, rclpy.time.Time())
        grasp_pose.pose = tf2_geometry_msgs.do_transform_pose(grasp_pose.pose, t_world_2_base_link)
        grasp_pose.header.frame_id = "world"
        self.publish_new_frame("grasp", grasp_pose)

        pregrasp_pose = self.create_pregrasp_pose(copy.deepcopy(grasp_pose))
        self.publish_new_frame("pregrasp", pregrasp_pose)
        
        # ask if the user wants to continue
        user_input = input("Press 'c' to continue or any other key to quit: ")
        if user_input == "c":

            self.stop_event.clear()
            self.movement_failed_flag.clear()
            self.movement_finished_flag.clear()
            state_machine_state = IDLE
            
            while True:
                if self.stop_event.is_set():
                    self.stop_event.clear()
                    self.get_logger().info("Goal cancelled")
                    self.send_goal(self.nbv_memory_pose)
                    state_machine_state = MOVING_BACK_TO_NBV
                    self.get_logger().info("StateMachine switched to MOVING_BACK_TO_NBV")

                if state_machine_state == IDLE:
                    # Send the goal to move to the pregrasp pose
                    self.send_goal(pregrasp_pose)
                    # Start the state machine
                    state_machine_state = MOVING_TO_PREGRASP
                    self.get_logger().info("StateMachine switched to MOVING_TO_PREGRASP")
                    # Create a thread to handle the input
                    self.cancel_thread = threading.Thread(target=self.get_input)
                    self.cancel_thread.start()

                elif state_machine_state == MOVING_TO_PREGRASP:
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Wait for manual grasp evaluation
                        time.sleep(2)
                        self.send_goal(grasp_pose)
                        # Start the state machine
                        state_machine_state = MOVING_TO_GRASP
                        self.get_logger().info("StateMachine switched to MOVING_TO_GRASP")
                        time.sleep(0.1)

                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                        
                elif state_machine_state == MOVING_TO_GRASP:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        time.sleep(2)
                        # Close the gripper
                        self.close_gripper()
                        state_machine_state = GRASPING
                        self.get_logger().info("StateMachine switched to GRASPING")
                        
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")

                elif state_machine_state == GRASPING:
                    # Wait for the gripper to finish
                    if self.gripper_movement_finished_flag.is_set():
                        self.gripper_movement_finished_flag.clear()
                        # Before gripper moves away, wait for 2 seconds
                        time.sleep(0.25)
                        # Send the goal to move to the pregrasp pose
                        self.send_goal(pregrasp_pose)
                        state_machine_state = MOVING_TO_PREGRASP_RETURN
                        self.get_logger().info("StateMachine switched to MOVING_TO_PREGRASP_RETURN")
                        time.sleep(1)
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                
                elif state_machine_state == MOVING_TO_PREGRASP_RETURN:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Send the goal to move to the drop off pose
                        self.send_goal(self.drop_off_pose)
                        state_machine_state = MOVING_TO_DROP_OFF
                        self.get_logger().info("StateMachine switched to MOVING_TO_DROP_OFF")
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                
                elif state_machine_state == MOVING_TO_DROP_OFF:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Open the gripper
                        self.open_gripper()
                        state_machine_state = RELEASEING
                        self.get_logger().info("StateMachine switched to RELEASEING")
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == RELEASEING:
                    # Wait for the gripper to finish
                    if self.gripper_movement_finished_flag.is_set():
                        self.gripper_movement_finished_flag.clear()
                        # Send the goal to move to the camera ready pose
                        self.send_goal(self.nbv_memory_pose)
                        state_machine_state = MOVING_BACK_TO_NBV
                        self.get_logger().info("StateMachine switched to MOVING_BACK_TO_NBV")
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == MOVING_BACK_TO_NBV:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        state_machine_state = IDLE
                        self.get_logger().info("State machine finished, please enter any key to finish the input thread.")
                        break
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == FAILED:
                    self.get_logger().info("State machine failed")
                    return False
            return False
    
    @property
    def nbv_fields(self):
        return self.policy.nbv_fields
    

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

