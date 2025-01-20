import threading
import time
import rclpy
import sensor_msgs.msg as sensor_msgs
from rclpy.node import Node
from rclpy.action import ActionClient
from arm_api2_msgs.action import MoveCartesian, MoveCartesianPath
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import StaticTransformBroadcaster
from tf2_sensor_msgs.tf2_sensor_msgs import transform_points
from cv_bridge import CvBridge
import cv2
import os, torch
import numpy as np
import copy
from tf_transformations import quaternion_matrix
import tf2_geometry_msgs
#import spatialmath as sm
from .utils_node import *
from lang_sam import LangSAM
from PIL import Image

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.spatial import *

current_file_folder = os.path.dirname(os.path.abspath(__file__))

#image_pil = Image.open("./assets/car.jpeg").convert("RGB")
#text_prompt = "wheel."
#results = model.predict([image_pil], [text_prompt])

# define state machine states
IDLE = "idle"
MOVING_TO_PREGRASP = "moving_to_pregrasp"
MOVING_TO_GRASP = "moving_to_grasp"
GRASPING = "grasping"
MOVING_TO_DROP_OFF = "moving_to_drop_off"
MOVING_TO_PREGRASP2 = "moving_to_pregrasp2"
RELEASEING = "releasing"
MOVING_TO_PREGRASP_RETURN = "moving_to_pregrasp2"
MOVING_TO_CAMERA_READY = "moving_to_camera_ready"
FAILED = "failed"
MOVING = "moving"

class AIRNode(Node):

    def __init__(self, use_langsam=False):
        super().__init__("pcd_subsriber_node")
        
        self.stitched_pointcloud_topic = "/cloud_stitched"  
        self.grasp_result_topic = "/detect_grasps/clustered_grasps" 

        # Set up a subscription to the 'pcd' topic with a callback to the
        # function `listener_callback`
        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,  # Msg type
            "/camera/depth/points",  # topic
            self.listener_callback_pcd,  # Function to call
            10,  # QoS
        )

        self.img_subscriber = self.create_subscription(
            sensor_msgs.Image,  # Msg type
            "/camera/color/image_raw",  # topic
            self.listener_callback_img,  # Function to call
            10,  # QoS
        )

        self.dpt_subscriber = self.create_subscription(
            sensor_msgs.Image,  # Msg type
            "/camera/depth/image_raw",  # topic
            self.listener_callback_dpt,  # Function to call
            10,  # QoS
        )

        self.camera_info_subscriber = self.create_subscription(
            sensor_msgs.CameraInfo,  # Msg type
            "/camera/color/camera_info",  # topic
            self.listener_callback_caminfo,  # Function to call
            10,  # QoS
        )

        self._robot_action_client = ActionClient(
            self, MoveCartesian, "arm/move_to_pose"
        )
        
        self._gripper_action_client = ActionClient(
            self, GripperCommand, "robotiq_2f_urcap_adapter/gripper_command"
        )
        
        self._robot_action_client_traj = ActionClient(
            self, MoveCartesianPath, "arm/move_to_pose_path"
        )

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Create TF Listener to get the transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_point_cloud_msg = None
        self.shutdown = False
        
        self.movement_finished_flag = threading.Event()
        self.movement_failed_flag = threading.Event()
        self.gripper_movement_finished_flag = threading.Event()
        self.gripper_movement_failed_flag = threading.Event()
        self.stop_event = threading.Event()
        self.send_traj_flag = threading.Event()
        self.bridge = CvBridge()

        self.camera_ready_pose = PoseStamped()
        self.camera_ready_pose.header.frame_id = "world"

        self.camera_ready_pose = list_to_pose_stamped([-0.435, -0.794, 1.381, 0.995, 0.009, 0.005, 0.100], "world") # long finger
        # self.camera_ready_pose = list_to_pose_stamped([-0.435, -0.572, 1.492, 0.995, 0.009, 0.005, 0.100], "world") # small finger
        self.drop_off_pose: PoseStamped = list_to_pose_stamped([0.15, -0.75, 1.3, 1.0, 0.0, 0.0, 0.0], "world")

        self.langsam_model = LangSAM() if use_langsam else None

    def get_camera_info(self):
        while True:
            try:
                print("Waiting for camera info...")
                self.intrinsics= CameraInfo(
                    width=int(self.image_width), 
                    height=int(self.image_height), 
                    fx=self.camera_matrix[0, 0], 
                    fy=self.camera_matrix[1, 1],
                    cx=self.camera_matrix[0, 2], 
                    cy=self.camera_matrix[1, 2], 
                    scale=1.0
                )
                self.get_logger().info(f"Camera info received: {self.intrinsics}")
                break
            except:
                self.get_logger().info("No camera info received yet.")
                time.sleep(1)

    def listener_callback_pcd(self, msg: sensor_msgs.PointCloud2):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_point_cloud_msg = msg
        if self.shutdown:
            raise SystemExit
    
    def listener_callback_img(self, msg: sensor_msgs.Image):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_image_msg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if self.shutdown:
            raise SystemExit
    
    def listener_callback_dpt(self, msg: sensor_msgs.Image):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_depth_msg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1") # / 1000.0
        if self.shutdown:
            raise SystemExit

    def listener_callback_caminfo(self, msg: sensor_msgs.CameraInfo):
        self.distortion = np.array([msg.d[i] for i in range(8)])
        self.camera_matrix = np.array([msg.k[i] for i in range(9)]).reshape(3, 3)
        self.image_width = float(msg.width)
        self.image_height = float(msg.height)
        if self.shutdown:
            raise SystemExit

    def process_point_cloud_and_rgbd(self, save_data=False):
        # TODO: add rgb image processing
        if self.last_point_cloud_msg is None:
            self.get_logger().info("No point cloud message received yet.")
            return [None] * 4, False

        if self.last_image_msg is None:
            self.get_logger().info("No image message received yet.")
            return [None] * 4, False
    
        if self.last_depth_msg is None:
            self.get_logger().info("No depth message received yet.")
            return [None] * 4, False

        if self.last_point_cloud_msg is None:
            self.get_logger().info("No camera info message received yet.")
            return [None] * 4, False
        
        # Get the latest point cloud and image messages
        pcd_msg_camera = self.last_point_cloud_msg
        img = self.last_image_msg

        # transform the point cloud to base_link frame
        camera= CameraInfo(
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        t_robot_2_camera = self.tf_buffer.lookup_transform(
                "base_link", pcd_msg_camera.header.frame_id, rclpy.time.Time()
            )
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_robot_2_camera.transform)  

        # remove previous masks
        for file in os.listdir(current_file_folder):
            if file.endswith(".jpg"):
                os.remove(os.path.join(current_file_folder, file))
        
        # transformation to pose
        cam_pose_robot = transform_to_pose(t_robot_2_camera.transform)

        # remove previous masks
        for file in os.listdir(current_file_folder):
            if file.endswith(".jpg"):
                os.remove(os.path.join(current_file_folder, file))
        
        if self.langsam_model is not None:
            time_curr = time.time()
            prompt_input = ""
            while prompt_input == "":
                prompt_input = input("Please enter what you would like to grasp: ")
            prompt_input = prompt_input.split(".")
            print("Prompt: ", prompt_input)
        
            self.masked_pcd_dict = {}
            # predict masks with lang_sam
            results = self.langsam_model.predict([Image.fromarray(img)], [". ".join(prompt_input)])

            print(f"Time taken for inference: {time.time() - time_curr}")

            print(f"save images to {current_file_folder}")
            cv2.imwrite(f"{current_file_folder}/image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # check if there are labels detected
            labels = results[0]["labels"]
            if len(labels) == 0:
                print("No labels detected.")
                return pcd_numpy_base_link, False
            # check duplicates in labels, if there are duplicates, mark them with a number
            labels = mark_duplicates(labels)
            print("Results: ", labels)
            print("Scores: ", results[0]["scores"])

            # mask point cloud and image
            for i, text in enumerate(labels):
                mask = results[0]["masks"][i].astype(np.uint8)[:, :, None]
                # mask image and point cloud
                pcd_masked = pcd_numpy_base_link[mask.reshape(-1) == 1]
                self.masked_pcd_dict[text] = pcd_masked
                
                # save masked image
                cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
                
                # # visualize the masked point cloud
                # pcd_masked_color = img.reshape(-1, 3)[mask.reshape(-1) == 1]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pcd_masked)
                # pcd.colors = o3d.utility.Vector3dVector(pcd_masked_color[:, ::-1] / 255)
                # o3d.visualization.draw_geometries([pcd], window_name=text)

        # save the image, depth, point cloud and camera pose as a dictionary of numpy arrays
        # viszualize the rgbd data
        if save_data:
            name = f"/home/yitian/data_active_grasp/{cam_pose_robot.position.x}_{cam_pose_robot.position.y}_{cam_pose_robot.position.z}"
            cv2.imwrite(f"image.jpg", img[..., ::-1])
            dict_rgbd = {
                "image": img,
                "depth": self.last_depth_msg,
                "pcd": pcd_numpy_base_link,
                "cam_pose": [cam_pose_robot.position.x, 
                            cam_pose_robot.position.y, 
                            cam_pose_robot.position.z,
                            cam_pose_robot.orientation.x, 
                            cam_pose_robot.orientation.y, 
                            cam_pose_robot.orientation.z, 
                            cam_pose_robot.orientation.w]
            }
            np.savez(f"{name}.npz", **dict_rgbd)

        return (pcd_numpy_base_link, 
                self.last_image_msg, 
                self.last_depth_msg.astype(np.float32) / 1000.0, 
                cam_pose_robot), True         
    
    def handle_user_input(self):
        raise NotImplementedError

    def get_extrinsics(self, inverse=False):
        extrinsics = np.expand_dims(np.eye(4), axis=0)
        # return extrinsics
        if inverse:
            t_robot_2_camera: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "camera_color_optical_frame", rclpy.time.Time()
            )
        else:
            t_robot_2_camera: TransformStamped = self.tf_buffer.lookup_transform(
                "camera_color_optical_frame", "origin_giga", rclpy.time.Time()
            )
        translation = [
            t_robot_2_camera.transform.translation.x,
            t_robot_2_camera.transform.translation.y,
            t_robot_2_camera.transform.translation.z,
        ]

        rotation = [
            t_robot_2_camera.transform.rotation.x,
            t_robot_2_camera.transform.rotation.y,
            t_robot_2_camera.transform.rotation.z,
            t_robot_2_camera.transform.rotation.w,
        ]

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix(rotation)[:3, :3]

        # Construct the transformation matrix
        extrinsics[:, :3, :3] = rotation_matrix
        extrinsics[:, :3, 3] = translation

        # self.get_logger().info(f"Extrinsics: {extrinsics}")

        return  extrinsics
    
    def camera_robot_pose_to_tcp_world_pose(self, pose_robot_2_camera: Pose):
        t_world_2_base_link: TransformStamped = self.tf_buffer.lookup_transform(
                "world", "base_link", rclpy.time.Time()
            )
        t_camera_2_tcp: TransformStamped = self.tf_buffer.lookup_transform(
            "camera_color_optical_frame","tcp", rclpy.time.Time()
        )
        # transform posestamped from base_link to world using t_world_2_base_link
        pose_world_2_camera = tf2_geometry_msgs.do_transform_pose(pose_robot_2_camera, t_world_2_base_link)
        t_world_2_camera = pose_to_transform(pose_world_2_camera, "world")
        pose_camera_2_tcp = transform_to_pose(t_camera_2_tcp.transform)
        pose_world_2_tcp = tf2_geometry_msgs.do_transform_pose(pose_camera_2_tcp, t_world_2_camera)

        pose_stamped = pose_stamped_from_pose(pose_world_2_tcp, "world")
        return pose_stamped, pose_world_2_camera
    
    def change_view(self, pose_robot_2_camera):

        self.get_logger().info("Sending goal now...")

        pose_stamped, pose_world_2_camera = self.camera_robot_pose_to_tcp_world_pose(pose_robot_2_camera)

        self.publish_new_frame(f"camera_target_view_velocity", pose_stamped_from_pose(pose_world_2_camera, "world"))
        
        self.stop_event.clear()
        self.movement_failed_flag.clear()
        self.movement_finished_flag.clear()
        state_machine_state = IDLE
        
        while True:
            if state_machine_state == IDLE:
                # Send the goal to move to the pregrasp pose
                self.send_goal(pose_stamped)
                # Start the state machine
                state_machine_state = MOVING
                self.get_logger().info("StateMachine switched to MOVING")
                # # Create a thread to handle the input
                # self.cancel_thread = threading.Thread(target=self.get_input)
                # self.cancel_thread.start()

            else:
                # Wait for the action server to finish
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()
                    state_machine_state = IDLE
                    self.get_logger().info("State machine finished")
                    break

                if self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("StateMachine switched to FAILED")

                elif state_machine_state == FAILED:
                    self.get_logger().info("State machine failed")
                    return False
        return True

    def execute_grasp(self, pose, frame = "base_link"):

        self.get_logger().info("Sending goal now...")

        if isinstance(pose, np.ndarray):
            pose = list(pose)
            self.get_logger().info(f"Grasp pose: {pose}")
            # pose = pose[4:] + pose[:4]
            pose = list_to_pose(pose)

        elif isinstance(pose, SpatialTransform):
            print(pose.translation)
            pose = pose_from_spacial_transform(pose)

        grasp_pose = pose_stamped_from_pose(pose, frame)

        grasp_pose = self.transform_pose_z(grasp_pose, z_offset=0.04) # GIGA is predicting the position of the finger end, so we need to move it a bit in z direction to the tcp

        
        t_world_2_base_link = self.tf_buffer.lookup_transform(
            "world", frame, rclpy.time.Time()
        )

        # transform posestamped from base_link to world using t_world_2_base_link
        grasp_pose.pose = tf2_geometry_msgs.do_transform_pose(grasp_pose.pose, t_world_2_base_link)
        grasp_pose.header.frame_id = "world"
        
        pregrasp_pose = self.create_pregrasp_pose(copy.deepcopy(grasp_pose))

        self.publish_new_frame("grasp", grasp_pose)
        self.publish_new_frame("pregrasp", pregrasp_pose)
        
        # ask if the user wants to continue
        user_input = input("Press 'c' to continue or any other key to quit: ")
        if user_input.lower() == 'c':

            self.stop_event.clear()
            self.movement_failed_flag.clear()
            self.movement_finished_flag.clear()
            state_machine_state = IDLE
            
            while True:
                if self.stop_event.is_set():
                    self.stop_event.clear()
                    self.get_logger().info("Goal cancelled")
                    self.send_goal(self.camera_ready_pose)
                    state_machine_state = MOVING_TO_CAMERA_READY
                    self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")

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
                        self.send_goal(self.camera_ready_pose)
                        state_machine_state = MOVING_TO_CAMERA_READY
                        self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == MOVING_TO_CAMERA_READY:
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
            return True

    def get_input(self):
        try:
            user_input = input("Press 'c' to cancel: ")
            self.stop_event.set()
            self.get_logger().info(f"\nYou entered: {user_input}")
        except EOFError:
            # Input terminated unexpectedly
            pass

    def cancel_done(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')   
           
    def create_pregrasp_pose(self, grasp_pose: PoseStamped) -> PoseStamped:
        # Create a pregrasp pose by transforming the grasp pose in the z direction
        pregrasp_pose = self.transform_pose_z(grasp_pose, z_offset=-0.1)
        return pregrasp_pose
    
    def transform_pose_z(self, pose_stamped: PoseStamped, z_offset: float) -> PoseStamped:
        # Copy the original pose
        new_pose_stamped = PoseStamped()
        new_pose_stamped.header = pose_stamped.header
        new_pose_stamped.pose = pose_stamped.pose

        # Extract the current pose
        current_position = pose_stamped.pose.position
        current_orientation = pose_stamped.pose.orientation

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix([current_orientation.x, 
                                             current_orientation.y, 
                                             current_orientation.z, 
                                             current_orientation.w])

        # Translation in the local z direction (12 cm = 0.12 meters)
        translation = [0.0, 0.0, z_offset, 1.0]

        # Apply the translation in the local frame
        transformed_translation = rotation_matrix.dot(translation)

        # Update the position with the transformed translation
        new_pose_stamped.pose.position.x = current_position.x + transformed_translation[0]
        new_pose_stamped.pose.position.y = current_position.y + transformed_translation[1]
        new_pose_stamped.pose.position.z = current_position.z + transformed_translation[2]  

        return new_pose_stamped
    
    def publish_new_frame(self, name, pose: PoseStamped):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = pose.header.frame_id
        t.child_frame_id = name
        t.transform.translation.x = pose.pose.position.x
        t.transform.translation.y = pose.pose.position.y
        t.transform.translation.z = pose.pose.position.z
        t.transform.rotation.x = pose.pose.orientation.x
        t.transform.rotation.y = pose.pose.orientation.y
        t.transform.rotation.z = pose.pose.orientation.z
        t.transform.rotation.w = pose.pose.orientation.w

        self.tf_static_broadcaster.sendTransform(t)

    def send_goal(self, goal):
        goal_msg = MoveCartesian.Goal()
        goal_msg.goal = goal

        self.get_logger().info("Waiting for action server...")

        self._robot_action_client.wait_for_server()

        self.get_logger().info("Sending goal request...")

        self._send_goal_future = self._robot_action_client.send_goal_async(
            goal_msg, feedback_callback=self.robot_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.robot_goal_response_callback)

    def send_goal_traj(self, goal_path):
            goal_msg = MoveCartesianPath.Goal()
            goal_msg.poses = goal_path

            self.get_logger().info("Waiting for action server...")

            self._robot_action_client_traj.wait_for_server()

            self.get_logger().info("Sending goal request...")

            self._send_goal_future = self._robot_action_client_traj.send_goal_async(
                goal_msg, feedback_callback=self.robot_feedback_callback 
            )

            self._send_goal_future.add_done_callback(self.robot_goal_response_callback)

    def robot_goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")

        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robot_get_result_callback)

    def robot_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")
        if result.success:
            self.movement_finished_flag.set()
        else:
            self.movement_failed_flag.set()
    
    def robot_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info("Feedback: {0}".format(feedback.status))
    
    def send_gripper_command(self, command: GripperCommandMsg):
        goal_msg = GripperCommand.Goal()
        goal_msg.command = command

        self.get_logger().info("Waiting for gripper action server...")

        self._gripper_action_client.wait_for_server()

        self.get_logger().info("Sending gripper goal request...")

        self._send_goal_future = self._gripper_action_client.send_goal_async(
            goal_msg, feedback_callback=self.gripper_feedback_callback
        )

        self._send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Gripper goal rejected")
            return

        self.get_logger().info("Gripper goal accepted")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_get_result_callback)

    def gripper_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Gripper result: {result}")
        #if result.success:
        self.gripper_movement_finished_flag.set()
        #else:
         #   self.movement_failed_flag.set()

    def gripper_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Gripper Feedback: {feedback}")
    
    def open_gripper(self):
        self.get_logger().info("Opening gripper...")
        command = GripperCommandMsg()
        command.position = 0.0  # rad
        command.max_effort = 60.0  # N
        self.send_gripper_command(command)

    def close_gripper(self):
        self.get_logger().info("Closing gripper...")
        command = GripperCommandMsg()
        command.position = 0.8  # 0.0 (open) - 0.085 (close) rad
        command.max_effort = 150.0  # 20 N - 235 N
        self.send_gripper_command(command)
    
    def create_trajectory(self, data):
            path = []
            for p in data:
                rosp = PoseStamped()
                cam_pose_to_robot = series_to_pose(p)
                tcp_pose_to_world, _ = self.camera_robot_pose_to_tcp_world_pose(cam_pose_to_robot)
                rosp.header.frame_id = "world"
                path.append(tcp_pose_to_world)
            return path




