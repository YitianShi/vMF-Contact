import threading
from .inference_node_base import *
import time
import rclpy
import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Transform
from vmf_contact_main.train import main_module, parse_args_from_yaml
from cv_bridge import CvBridge
import os, torch
import numpy as np
import copy
from tf_transformations import quaternion_from_matrix, translation_from_matrix
from PIL import Image
from ros2_nodes.utils_camera import *
import time

O_SIZE = .3

class AIRNodevMF(AIRNode):

    def __init__(self):
        super().__init__(use_langsam=False)

        self.pcd_shift=np.array([-0.86, 0.1, 0.0])
        self.pcd_center = list_to_pose_stamped(self.pcd_shift.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", self.pcd_center)

        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
        self.agent = main_module(parse_args_from_yaml(current_file_folder + "/../vmf_contact_main/config.yaml"), learning=False)
        self.set_vel_acc(.3, .1)

    
    def process_point_cloud(self):
        if self.last_point_cloud_msg is None:
            print("No point cloud message received yet.")
            return None, False

        if self.last_image_msg is None:
            print("No image message received yet.")
            return None, False
    
        if self.last_depth_msg is None:
            print("No depth message received yet.")
            return None, False

        if self.last_point_cloud_msg is None:
            print("No camera info message received yet.")
            return None, False
        
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
        from_frame_rel = "base_link"
        to_frame_rel = pcd_msg_camera.header.frame_id
        t_base_link_2_camera = self.tf_buffer.lookup_transform(
                from_frame_rel, to_frame_rel, rclpy.time.Time()
            )
        
        # pcd_msg_base_link_1 = do_transform_cloud(pcd_msg_camera, t_base_link_2_camera)        
        # pcd_numpy_base_link_1 = read_points_numpy(pcd_msg_base_link_1)
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_base_link_2_camera.transform)  
        # print("First 10 points: ", pcd_numpy_base_link[:10])

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_numpy_base_link_1)
        # pcd_new = o3d.geometry.PointCloud()
        # pcd_new.points = o3d.utility.Vector3dVector(pcd_numpy_base_link)
        # o3d.visualization.draw_geometries([pcd, pcd_new])
        
        ## save the depth image with gray scale, normalized to 0-255
        # pcd_from_depth_z = pcd_from_depth[..., 2]
        # pcd_from_depth_z = (pcd_from_depth_z - pcd_from_depth_z.min()) / (pcd_from_depth_z.max() - pcd_from_depth_z.min()) * 255
        # cv2.imwrite(f"{current_file_folder}/depth.jpg", pcd_from_depth_z)

        return pcd_numpy_base_link, True
    
    
    def handle_user_input(self):
        while True:
            self.to_camera_ready_pose()
            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                (pcd, rgb, d, cam_pose, _, _), identifier = self.process_point_cloud_and_rgbd()
                if not identifier:
                    print("No object detected, please try again.")
                    continue

                grasp, grasp_criterien = self.agent_inference(pcd)
                if grasp is not None:
                    pose = self.process_grasp(grasp)
                    success = self.execute_grasp(pose)
                else:
                    print("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break

    def process_grasp(self, pose):
        pose = list_to_pose(pose)
        return pose
    
    def agent_inference(self, pcd_raw):
        # TODO: add criteria for grasp execution
        # Process the point cloud
        self.pcd_shift = pcd_raw.mean(axis=0)
        self.pcd_shift[2] = 0.0
        pcd = (pcd_raw - self.pcd_shift) 
        pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        pcd = pcd[(pcd[:, 2] > 0.03) & (pcd[:, 2] < 0.45)]

        print("Processed point cloud: ", pcd.shape)

        # # Viszualize the point cloud
        # o3d_pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd)
        # )
        # # draw  the origin as a red sphere
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0]
        # )
        # o3d.visualization.draw_geometries([o3d_pcd, mesh_frame])

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

        # draw the point cloud and prompt point cloud
        # o3d_pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd)
        # )
        # o3d_pcd_from_prompt = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd_from_prompt)
        # )
        # # draw  the origin as a red sphere
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0]
        # )
        # o3d.visualization.draw_geometries([o3d_pcd_from_prompt, mesh_frame])


        # inference
        pose_chosen = self.agent.inference(pcd, 
                                        pcd_from_prompt=pcd_from_prompt,
                                        shift=self.pcd_shift,
                                        graspness_th=0.7,
                                        grasp_height_th = 5e-3,
                                        fused_pose=True)
        # Add the new geometry for the current frame

        if pose_chosen is None:
            return None, False
        # translate back to the original coordinate system
        print("Chosen pose: ", pose_chosen)
        # pose_chosen[:3, 3] = pose_chosen[:3, 3] * self.pcd_resize + self.pcd_shift
        # Visualize the poses
        # frames = []

        # for i in range(poses.shape[0]):
        #     # Extract the transformation matrix for the ith pose
        #     pose_matrix = poses[i]

        #     # Create a coordinate frame at this pose
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)  # Size can be adjusted
        #     frame.transform(pose_matrix)  # Apply the transformation matrix to the frame

        #     # Add the frame to the list of frames
        #     frames.append(frame)
        #     break

        # pcd_raw = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd_raw)
        # )stop_eventation matrix: ", pose_chosen[:3, :3])

        # Convert the rotation matrix to a quaternion
        quat = quaternion_from_matrix(pose_chosen)
        translation = translation_from_matrix(pose_chosen)
        # print("Chosen quaternion: ", quat)

        pose_chosen = np.concatenate([translation, quat])
        print("Chosen pose: ", pose_chosen)

        return pose_chosen, False


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodevMF()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



