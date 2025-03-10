import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from tf_transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from active_grasp.spatial import SpatialTransform
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import cv2
import os
from sensor_msgs.msg import Image
import datetime
import pyrealsense2 as rs

HOME = str(os.path.expanduser('~'))

# Function to mark duplicates with a number
def mark_duplicates(labels):
    label_count = {}
    result = []
    
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        result.append(f"{label}_{label_count[label]}")
    
    return result

def swap_z(quaternion):
    """
    Apply a 90-degree rotation around the Z-axis in the body's local coordinate system.

    Parameters:
        quaternion (tuple): A quaternion in (x, y, z, w) format.

    Returns:
        tuple: The new quaternion after applying the body transformation in (x, y, z, w) format.
    """
    # Define a 90-degree rotation around the Z-axis in quaternion form (x, y, z, w)
    angle = np.pi / 2  # 90 degrees in radians
    rotation_z_90 = R.from_euler('z', angle).as_quat()  # Returns (x, y, z, w)

    # Perform quaternion multiplication (body transformation)
    q_result = R.from_quat(quaternion) * R.from_quat(rotation_z_90)

    # Convert back to (x, y, z, w) format
    return q_result.as_quat()


def pose_from_spacial_transform(spacial_transform: SpatialTransform) -> Pose:
    pose = Pose()
    pose.position.x = spacial_transform.translation[0]
    pose.position.y = spacial_transform.translation[1]
    pose.position.z = spacial_transform.translation[2]
    quat = spacial_transform.rotation.as_quat()
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose

def translate_pose(self, pose_chosen):

    # Convert the rotation matrix to a quaternion
    quat = quaternion_from_matrix(pose_chosen)
    translation = translation_from_matrix(pose_chosen)
    # print("Chosen quaternion: ", quat)

    pose_chosen = np.concatenate([translation, quat])
    print("Chosen pose: ", pose_chosen)

    return pose_chosen, False
    
def look_at_transformation(gaze_point, robot_position):
    """
    Compute a transformation matrix that aligns the robot's orientation to look at a gaze point.
    
    :param gaze_point: (x, y, z) coordinates of the gaze target in the world frame
    :param robot_position: (x, y, z) coordinates of the robot's reference point (e.g., end-effector or camera)
    :return: (position, quaternion) representing the pose
    """
    gaze_point = np.array(gaze_point)
    robot_position = np.array(robot_position)

    # Compute direction vector from robot to gaze point
    direction = gaze_point - robot_position
    direction /= np.linalg.norm(direction)  # Normalize

    # Define a reference up vector (assuming Z-up world frame)
    left_vector = np.array([0, -1, 0])

    # Compute right vector (cross product of up and direction)
    up_vector = np.cross(left_vector, direction)
    up_vector /= np.linalg.norm(up_vector)

    # Compute new up vector (orthogonal to both direction and right)
    right_vector = np.cross(up_vector, direction)

    # Construct rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right_vector
    rotation_matrix[:3, 1] = up_vector
    rotation_matrix[:3, 2] = direction
    rotation_matrix[:3, 3] = robot_position  # Set translation

    # Convert rotation matrix to quaternion
    quaternion = quaternion_from_matrix(rotation_matrix)

    return list(quaternion)

def series_to_pose(series):
    pose = Pose()
    pose.position.x = series[0]
    pose.position.y = series[1]
    pose.position.z = series[2]
    pose.orientation.x = series[3]
    pose.orientation.y = series[4]
    pose.orientation.z = series[5]
    pose.orientation.w = series[6]
    return pose

def pose_to_transform(pose: Pose, header = None) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = header
    tf.transform.translation.x = pose.position.x
    tf.transform.translation.y = pose.position.y
    tf.transform.translation.z = pose.position.z
    tf.transform.rotation.x = pose.orientation.x
    tf.transform.rotation.y = pose.orientation.y
    tf.transform.rotation.z = pose.orientation.z
    tf.transform.rotation.w = pose.orientation.w
    return tf

def transform_to_pose(tf: Transform) -> Pose:
    pose = Pose()
    pose.position.x = tf.translation.x
    pose.position.y = tf.translation.y
    pose.position.z = tf.translation.z
    pose.orientation.x = tf.rotation.x
    pose.orientation.y = tf.rotation.y
    pose.orientation.z = tf.rotation.z
    pose.orientation.w = tf.rotation.w
    return pose

def apply_transform_to_pose(pose, transform):
    """
    Apply a 6-element positional and angular transform (rotation vector) to a geometry_msgs.msg.Pose.

    Args:
        pose (Pose): The input pose.
        transform (list): A 6-element list [dx, dy, dz, rx, ry, rz] where
                          dx, dy, dz are positional changes,
                          rx, ry, rz are the angular changes as a rotation vector.

    Returns:
        Pose: The transformed pose.
    """
    if len(transform) != 6:
        raise ValueError("Transform must be a 6-element list: [dx, dy, dz, rx, ry, rz]")

    # Extract position and rotation vector changes
    dx, dy, dz, rx, ry, rz = transform

    # Apply positional changes
    pose.position.x += dx
    pose.position.y += dy
    pose.position.z += dz

    # Convert current pose quaternion to rotation matrix
    current_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    current_rot = R.from_quat(current_quat)

    # Convert rotation vector to rotation matrix
    delta_rot = R.from_rotvec([rx, ry, rz])

    # Apply the rotation change by combining rotations
    new_rot = delta_rot * current_rot 

    # Convert new rotation back to quaternion
    new_quat = new_rot.as_quat()
    
    # Update the pose with the new quaternion
    pose.orientation.x = new_quat[0]
    pose.orientation.y = new_quat[1]
    pose.orientation.z = new_quat[2]
    pose.orientation.w = new_quat[3]

    return pose

def transform_to_matrix(tf: Transform) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
    matrix[:3, :3] = quaternion_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])[:3, :3]
    return matrix

def pose_stamped_from_pose(pose: Pose, frame_id: str) -> PoseStamped:
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose = pose
    return pose_stamped

def azi_to_pos(azimuth, elevation, distance):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    return [x, y, z]

def pos_to_azi_elev(position, distance=0.47):
    # Convert 3D position to azimuth and elevation
    # position: list or array containing [x, y, z] coordinates
    # distance: the distance from the camera to the object
    # return: azimuth in range [-90, 90] and elevation in range [0, 180] in degrees

    x, y, z = position

    # Compute the azimuth angle in the range [-180, 180]
    azimuth = np.rad2deg(np.arctan2(y, x))

    # Compute the elevation angle (range [0, 90])
    elevation = np.rad2deg(np.arcsin(z / distance))  # arccos ensures output in [0, 180]

    return azimuth, elevation

def pose_to_transform(pose: Pose, header = None) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = header
    tf.transform.translation.x = pose.position.x
    tf.transform.translation.y = pose.position.y
    tf.transform.translation.z = pose.position.z
    tf.transform.rotation.x = pose.orientation.x
    tf.transform.rotation.y = pose.orientation.y
    tf.transform.rotation.z = pose.orientation.z
    tf.transform.rotation.w = pose.orientation.w
    return tf

def list_to_pose(pose_list: list) -> Pose:
    pose = Pose()
    pose.position.x = pose_list[0]
    pose.position.y = pose_list[1]
    pose.position.z = pose_list[2]
    pose.orientation.x = pose_list[3]
    pose.orientation.y = pose_list[4]
    pose.orientation.z = pose_list[5]
    pose.orientation.w = pose_list[6]
    return pose

def list_to_pose_stamped(pose_list: list, frame_id: str) -> PoseStamped:
    pose = list_to_pose(pose_list)
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose = pose
    return pose_stamped

def transform_to_pose(tf: Transform) -> Pose:
    pose = Pose()
    pose.position.x = tf.translation.x
    pose.position.y = tf.translation.y
    pose.position.z = tf.translation.z
    pose.orientation.x = tf.rotation.x
    pose.orientation.y = tf.rotation.y
    pose.orientation.z = tf.rotation.z
    pose.orientation.w = tf.rotation.w
    return pose

def pose_stamped_from_pose(pose_in: Pose, frame_id: str) -> PoseStamped:

    if not isinstance(pose_in, Pose):
        pose = transform_to_pose(pose_in)
    else:
        pose = pose_in
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose = pose
    return pose_stamped

def transform_points(
        point_cloud: np.ndarray,
        transform: Transform) -> np.ndarray:
    """
    Transform a bulk of points from an numpy array using a provided `Transform`.

    :param point_cloud: nx3 Array of points where n is the number of points
    :param transform: TF2 transform used for the transformation
    :returns: Array with the same shape as the input array, but with the transformation applied
    """
    # Build affine transformation
    transform_translation = np.array([
        transform.translation.x,
        transform.translation.y,
        transform.translation.z
    ])
    transform_rotation_matrix = _get_mat_from_quat(
        np.array([
            transform.rotation.w,
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z
        ]))

    # "Batched" matmul meaning a matmul for each point
    # First we offset all points by the translation part
    # followed by a rotation using the rotation matrix
    return np.einsum(
        'ij, pj -> pi',
        transform_rotation_matrix,
        point_cloud) + transform_translation

def _get_mat_from_quat(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.

    This method is currently needed because transforms3d is not released as a `.dep` and
    would require user interaction to set up.

    For reference see: https://github.com/matthew-brett/transforms3d/blob/
    f185e866ecccb66c545559bc9f2e19cb5025e0ab/transforms3d/quaternions.py#L101

    :param quaternion: A numpy array containing the w, x, y, and z components of the quaternion
    :returns: An array containing an X, Y, and Z translation component
    """
    Nq = np.sum(np.square(quaternion))
    if Nq < np.finfo(np.float64).eps:
        return np.eye(3)

    XYZ = quaternion[1:] * 2.0 / Nq
    wXYZ = XYZ * quaternion[0]
    xXYZ = XYZ * quaternion[1]
    yYZ = XYZ[1:] * quaternion[2]
    zZ = XYZ[2] * quaternion[3]

    return np.array(
        [[1.0-(yYZ[0]+zZ), xXYZ[1]-wXYZ[2], xXYZ[2]+wXYZ[1]],
         [xXYZ[1]+wXYZ[2], 1.0-(xXYZ[0]+zZ), yYZ[1]-wXYZ[0]],
         [xXYZ[2]-wXYZ[1], yYZ[1]+wXYZ[0], 1.0-(xXYZ[0]+yYZ[0])]])

def color_image_callback(msg, video_writer_info):
    try:
        # Convert ROS2 Image message to OpenCV format
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if frame.shape[2] == 3:  # If it's a color image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (640, 480))
        # Initialize VideoWriter if not already initialized
        if video_writer_info["writer"] is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MP4
            video_writer_info["writer"] = cv2.VideoWriter(
                video_writer_info["output_file"], fourcc, video_writer_info["fps"], (640, 480)
            )

        # Write frame to video
        video_writer_info["writer"].write(frame_resized)

    except Exception as e:
        print(f"Error processing frame: {e}")

def depth_image_callback(msg, video_writer_info):
    try:
        # Convert ROS2 Image message (16-bit depth) to NumPy array
        frame = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))

        # Normalize depth values to 8-bit range (0-255)
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit grayscale
        frame_8bit = np.uint8(frame_normalized)

        frame_resized = cv2.resize(frame_8bit, (640, 480))

        if video_writer_info["writer"] is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer_info["writer"] = cv2.VideoWriter(
                video_writer_info["output_file"], fourcc, video_writer_info["fps"], (640, 480), isColor=False
            )
        video_writer_info["writer"].write(frame_resized)

    except Exception as e:
        print(f"Error processing depth frame: {e}")

def record_orbbec_video(node, topic_name="color", fps=30):

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{HOME}/Desktop/Orbbec_{topic_name}_{start_time}.avi"

    # Dictionary to store video writer info
    video_writer_info = {"writer": None, "output_file": filename, "fps": fps}

    if topic_name == "color":
        # Subscribe to the image topic
        subscription = node.create_subscription(
            Image,
            "/camera/" + topic_name + "/image_raw",
            lambda msg: color_image_callback(msg, video_writer_info),
            10
        )
        print(f"Recording video from ROS 2 topic: {topic_name}, saving to {filename}...")
    elif topic_name == "depth":
        # Subscribe to the depth image topic
        subscription = node.create_subscription(
            Image,
            "/camera/" + topic_name + "/image_raw",
            lambda msg: depth_image_callback(msg, video_writer_info),
            10
        )
        print(f"Recording depth video from ROS 2 topic: {topic_name}, saving to {filename}...")

def record_realsense_video():

    # Initialize RealSense pipelines for multiple cameras
    pipelines = []
    video_writers = []
    serials = []

    # Get a list of connected devices
    context = rs.context()
    devices = context.query_devices()

    if not devices:
        raise RuntimeError("No RealSense cameras found.")

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        serials.append(serial)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines.append(pipeline)

        # Generate video filename based on device serial
        video_filename = f"{HOME}/Desktop/Realsense_{start_time}.avi"

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        video_writers.append(out)

    try:
        while True:
            for i, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert image to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Check if the file has been removed and recreate it
                if not os.path.exists(video_filename):
                    video_writers[i].release()
                    video_writers[i] = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

                # Write the frame to the video file
                video_writers[i].write(color_image)

                # # Show the frame (optional)
                # cv2.imshow(f'RealSense Video {i}', color_image)

    finally:
        # Stop recording
        for out in video_writers:
            out.release()
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Recording stopped.")


