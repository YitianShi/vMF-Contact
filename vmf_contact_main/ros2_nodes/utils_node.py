import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from tf_transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from vmf_contact_main.active_grasp.spatial import SpatialTransform
from scipy.spatial.transform import Rotation

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
    current_rot = Rotation.from_quat(current_quat)

    # Convert rotation vector to rotation matrix
    delta_rot = Rotation.from_rotvec([rx, ry, rz])

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

def azi_to_pos(azimuth, elevation, distance):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    return [x, y, z]


