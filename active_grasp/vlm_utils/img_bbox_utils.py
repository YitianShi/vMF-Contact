import numpy as np
import open3d as o3d
import cv2
import base64
from io import BytesIO

class SceneObject:
    def __init__(self, pcd, label, adjectives=None, relations=[]):
        """
        :param center: object center (N, 3)
        :param bbox_dims: bbox size (length, width, height)
        :param bbox_3d: 3D bounding box corners, shape (8, 3)
        :param label: object label string
        :param adjectives: list of descriptive adjectives (List[str])
        """
        self.pcd = pcd
        self.label = label
        self.adjectives = adjectives
        self.relations = relations
        self.instance_id = f"{adjectives[0]} {label}" if adjectives is not None else label
        self.center, self.bbox_dims, self.bbox_3d = compute_oriented_bounding_box(pcd)
        
    def __str__(self):
        return (
                # f"Instance ID: {self.instance_id}\n"
                f"Label: {self.label}\n"
                f"Center: {self.center}\n"
                f"BBox size: (length, width, height): {self.bbox_dims}\n"
                f"3D BBox: {self.bbox_3d}\n"
                f"Adjectives: {self.adjectives}\n")
    
    def return_dict(self):
        return{
            "bbox_center": np.round(self.center * np.array([-1, 1, 1]), 3).tolist(),
            "bbox_corners": np.round(self.bbox_3d * np.array([-1, 1, 1]), 3).tolist(),
            "bbox_lwh": np.round(self.bbox_dims, 3).tolist()
        }

def format_bbox_data(bbox_center_dict, bbox_corners_dict, bbox_lwh_dict):
    formatted_data = {}

    for obj_name in bbox_center_dict.keys():
        formatted_data[obj_name] = {
            "bbox_center": np.round(bbox_center_dict[obj_name] * np.array([-1, 1, 1]), 3).tolist(),  # 翻转 X 轴
            "bbox_corners": np.round(bbox_corners_dict[obj_name] * np.array([-1, 1, 1]), 3).tolist(),  # 翻转 X 轴
            "bbox_lwh": np.round(bbox_lwh_dict[obj_name], 3).tolist()
        }
    return formatted_data

####################################################################################################

def compute_oriented_bounding_box(pcd, lower_percentile=4, upper_percentile=96):
    """
    Computes a SceneObject with an Oriented Bounding Box (OBB) from a point cloud.

    Args:
        pcd (np.ndarray): A numpy array of shape (N, 3) representing the point cloud.
        label (str): The label for the object.
        adjectives (list): List of descriptive adjectives (e.g., color, size, properties).
        lower_percentile (float): The lower percentile to clip outliers.
        upper_percentile (float): The upper percentile to clip outliers.

    Returns:
        SceneObject: An object with the computed OBB and other properties.
    """
    # Step 1: Remove outliers using percentile clipping
    lower_bound = np.percentile(pcd, lower_percentile, axis=0)
    upper_bound = np.percentile(pcd, upper_percentile, axis=0)
    
    mask = np.all((pcd >= lower_bound) & (pcd <= upper_bound), axis=1)
    filtered_points = pcd[mask]  # Keep only inliers

    # Step 2: Compute PCA (Covariance matrix)
    mean = np.mean(filtered_points, axis=0)
    centered_points = filtered_points - mean
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Step 3: Eigen decomposition (PCA axes)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # Eigen decomposition
    rotation_matrix = eigenvectors  # The eigenvectors form the rotation matrix

    # Step 4: Transform points to PCA coordinate space
    transformed_points = centered_points @ rotation_matrix

    # Step 5: Get min/max in the rotated frame
    min_corner = np.min(transformed_points, axis=0)
    max_corner = np.max(transformed_points, axis=0)

    # Step 6: Compute OBB center and dimensions
    obb_center = (min_corner + max_corner) / 2
    obb_center_world = obb_center @ rotation_matrix.T + mean  # Transform back
    bbox_dims = max_corner - min_corner  # Dimensions: (length, width, height)

    # Step 7: Compute the 8 corners of the bounding box in the PCA space
    corner_offsets = np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
    ])
    
    # Transform corners back to world space
    obb_corners = (corner_offsets @ rotation_matrix.T) + mean

    # Create and return a SceneObject instance
    return obb_center_world, bbox_dims, obb_corners


def visualize_pcd_with_obb(pcd, obb_corners, pcd_color=None):
    """
    Visualize the point cloud and its Oriented Bounding Box (OBB) in Open3D.
    
    Args:
        pcd (np.ndarray): (N, 3) numpy array containing point cloud data.
        obb_corners (np.ndarray): (8, 3) numpy array containing the 8 OBB corners.
    """
    # Create Open3D Point Cloud
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd)
    if pcd_color is not None:
        pcd_color = pcd_color[None, :].repeat(len(pcd), axis=0).astype(np.float64)  # Broadcast color
        pcd_vis.colors = o3d.utility.Vector3dVector(pcd_color)
    
    # Create Open3D Line Set for OBB
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for bounding box

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(obb_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return [pcd_vis, line_set]

####################################################################################################

def get_edges(corners):
    """Compute edge vectors from 8 corner points of an OBB."""
    edges = [
        corners[1] - corners[0],  # Edge along width
        corners[3] - corners[0],  # Edge along height
        corners[4] - corners[0],  # Edge along depth
    ]
    return np.array(edges)

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def project_obb(corners, axis):
    """Project the 8 corners of an OBB onto a given axis and return min/max projection."""
    projections = np.dot(corners, axis)
    return np.min(projections), np.max(projections)

def overlap(min1, max1, min2, max2):
    """Check if two 1D projections overlap."""
    return max1 >= min2 and max2 >= min1

def obb_collision(obb1, obb2):
    """
    Check if two 3D OBBs collide using the Separating Axis Theorem.

    Parameters:
    obb1, obb2: numpy arrays of shape (8,3), representing the 8 corner points of each OBB.

    Returns:
    True if OBBs collide, False otherwise.
    """
    edges1 = get_edges(obb1)
    edges2 = get_edges(obb2)

    # Compute 15 possible separating axes
    axes = []
    axes.extend([normalize(e) for e in edges1])  # 3 axes of OBB1
    axes.extend([normalize(e) for e in edges2])  # 3 axes of OBB2
    for e1 in edges1:
        for e2 in edges2:
            cross_axis = np.cross(e1, e2)
            if np.linalg.norm(cross_axis) > 1e-6:  # Avoid zero vector
                axes.append(normalize(cross_axis))

    # Check for separation on any axis
    for axis in axes:
        min1, max1 = project_obb(obb1, axis)
        min2, max2 = project_obb(obb2, axis)
        if not overlap(min1, max1, min2, max2):
            return False  # Separating axis found → No collision

    return True  # No separating axis found → Collision


def expand_obb(corners, expansion_distance):
    """
    Expand an OBB by a given distance in all directions.
    
    Parameters:
    corners (numpy array): (8,3) array representing the 8 corner points of the OBB.
    expansion_distance (float): Distance to expand the OBB.

    Returns:
    numpy array: (8,3) expanded OBB corner points.
    """
    # Compute OBB center
    center = np.mean(corners, axis=0)

    # Compute the primary axes (edge vectors)
    edges = get_edges(corners)
    axes = np.array([normalize(edge) for edge in edges])  # Normalize for unit directions

    # Expand along each axis
    expanded_corners = []
    for corner in corners:
        expanded_corner = corner.copy()
        for axis in axes:
            expanded_corner += axis * expansion_distance if np.dot(corner - center, axis) > 0 else -axis * expansion_distance
        expanded_corners.append(expanded_corner)

    return np.array(expanded_corners)

def obb_collision_expanded(obb1:SceneObject, obb2:SceneObject, expansion_distance=0.0):
    """
    Check if two 3D OBBs collide using the Separating Axis Theorem with expanded OBBs.
    
    Parameters:
    obb1, obb2: numpy arrays of shape (8,3), representing the 8 corner points of each OBB.
    expansion_distance: float, distance to expand the OBBs.

    Returns:
    True if OBBs collide, False otherwise.
    """
    obb1 = expand_obb(obb1, expansion_distance)
    obb2 = expand_obb(obb2, expansion_distance)
    return obb_collision(obb1, obb2)

####################################################################################################

def preprocess_image(img, vis=False):
    _, buffer = cv2.imencode(".png", img[:, :, ::-1])
    base64_image = base64.b64encode(buffer).decode("utf-8")
    if vis:
      # save the base64 image with cv2
      img = base64.b64decode(base64_image)
      img = BytesIO(img)
      img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
      cv2.imwrite("image.jpeg", img)
    return base64_image 

def crop_max_circle(image):
    """
    Crop the largest possible circle from the center of the image.
    """
    height, width = image.shape[:2]
    radius = min(height, width) // 2  # Maximum radius possible

    # Compute the center of the image
    center_x, center_y = width // 2, height // 2

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    # Crop the square region containing the circle
    cropped_circle = result[center_y - radius:center_y + radius, center_x - radius:center_x + radius]

    # Create transparent background for cropped circle
    b, g, r = cv2.split(cropped_circle)
    alpha = mask[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    cropped_circle_rgba = cv2.merge([b, g, r, alpha])

    return cropped_circle_rgba

def rotate_circle(image, angle):
    """
    Rotate only the circular cropped part, keeping the background transparent.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate only the cropped circle, keeping transparency
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    return rotated_image

def crop_max_and_rotate(image, angle, size=(700, 700)):
    """
    Crop the largest possible circle from the center of the image and rotate it.
    """
    cropped_circle = crop_max_circle(image)
    rotated_circle = rotate_circle(cropped_circle, angle)

    return cv2.resize(rotated_circle, size, interpolation=cv2.INTER_LINEAR)[..., :3]

import colorsys

def generate_distinct_colors(n):
    """Generates `n` distinct colors as a NumPy array with float RGB values (0-1)."""
    colors = np.array([colorsys.hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)], dtype=np.float32)
    return colors

