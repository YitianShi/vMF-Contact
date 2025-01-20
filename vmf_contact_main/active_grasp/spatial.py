import copy
import numpy as np
from scipy.spatial.transform import Rotation
from math import sin, cos
from scipy import ndimage
from vgn.grasp import Grasp

class ViewHalfSphere:
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[-1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError
    
class SpatialTransform:
    def __init__(self, rotation, translation):
        self.rotation = copy.deepcopy(rotation)
        self.translation = np.asarray(translation, np.double).copy()

    @classmethod
    def from_rotation(cls, rotation):
        translation = np.zeros(3)
        return cls(rotation, translation)

    @classmethod
    def from_translation(cls, translation):
        rotation = Rotation.identity()
        return cls(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, l):
        return cls(Rotation.from_quat(l[:4]), l[4:])

    @classmethod
    def identity(cls):
        rotation = Rotation.identity()
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, target, up):
        forward = np.subtract(target, eye)
        forward = np.divide(forward, np.linalg.norm(forward))

        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-3:
            right = np.cross(forward, up + np.r_[1e-3, 0, 0])
        right = np.divide(right, np.linalg.norm(right))

        up = np.cross(right, forward)
        up = np.divide(up, np.linalg.norm(up))

        m = np.array(
            [
                [right[0], -up[0], forward[0], eye[0]],
                [right[1], -up[1], forward[1], eye[1]],
                [right[2], -up[2], forward[2], eye[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        return cls.from_matrix(m)

    def __mul__(self, other):
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def inv(self):
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    def apply(self, point):
        return self.rotation.apply(point) + self.translation

    def as_matrix(self):
        return np.vstack(
            (
                np.c_[self.rotation.as_matrix(), self.translation],
                [0.0, 0.0, 0.0, 1.0],
            )
        )

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    class TClass:
        """
        Convenient way to create a pure translation.

        Transform.t_[x, y, z] is equivalent to Transform.from_translation(np.r_[x, y, z]).
        """

        def __getitem__(self, key):
            return SpatialTransform.from_translation(np.r_[key])

    t_ = TClass()


def find_urdfs(root):
    return list(root.glob("**/*.urdf"))


def cartesian_to_spherical(p):
    x, y, z = p
    r = np.linalg.norm(p)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    return np.r_[r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]


def look_at(eye, center, up):
    eye = np.asarray(eye)
    center = np.asarray(center)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.asarray(up) / np.linalg.norm(up)
    up = np.cross(right, forward)
    m = np.eye(4, 4)
    m[:3, 0] = right
    m[:3, 1] = -up
    m[:3, 2] = forward
    m[:3, 3] = eye
    return SpatialTransform.from_matrix(m)


def view_on_sphere(origin, r, theta, phi):
    eye = spherical_to_cartesian(r, theta, phi)
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return origin * look_at(eye, target, up)


def map_cloud_to_grid(voxel_size, points, distances):
    grid = np.zeros((40, 40, 40), dtype=np.float32)
    indices = (points // voxel_size).astype(int)
    grid[tuple(indices.T)] = distances.squeeze()
    return grid


def grid_to_map_cloud(voxel_size, grid, threshold=1e-2):
    points = np.argwhere(grid > threshold) * voxel_size
    distances = np.expand_dims(grid[grid > threshold], 1)
    return points, distances


def box_lines(lower, upper):
    x_l, y_l, z_l = lower
    x_u, y_u, z_u = upper
    return [
        ([x_l, y_l, z_l], [x_u, y_l, z_l]),
        ([x_u, y_l, z_l], [x_u, y_u, z_l]),
        ([x_u, y_u, z_l], [x_l, y_u, z_l]),
        ([x_l, y_u, z_l], [x_l, y_l, z_l]),
        ([x_l, y_l, z_u], [x_u, y_l, z_u]),
        ([x_u, y_l, z_u], [x_u, y_u, z_u]),
        ([x_u, y_u, z_u], [x_l, y_u, z_u]),
        ([x_l, y_u, z_u], [x_l, y_l, z_u]),
        ([x_l, y_l, z_l], [x_l, y_l, z_u]),
        ([x_u, y_l, z_l], [x_u, y_l, z_u]),
        ([x_u, y_u, z_l], [x_u, y_u, z_u]),
        ([x_l, y_u, z_l], [x_l, y_u, z_u]),
    ]

def from_voxel_coordinates(voxel_size, grasp):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)

def select_at(out, index):
    i, j, k = index
    ori = Rotation.from_quat(out.rot[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = out.width[i, j, k]
    quality = out.qual[i, j, k]
    return Grasp(SpatialTransform(ori, pos), width), quality

def select_local_maxima(
    voxel_size,
    out,
    threshold=0.9,
    max_filter_size=3.0,
):
    max = ndimage.maximum_filter(out.qual, size=max_filter_size)
    index_list = np.argwhere(np.logical_and(out.qual == max, out.qual > threshold))
    grasps, qualities = [], []
    for index in index_list:
        grasp, quality = select_at(out, index)
        grasps.append(grasp)
        qualities.append(quality)
    grasps = np.array([from_voxel_coordinates(voxel_size, g) for g in grasps])
    qualities = np.asarray(qualities)
    return grasps, qualities


