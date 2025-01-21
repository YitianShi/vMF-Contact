import numpy as np
import open3d as o3d
import torch
from typing import Union, Optional

#### Codes borrowed from pytorch3d ####


from typing import Optional, Union, Tuple, List
import math
import torch
import torch.nn.functional as F
import random

Device = Union[str, torch.device]

class GraspBuffer:
    def __init__(self, device="cuda:0"):
        self.buffer_dict = {
            "pcds": [],
            "baselines": [], 
            "approaches": [], 
            "cp": [], 
            "cp2": [], 
            "kappa": [], 
            "graspness": []}
        self.buffer_size = 0
        
    def create_vis(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def update(self, 
               pcds, 
               predictions,
               shift=0.0,
               resize=1.0, 
               grasp_height_th=5e-3, 
               grasp_width_th=0.1, 
               graspness_th=0.3, 
               pcd_from_prompt=None):
        
        if not isinstance(shift, torch.Tensor):
            shift = torch.tensor(shift, device=pcds.device, dtype=torch.float32)
        if not isinstance(resize, torch.Tensor):
            resize = torch.tensor(resize, device=pcds.device, dtype=torch.float32)
        
        cp = predictions["contact_point"]
        cp2 = predictions["contact_point"] + predictions["grasp_width"].unsqueeze(-1) * predictions["baseline"]
        grasp_width = predictions["grasp_width"]
        graspness = predictions["graspness"]
        approach = predictions["approach"]
        baseline = predictions["baseline"]
        kappa = predictions["kappa"]

        filter = (graspness > graspness_th) & \
                    (grasp_width < grasp_width_th) & \
                    (cp[..., -1] > grasp_height_th) & \
                    (cp2[..., -1] > grasp_height_th)

        if pcd_from_prompt is not None:
            pcd_from_prompt = torch.tensor(pcd_from_prompt, device=self.device, dtype=torch.float32)
            # calculate the distance between the contact points and the prompt points
            dist = torch.cdist(cp, pcd_from_prompt)
            # dist2 = torch.cdist(cp2, pcd_from_prompt)
            filter = filter & (dist.min(1).values < 0.01)
        
        pcds = pcds * resize + shift

        if filter.sum() == 0:
            print("No valid grasp")
            return False

        # print(f"Number of grasps: {filter.sum()}")
        else:
            cp = cp[filter] * resize + shift
            cp2 = cp2[filter] * resize + shift
            mid_pt = (cp2 + cp) / 2
            
            baseline = predictions["baseline"][filter]
            kappa = predictions["kappa"][filter]
            approach = approach[filter]
            graspness = graspness[filter]

            self.buffer_dict["pcds"].append(pcds)  
            self.buffer_dict["baselines"].append(baseline)
            self.buffer_dict["approaches"].append(approach)
            self.buffer_dict["cp"].append(cp)
            self.buffer_dict["cp2"].append(cp2)
            self.buffer_dict["kappa"].append(kappa)
            self.buffer_dict["graspness"].append(graspness)

            self.buffer_size += 1
            return True
  
    def get_pcds_all(self):
        return torch.cat(self.buffer_dict["pcds"], dim=0)
    
    def get_grasp_all(self):
        baselines = torch.cat(self.buffer_dict["baselines"], dim=0)
        approaches = torch.cat(self.buffer_dict["approaches"], dim=0)
        cp = torch.cat(self.buffer_dict["cp"], dim=0)
        cp2 = torch.cat(self.buffer_dict["cp2"], dim=0)
        kappa = torch.cat(self.buffer_dict["kappa"], dim=0)
        graspness = torch.cat(self.buffer_dict["graspness"], dim=0)
        return baselines, approaches, cp, cp2, kappa, graspness
    
    def get_pose_all(self, convention="xzy"):
        baselines, approaches, cp, cp2, kappa, graspness = self.get_grasp_all()
        poses = rotation_from_contact(baseline=baselines, 
                                      approach=approaches, 
                                      translation=(cp+cp2)/2, 
                                      convention=convention)
        return poses, kappa, graspness
    
    def get_pcds_curr(self):
        return self.buffer_dict["pcds"][-1]
    
    def get_grasp_curr(self):
        baseline = self.buffer_dict["baselines"][-1]
        approach = self.buffer_dict["approaches"][-1]
        cp = self.buffer_dict["cp"][-1]
        cp2 = self.buffer_dict["cp2"][-1]
        kappa = self.buffer_dict["kappa"][-1]
        graspness = self.buffer_dict["graspness"][-1]
        return baseline, approach, cp, cp2, kappa, graspness
    
    def get_pose_curr(self, convention="xzy"):
        baseline, approach, cp, cp2, kappa, graspness = self.get_grasp_curr()
        poses = rotation_from_contact(baseline=baseline, 
                                      approach=approach, 
                                      translation=(cp+cp2)/2,
                                      convention=convention)
        return poses, kappa, graspness
    
    def set_view(self, center):
        """Set a specific viewpoint."""
        ctr = self.vis.get_view_control()

        # Set camera parameters
        ctr.set_zoom(1)  # Zoom factor
        ctr.set_lookat(center)  # Look at center
        ctr.set_front([-1, 0, 1])  # View direction
        ctr.set_up([0, 0, 1])  # Up vector
    
    def vis_grasps(self, all = False):

        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to visualize")
            return

        pcd = self.get_pcds_curr()
        baseline, approach, cp, cp2, kappa, graspness = self.get_grasp_curr()
        vis_list = vis_grasps(
                    samples=pcd,
                    cp=cp,
                    cp2=cp2,
                    kappa=kappa,
                    approach=approach,
                    score = graspness,
                )
            
        if not hasattr(self, "vis"):
            self.create_vis()

        if not all:
            self.vis.clear_geometries()

        for geom in vis_list:
            self.vis.add_geometry(geom)
        # Update the visualizer
        center = pcd.mean(0).cpu().numpy()
        self.set_view(center = center)
        self.vis.poll_events()
        self.vis.update_renderer()            

    def get_pose_curr_best(self, convention="xzy", sort_by="kappa", sample_num=1):

        if len(self.buffer_dict["pcds"]) == 0:
            print("Buffer is empty, no grasp to choose")
            return None
        
        poses, kappa, graspness = self.get_pose_curr(convention)
        
        score = kappa if sort_by == "kappa" else graspness
        
        #sort poses by criterion
        sample_num = min(sample_num, poses.size(0))
        poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

        #randomly sample 1 poses
        pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
        return pose_chosen
    

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.
    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).
    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.
    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).
    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.
    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.
    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.
    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

@torch.jit.script
def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    return q / torch.norm(q, dim=-1, keepdim=True)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)



# def random_quaternions(
#     n: int, *ns, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
# ) -> torch.Tensor:
#     """
#     Generate random quaternions representing rotations,
#     i.e. versors with nonnegative real part.
#     Args:
#         n: Number of quaternions in a batch to return.
#         dtype: Type to return.
#         device: Desired device of returned tensor. Default:
#             uses the current device for the default tensor type.
#     Returns:
#         Quaternions as tensor of shape (N, 4).
#     """
#     if isinstance(device, str):
#         device = torch.device(device)
#     shape = [n] + [i for i in ns] + [4]
#     o = torch.randn(shape, dtype=dtype, device=device)
#     s = (o * o).sum(dim=-1)
#     o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
#     return o

@torch.jit.script
def random_quaternions(n: int, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None):
    if isinstance(device, str):
        device = torch.device(device)
    q = torch.randn(n,4, device=device, dtype=dtype)

    return standardize_quaternion(q / torch.norm(q, dim=-1, keepdim=True))


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def hat_inv(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.
    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.
    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.
    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.
    A SE(3) matrix has the following form:
        ```
        [ R T ]
        [ 0 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.
    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.
    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) log_translation ]
                         [   0                        1      ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].
    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```
    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.
    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.
    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.
    Raises:
        ValueError if `log_transform` is of incorrect shape.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform


DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4

def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.
    More specifically::
        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)
    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)
#### Codes borrowed from pytorch3d ####


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.
    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_log_map(
    R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.
    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.
    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles


def se3_log_map(
    transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.
    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.
    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.
    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [log_translation | log_rotation]`
    is done as follows:
        ```
        log_transform = log(transform)
        log_translation = log_transform[3, :3]
        log_rotation = inv_hat(log_transform[:3, :3])
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].
    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_exp_map(se3_log_map(transform)) == transform
    ```
    The conversion has a singularity around `(transform=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.
    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid division by zero in the singular case.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
            The non-finite outputs can be caused by passing small rotation angles
            to the `acos` function in `so3_rotation_angle` of `so3_log_map`.
    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.
    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, 3, :3], torch.zeros_like(transform[:, 3, :3])):
        raise ValueError("All elements of `transform[:, 3, :3]` should be 0.")

    # log_rot is just so3_log_map of the upper left 3x3 block
    R = transform[:, :3, :3]
    log_rotation = so3_log_map(R, eps=eps, cos_bound=cos_bound)

    # log_translation is V^-1 @ T
    T = transform[:, :3, 3]
    V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
    log_translation = torch.linalg.solve(V, T[:, :])[:, :]

    return torch.cat((log_translation, log_rotation), dim=1)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

@torch.jit.script
def multiply_se3(T1: torch.Tensor, T2: torch.Tensor, pre_normalize: bool = False, post_normalize: bool = True) -> torch.Tensor:
    if len(T1) == 1 or len(T2) == 1:
        assert T1.shape[1:] == T2.shape[1:], f"Shape mismatch: T1: {T1.shape} || T2: {T2.shape}"
    elif T1.ndim + 1 == T2.ndim:
        assert T1.shape[:] == T2.shape[1:], f"Shape mismatch: T1: {T1.shape} || T2: {T2.shape}"
    elif T1.ndim == T2.ndim + 1:
        assert T1.shape[1:] == T2.shape[:], f"Shape mismatch: T1: {T1.shape} || T2: {T2.shape}"
    else:
        assert T1.shape == T2.shape, f"Shape mismatch: T1: {T1.shape} || T2: {T2.shape}"

    q1, x1 = T1[...,:4], T1[...,4:]
    q2, x2 = T2[...,:4], T2[...,4:]
    if pre_normalize:
        q1 = normalize_quaternion(q1)
        q2 = normalize_quaternion(q2)
        
    x = quaternion_apply(q1, x2) + x1
    q = quaternion_multiply(q1, q2)
    if post_normalize:
        q = normalize_quaternion(q)
    
    return torch.cat([q,x], dim=-1)

@torch.jit.script
def multiply_so3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    if len(R1) == 1 or len(R2) == 1:
        assert R1.shape[1:] == R2.shape[1:], f"Shape mismatch: R1: {R1.shape} || R2: {R2.shape}"
    elif R1.ndim + 1 == R2.ndim:
        assert R1.shape[:] == R2.shape[1:], f"Shape mismatch: R1: {R1.shape} || R2: {R2.shape}"
    elif R1.ndim == R2.ndim + 1:
        assert R1.shape[1:] == R2.shape[:], f"Shape mismatch: R1: {R1.shape} || R2: {R2.shape}"
    else:
        assert R1.shape == R2.shape, f"Shape mismatch: R1: {R1.shape} || R2: {R2.shape}"
    return torch.bmm(R1, R2)

@torch.jit.script
def se3_invert(T: torch.Tensor) -> torch.Tensor:
    qinv = quaternion_invert(T[...,:4])
    return torch.cat([qinv, quaternion_apply(qinv, -T[..., 4:])], dim=-1)

@torch.jit.script
def quaternion_identity(n: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.tensor((1., 0., 0., 0.), device=device, dtype=dtype).repeat((n,1))

@torch.jit.script
def se3_from_r3(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones_like(x[...,0:1]), torch.zeros_like(x[...,:3]), x], dim=-1)


def rotate_circle_to_batch_of_vectors(bin_num, target_vectors):
    u_batch = perpendicular_highest_z(target_vectors)
    bin_vectors = generate_bin_vectors(target_vectors, u_batch, bin_num)
    return -bin_vectors


def perpendicular_highest_z(v):
    v = v / torch.norm(v, dim=1, keepdim=True)
    # Components of the input vector
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    u_x = -vy
    u_y = vx
    u = torch.stack([u_x, u_y, vz*0], dim=1)
    u = u / torch.norm(u, dim=1, keepdim=True)

    u = torch.linalg.cross(v, u)
    return u


def rotation_matrix(v, theta):
    # Normalize v to ensure it's a unit vector
    v = v / torch.norm(v, dim=1, keepdim=True)

    # Components of v
    vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]

    # Compute cos(theta) and sin(theta)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    one_minus_cos_theta = 1 - cos_theta

    # Rotation matrix components
    rotation = torch.zeros(v.size(0), 3, 3, device=v.device)

    rotation[:, 0, 0] = cos_theta.squeeze() + (vx * vx * one_minus_cos_theta).squeeze()
    rotation[:, 0, 1] = (vx * vy * one_minus_cos_theta - vz * sin_theta).squeeze()
    rotation[:, 0, 2] = (vx * vz * one_minus_cos_theta + vy * sin_theta).squeeze()

    rotation[:, 1, 0] = (vy * vx * one_minus_cos_theta + vz * sin_theta).squeeze()
    rotation[:, 1, 1] = cos_theta.squeeze() + (vy * vy * one_minus_cos_theta).squeeze()
    rotation[:, 1, 2] = (vy * vz * one_minus_cos_theta - vx * sin_theta).squeeze()

    rotation[:, 2, 0] = (vz * vx * one_minus_cos_theta - vy * sin_theta).squeeze()
    rotation[:, 2, 1] = (vz * vy * one_minus_cos_theta + vx * sin_theta).squeeze()
    rotation[:, 2, 2] = cos_theta.squeeze() + (vz * vz * one_minus_cos_theta).squeeze()

    return rotation


def generate_bin_vectors(v, u, num_points):
    num_points = (num_points + 1) // 2

    # Define the angles for 90-degree rotations
    theta1 = torch.tensor([torch.pi / 2]).repeat(v.size(0)).to(v.device)
    theta2 = torch.tensor([-torch.pi / 2]).repeat(v.size(0)).to(v.device)

    # Generate rotation matrices for +90 and -90 degrees
    rotation_matrix1 = rotation_matrix(v, theta1)
    rotation_matrix2 = rotation_matrix(v, theta2)

    # Rotate the starting vector by +90 and -90 degrees
    u_plus_90 = torch.matmul(rotation_matrix1, u.unsqueeze(2)).squeeze(2)
    u_minus_90 = torch.matmul(rotation_matrix2, u.unsqueeze(2)).squeeze(2)

    # Generate linspace for the 180-degree coverage
    theta_values = torch.linspace(0, 1, num_points).to(v.device).view(1, -1, 1)

    # Interpolate between u and u_plus_90
    vectors_pos = (1 - theta_values) * u.unsqueeze(
        1
    ) + theta_values * u_plus_90.unsqueeze(1)

    # Interpolate between u and u_minus_90
    vectors_neg = theta_values * u.unsqueeze(1) + (
        1 - theta_values
    ) * u_minus_90.unsqueeze(1)

    # Combine positive and negative rotations
    combined_vectors = torch.cat((vectors_neg, vectors_pos[:, 1:]), dim=1)

    combined_vectors = combined_vectors / torch.norm(
        combined_vectors, dim=2, keepdim=True
    )

    return combined_vectors


# Function to create a cylinder between two points
def create_cylinder_between_points(p1, p2, radius=0.05, color=[0.1, 0.1, 0.7]):
    # Calculate the direction and length of the cylinder
    direction = p2 - p1
    length = np.linalg.norm(direction)
    direction /= length

    # Create a cylinder mesh
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()

    # Rotate the cylinder to align with the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    # Translate the cylinder to start at point p1
    cylinder.translate((p1 + p2) / 2)

    # Paint the cylinder with the specified color
    cylinder.paint_uniform_color(color)

    return cylinder


def over_or_re_sample(pcd, num_points):
    c = pcd.shape[-1]
    # Determine the maximum size
    if pcd.shape[0] < num_points:
        # Oversample the point cloud
        pad_size = num_points - pcd.shape[0]
        pcd_rest_ind = torch.randint(0, pcd.shape[0], (pad_size,), device=pcd.device)
        pcd = torch.cat([pcd, pcd[pcd_rest_ind]], dim=0)
    else:
        # Resample the point cloud
        indices = torch.randint(0, pcd.shape[0], (num_points,), device=pcd.device)
        pcd = torch.gather(pcd, 0, indices.unsqueeze(-1).expand(-1, c))
    return pcd


def rotation_from_contact(baseline, approach, translation, convention = "xzy", quat=False):
    """
    Compute the rotation matrix from the baseline and approach vectors.
    convention follows: baseline, approach, up
    """
    
    # Define the front direction (negative x-axis)
    up_direction = torch.tensor([-1, 0, 0], dtype=baseline.dtype, device=baseline.device)
    
    if convention == "xzy":
        x = torch.nn.functional.normalize(baseline, dim=-1)  # Baseline vector (B, 3)
        z = torch.nn.functional.normalize(approach, dim=-1) # Approach vector (B, 3)
        y = torch.nn.functional.normalize(torch.linalg.cross(z, x), dim=-1)
        
        # Ensure y is aligned with the up direction
        # dot_product = torch.sum(y * up_direction, dim=-1, keepdim=True)  # dot product with up direction
        # y = torch.where(dot_product < 0, -y, y)  # Flip y if it's pointing downward
        
    
    elif convention == "zyx":
        x = torch.nn.functional.normalize(baseline, dim=-1)
        y = torch.nn.functional.normalize(approach, dim=-1)
        z = torch.nn.functional.normalize(torch.linalg.cross(x, y), dim=-1)
        
        # Ensure x is aligned with the up direction
        dot_product = torch.sum(x * up_direction, dim=-1, keepdim=True)  # dot product with up direction
        # x = torch.where(dot_product < 0, -x, x)  # Flip x if it's pointing downward

    # Construct the rotation matrix
    rotation_matrices = torch.stack([x, y, z], dim=-1)  # Shape (B, 3, 3)

    # Step 6: Construct the homogeneous transformation matrix
    # Create a (B, 4, 4) tensor to store the transformation matrix
    homogeneous_matrices = torch.zeros((x.shape[0], 4, 4), dtype=x.dtype, device=x.device)

    # Place the rotation matrix in the top-left 3x3 block
    homogeneous_matrices[:, :3, :3] = rotation_matrices

    # Place the translation vector in the top-right 3x1 block
    homogeneous_matrices[:, :3, 3] = translation

    # Set the bottom row to [0, 0, 0, 1] for each matrix
    homogeneous_matrices[:, 3, 3] = 1

    if quat:
        quaternion = matrix_to_quaternion(rotation_matrices)
        poses = torch.cat([translation, quaternion], dim=-1)
        return poses
        
    return homogeneous_matrices


def contact_from_rotations(rotation_matrices, convention="xzy"):
    """
    Extract the contact frame from the rotation matrix.
    convention follows: baseline, approach, up
    """
    if convention == "xzy":
        baseline = rotation_matrices[..., 0, :]
        approach = rotation_matrices[..., 2, :]
    elif convention == "zyx":
        baseline = rotation_matrices[..., 2, :]
        approach = rotation_matrices[..., 1, :]
    return baseline, approach


def contact_from_quaternion(quaternions, convention="xzy"):
    """
    Extract the contact frame from the rotation matrix.
    convention follows: baseline, approach, up
    """
    rotation_matrix = quaternion_to_matrix(quaternions)
    
    if convention == "xzy":
        baseline = rotation_matrix[..., 0, :]
        approach = rotation_matrix[..., 2, :]
    elif convention == "zyx":
        baseline = rotation_matrix[..., 2, :]
        approach = rotation_matrix[..., 1, :]
    return baseline, approach


def draw_grasps(cp, cp2, approach, bin_vectors=None, score=None, kappa=None,
                color=[0.7, 0.1, 0.1], graspline_width=5e-4, finger_length=0.025,
                arm_length=0.02, sphere_radius=2e-3):
    
    vis_list = []
    color_max = np.array([1, 1, 1])  # Light red (RGB)
    color_min = np.array([0, 0, 0])
    cp_half = (cp + cp2) / 2

    if cp is not None and cp2 is not None:
        for i, (q, a, app, half_q, half_a) in enumerate(zip(cp, cp2, approach, 
                                                           cp_half - approach * finger_length, 
                                                           cp_half - approach * (finger_length + arm_length))):
            # Determine color based on score
            color = color_max * score[i] + color_min * (1 - score[i]) if score is not None else color
            
            # Draw fingers and arm cylinders
            vis_list.extend([
                create_cylinder_between_points(a - app * finger_length, a, radius=graspline_width, color=color),
                create_cylinder_between_points(q - app * finger_length, q, radius=graspline_width, color=color),
                create_cylinder_between_points(q - app * finger_length, a - app * finger_length, radius=graspline_width, color=color),
                create_cylinder_between_points(half_q, half_a, radius=graspline_width, color=color)
            ])
            
            # Draw bin_vectors lines if provided
            if bin_vectors is not None:
                bin_vectors_np = bin_vectors.detach().cpu().numpy() if isinstance(bin_vectors, torch.Tensor) else bin_vectors
                for vec in bin_vectors_np[i]:
                    line = o3d.geometry.LineSet()
                    line.points = o3d.utility.Vector3dVector([half_q, half_q + vec * 0.1])
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    line.colors = o3d.utility.Vector3dVector([color])
                    vis_list.append(line)

            # Draw spheres if kappa is provided
            if kappa is not None:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * kappa[i] / 10)
                sphere.paint_uniform_color(color)
                sphere.translate(q - app * finger_length)
                vis_list.append(sphere)
    
    return vis_list


def gram_schmidt(batch_a, batch_b):
    """
    Perform 1-to-1 Gram-Schmidt process where batch_a is processed w.r.t batch_b.
    
    Args:
    - batch_a: Tensor of shape (M, 3), the first batch of vectors to process.
    - batch_b: Tensor of shape (M, 3), the second batch of vectors (reference).
    
    Returns:
    - processed_a: Tensor of shape (M, 3), orthonormalized version of batch_a w.r.t. batch_b.
    - normalized_b: Tensor of shape (M, 3), normalized version of batch_b.
    """
    # Normalize batch_b
    norm_b = torch.norm(batch_b, dim=1, keepdim=True)
    normalized_b = batch_b / norm_b

    # Orthogonalize batch_a with respect to normalized_b
    projection = (torch.sum(batch_a * normalized_b, dim=1, keepdim=True) * normalized_b)
    orthogonal_a = batch_a - projection

    # Normalize orthogonal_a
    norm_a = torch.norm(orthogonal_a, dim=1, keepdim=True)
    processed_a = orthogonal_a / norm_a

    return processed_a


def vis_grasps(
        samples = None,
        groups=None,
        cp_gt=None,
        cp2_gt=None,
        cp=None,
        cp2=None,
        kappa=None,
        approach_gt=None,
        approach=None,
        bin_vectors=None,
        bin_vectors_gt=None,
        score=None,
    ):

        vis_list = []

        if approach is not None:
            approach = (approach.detach().cpu().numpy() if isinstance(approach, torch.Tensor) else approach)
        if approach_gt is not None:
            approach_gt = (approach_gt.detach().cpu().numpy() if isinstance(approach_gt, torch.Tensor) else approach_gt)
        if cp is not None:
            cp = cp.detach().cpu().numpy() if isinstance(cp, torch.Tensor) else cp
            cp2 = cp2.detach().cpu().numpy() if isinstance(cp2, torch.Tensor) else cp2
        if cp_gt is not None:
            cp_gt = cp_gt.detach().cpu().numpy() if isinstance(cp_gt, torch.Tensor) else cp_gt
            cp2_gt = cp2_gt.detach().cpu().numpy() if isinstance(cp2_gt, torch.Tensor) else cp2_gt
        if score is not None:
            score = score.cpu().numpy() if isinstance(score, torch.Tensor) else score
        if kappa is not None:
            kappa = kappa.detach().cpu().numpy() if isinstance(kappa, torch.Tensor) else kappa

        # Visualize the sampled points
        if samples is not None:
            samples = samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(samples)
            vis_list.append(pcd)

        # Connect line between the cp_gt anchor and the cp
        if cp_gt is not None and cp is not None:
            for q, a in zip(cp, cp_gt):
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([a, q])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector(
                    np.tile([0.1, 0.1, 0.7], (1, 1))
                )
                vis_list.append(line)

        if groups is not None:
            rgb_groups = torch.rand((groups.shape[0], 3))
            groups = (
                groups.cpu().numpy() if isinstance(groups, torch.Tensor) else groups
            )
            rgb_groups = (
                rgb_groups.cpu().numpy()
                if isinstance(rgb_groups, torch.Tensor)
                else rgb_groups
            )
            pcds_groups = []
            for i in range(groups.shape[0]):
                pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(groups[i] + samples[i])
                pcd.points = o3d.utility.Vector3dVector(groups[i])
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(rgb_groups[i], (groups.shape[1], 1))
                )
                pcds_groups.append(pcd)

            vis_list += pcds_groups

        if cp is not None:
            vis_list += draw_grasps(cp, cp2, approach, bin_vectors, score, kappa)
        #if cp_gt is not None:
            #vis_list += draw_grasps(cp_gt, cp2_gt, approach_gt, bin_vectors_gt, score, None, color=[0.1, 0.7, 0.1])

        return vis_list
