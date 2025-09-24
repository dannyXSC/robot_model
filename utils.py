import numpy as np
from scipy.spatial.transform import Rotation as R


def point_to_hom(points):
    """
    Convert points to homogeneous coordinates.

    points: (3,) 或 (N, 3) numpy array
    return: (4,) 或 (N, 4) numpy array
    """
    points = np.asarray(points)

    if points.ndim == 1 and points.shape[0] == 3:
        # 单点
        return np.hstack([points, 1])
    elif points.ndim == 2 and points.shape[1] == 3:
        # 多点
        return np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        raise ValueError(f"Expected input shape (3,) or (N,3), got {points.shape}")


def hom_to_point(points_hom):
    """
    Convert points from homogeneous coordinates to 3D.

    points_hom: (4,) 或 (N, 4) numpy array
    return: (3,) 或 (N, 3) numpy array
    """
    points_hom = np.asarray(points_hom)

    if points_hom.ndim == 1 and points_hom.shape[0] == 4:
        # 单点
        return points_hom[:3] / points_hom[3]
    elif points_hom.ndim == 2 and points_hom.shape[1] == 4:
        # 多点
        return points_hom[:, :3] / points_hom[:, 3:4]
    else:
        raise ValueError(f"Expected input shape (4,) or (N,4), got {points_hom.shape}")


def to_hom(point, quat):
    """
    Convert points to homogeneous coordinates and apply rotation.

    points: (3,) numpy array
    quat: (4,) numpy array representing a quaternion(xyzw)
    return: (4, 4) numpy array
    """
    R_mat = R.from_quat(quat).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = point
    return T


def from_hom(hom_matrix):
    """
    Convert from homogeneous transformation matrix to position and quaternion.

    hom_matrix: (4, 4) numpy array
    return: point (3,) numpy array, quat (4,) numpy array representing a quaternion(xyzw)
    """
    point = hom_matrix[:3, 3]
    R_mat = hom_matrix[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # xyzw

    return point, quat
