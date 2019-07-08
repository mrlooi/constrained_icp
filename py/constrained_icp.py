import open3d as o3d
import numpy as np
# import transformations as tr
from scipy.optimize import least_squares
from transforms3d.euler import euler2mat

rotation_axis = np.array([0, 0, 1], dtype=float)
translation_axis1 = np.array([1, 0, 0], dtype=float)
translation_axis2 = np.array([0, 1, 0], dtype=float)


def rotation_matrix(angle, rotation_axis):
    rot_angle = rotation_axis * angle

    M = np.eye(4)
    M[:3, :3] = euler2mat(*np.deg2rad(rot_angle))
    return M


def point_plane_residual(x: np.ndarray, base_transform: np.ndarray,
                         pcd_source: o3d.PointCloud, pcd_target: o3d.PointCloud, kdtree_target: o3d.KDTreeFlann,
                         max_correspond_dist):
    assert pcd_target.has_normals()

    num_points = len(pcd_source.points)
    points_source = np.asarray(pcd_source.points)

    nearest_points = np.empty((num_points, 3), dtype=float)
    nearest_normals = np.empty((num_points, 3), dtype=float)
    points_target = np.asarray(pcd_target.points)
    normals_target = np.asarray(pcd_target.normals)

    for i in range(num_points):
        k, idx, _ = kdtree_target.search_hybrid_vector_3d(pcd_source.points[i], max_correspond_dist, 1)
        if k == 1:
            nearest_points[i, :] = points_target[idx[0], :]
            nearest_normals[i, :] = normals_target[idx[0], :]
        else:
            nearest_points[i, :] = points_source[i, :]
            nearest_normals[i, :] = np.array([0, 0, 0], dtype=float)    # cause a zero residual

    angle = x[0]
    t1 = x[1]
    t2 = x[2]

    transform = rotation_matrix(angle, rotation_axis)
    transform[:3, 3] = t1 * translation_axis1 + t2 * translation_axis2
    transform = np.dot(transform, base_transform)

    position_residuals = np.dot(transform[:3, :3], points_source.T) + transform[:3, 3].reshape(3, 1) - nearest_points.T
    residuals = position_residuals * nearest_normals.T
    residuals = np.sum(residuals, axis=0)

    return residuals


def cicp(pcd_source: o3d.PointCloud, pcd_target: o3d.PointCloud, base_transform: np.ndarray,
         max_correspond_dist_coarse, max_correspond_dist_fine):
    kdtree = o3d.KDTreeFlann(pcd_target)
    if not pcd_target.has_normals():
        o3d.estimate_normals(pcd_target, search_param=o3d.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    x0 = np.array([0, 0, 0], dtype=float)
    result = least_squares(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        loss='soft_l1',
        args=(base_transform, pcd_source, pcd_target, kdtree, max_correspond_dist_coarse)
    )

    x0 = result.x
    result = least_squares(
        point_plane_residual,
        x0,
        jac='2-point',
        method='trf',
        loss='soft_l1',
        args=(base_transform, pcd_source, pcd_target, kdtree, max_correspond_dist_fine)
    )

    x = result.x
    angle = x[0]
    t1 = x[1]
    t2 = x[2]

    transform = rotation_matrix(angle, rotation_axis)
    transform[:3, 3] = t1 * translation_axis1 + t2 * translation_axis2
    transform = np.dot(transform, base_transform)

    return transform, result
