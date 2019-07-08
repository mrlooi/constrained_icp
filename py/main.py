import open3d
import numpy as np
# import transformations as tr
from transforms3d.quaternions import quat2mat, mat2quat
import constrained_icp as cicp

o3d = open3d

file_id_start = 0
file_id_stop = 15

voxel_size = 0.02
max_point_depth = 2
icp_dist_coarse = voxel_size * 15
icp_dist_fine = voxel_size * 5


def main():
    pcds = []
    for i in range(file_id_start, file_id_stop + 1, 1):
        pcd_file = './data/pcd_%d.pcd' % (i)
        print("Reading %s..."%(pcd_file))
        pcd = o3d.read_point_cloud(pcd_file)
        pcds.append(pcd)

    pcds = crop_clouds_by_depth(pcds, max_point_depth)
    pcds = remove_clouds_outliers(pcds, 30, voxel_size, 1)  # removing outliers before downsample give good result.
    pcds = downsample_clouds(pcds, voxel_size)
    # pcds = remove_clouds_outliers(pcds, 5, 0.03, 1)
    estimate_clouds_normals(pcds, voxel_size * 5, 30)

    poses = np.loadtxt('./data/poses.txt')[:, 1:]
    transforms = translations_quaternions_to_transforms(poses)
    pcds = transform_clouds_by_pose(pcds, transforms)

    mesh_frame = open3d.create_mesh_coordinate_frame(size = 0.5, origin = [0,0,0]) #original camera frame
    print("Showing initial cloud, pre-registration")
    o3d.draw_geometries(pcds + [mesh_frame])

    # source = 5
    # target = 6
    # # delta_transform = transforms[target] @ np.linalg.inv(transforms[source])
    # transform01, information01 = pairwise_registration(
    #     pcds[source], pcds[target], np.eye(4), icp_dist_coarse, icp_dist_fine)
    # pcds[source].transform(transform01)
    #
    # coord = o3d.create_mesh_coordinate_frame(0.2, poses[0, :3])
    # o3d.draw_geometries([*pcds[source:target+1], coord])
    # # o3d.draw_geometries([pcds[source], pcds[target], coord])

    # pcds = pcds[:6]
    pose_graph = build_pose_graph(pcds, transforms, icp_dist_coarse, icp_dist_fine)
    option = o3d.GlobalOptimizationOption(
        max_correspondence_distance=icp_dist_fine,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.global_optimization(pose_graph, o3d.GlobalOptimizationLevenbergMarquardt(),
                            o3d.GlobalOptimizationConvergenceCriteria(), option)

    for i in range(len(pcds)):
        pcds[i].transform(pose_graph.nodes[i].pose)

    o3d.draw_geometries(pcds)


def pairwise_registration(source, target, init_transform, dist_coarse, dist_fine):
    print("Apply point-to-plane ICP")
    # icp_coarse = o3d.registration_icp(
    #     source, target,
    #     dist_coarse, init_transform,
    #     o3d.TransformationEstimationPointToPlane())
    #
    # icp_fine = o3d.registration_icp(
    #     source, target,
    #     dist_fine, icp_coarse.transformation,
    #     o3d.TransformationEstimationPointToPlane())
    #
    # transformation_icp = icp_fine.transformation
    # information_icp = o3d.get_information_matrix_from_point_clouds(
    #     source, target, dist_fine,
    #     icp_fine.transformation)

    transformation_icp, _ = cicp.cicp(source, target, init_transform, dist_coarse, dist_fine)
    information_icp = o3d.get_information_matrix_from_point_clouds(
        source, target, dist_fine, transformation_icp)
    return transformation_icp, information_icp


def build_pose_graph(pcds, transforms, dist_coarse, dist_fine):
    pose_graph = o3d.PoseGraph()
    pose_graph.nodes.append(o3d.PoseGraphNode(transforms[0]))

    for i in range(1, len(pcds)):
        pose_graph.nodes.append(o3d.PoseGraphNode(transforms[i]))
        # odometry = transforms[i] @ np.linalg.inv(transforms[i-1])
        transform, information = pairwise_registration(pcds[i - 1], pcds[i], np.eye(4), dist_coarse, dist_fine)
        pose_graph.edges.append(o3d.PoseGraphEdge(i-1, i, transform, information, uncertain=False))

        # if i >= 2:
        #     transform, information = pairwise_registration(pcds[i - 2], pcds[i], np.eye(4), dist_coarse, dist_fine)
        #     pose_graph.edges.append(o3d.PoseGraphEdge(i - 2, i, transform, information, uncertain=True))

    transform, information = pairwise_registration(pcds[len(pcds)-1], pcds[0], np.eye(4), dist_coarse, dist_fine)
    pose_graph.edges.append(o3d.PoseGraphEdge(len(pcds)-1, 0, transform, information, uncertain=True))
    return pose_graph


def crop_clouds_by_depth(pcds, max_depth):
    for i in range(len(pcds)):
        points = np.asarray(pcds[i].points)
        mask = points[:, 2] < max_depth
        cropped_points = points[mask, :]

        colors = np.asarray(pcds[i].colors)
        cropped_colors = colors[mask, :]

        pcds[i].points = o3d.Vector3dVector(cropped_points)
        pcds[i].colors = o3d.Vector3dVector(cropped_colors)

    return pcds


def downsample_clouds(pcds, voxel_size):
    for i in range(len(pcds)):
        pcds[i] = o3d.voxel_down_sample(pcds[i], voxel_size)
    return pcds


def translations_quaternions_to_transforms(poses):
    transforms = []
    for pose in poses:
        t = pose[:3]
        q = pose[3:]

        T = np.eye(4)
        T[:3, :3] = quat2mat(q)
        T[:3, 3] = t
        transforms.append(T)
    return transforms


def transform_clouds_by_pose(pcds, transforms):
    for i in range(len(pcds)):
        # points_ref = np.asarray(pcds[i].points)
        # points_copy = np.array(pcds[i].points)
        # points_ref[:, :] = (transforms[i][:3, :3] @ points_copy.T + transforms[i][:3, 3].reshape(3, 1)).T
        pcds[i].transform(transforms[i])
    return pcds


def remove_clouds_outliers(pcds, num_points, radius, ratio=0):
    for i in range(len(pcds)):
        cl, ind = o3d.radius_outlier_removal(pcds[i], num_points, radius)
        pcds[i] = o3d.select_down_sample(pcds[i], ind)

        if ratio > 0:
            cl, ind = o3d.statistical_outlier_removal(pcds[i], 50, ratio)
            pcds[i] = o3d.select_down_sample(pcds[i], ind)

    return pcds


def estimate_clouds_normals(pcds, radius, max_nn):
    for i in range(len(pcds)):
        o3d.estimate_normals(
            pcds[i],
            search_param=o3d.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


if __name__ == '__main__':
    main()
