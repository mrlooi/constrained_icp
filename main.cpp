#include <iostream>
#include <Open3D/Visualization/Visualization.h>
#include <Open3D/IO/IO.h>
#include <Open3D/Core/Registration/GlobalOptimization.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <iostream>
#include <fstream>

#include "global.hh"
#include "constrained-icp.hpp"


void crop_cloud_by_depth(open3d::PointCloud& pcd, float depth)
{
    std::vector<Eigen::Vector3d> cropped_points;
    std::vector<Eigen::Vector3d> cropped_colors;

    for (int i = 0; i < pcd.points_.size(); ++i)
    {
        if (pcd.points_[i].z() < depth)
        {
            cropped_points.push_back(pcd.points_[i]);
            cropped_colors.push_back(pcd.colors_[i]);
        }
    }

    pcd.points_ = std::move(cropped_points);
    pcd.colors_ = std::move(cropped_colors);
}

std::vector<Eigen::Matrix4d> load_cloud_poses(const std::string& file)
{
    std::vector<Eigen::Matrix4d> poses;
    std::ifstream ifs(file, std::ifstream::in);
    std::string line;
    while (std::getline(ifs, line))
    {
        int id;
        float x, y, z, qw, qx, qy, qz;
        std::istringstream iss(line);
        iss >> id >> x >> y >> z >> qw >> qx >> qy >> qz;
        Eigen::Vector3d t(x, y, z);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3, 3>(0, 0) = q.matrix();
        pose.block<3, 1>(0, 3) = t;

        poses.push_back(pose);
    }
    ifs.close();

    return std::move(poses);
}

void remove_cloud_outliers(open3d::PointCloud& pcd, int num_points, double radius, double ratio)
{
    std::shared_ptr<open3d::PointCloud> cloud;
    std::vector<size_t> indices;
    std::tie(cloud, indices) = open3d::RemoveRadiusOutliers(pcd, num_points, radius);
    cloud = open3d::SelectDownSample(pcd, indices);
    pcd = *cloud;

    if (ratio > 0)
    {
        std::tie(cloud, indices) = open3d::RemoveStatisticalOutliers(pcd, 50, ratio);
        cloud = open3d::SelectDownSample(pcd, indices);
        pcd = *cloud;
    }
}

void constrained_icp_single_step(open3d::PointCloud& source, open3d::PointCloud& target, 
    Eigen::Matrix4d& base_transform, double max_correspond_dist, double* x)
{
    int num_points = source.points_.size();
    ConstrainedICPCost* icp_cost = new ConstrainedICPCost(
                source, target, base_transform, ICP_ROTATION_AXIS, ICP_TRANSLATION_AXIS0, ICP_TRANSLATION_AXIS1, max_correspond_dist);
    ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(1.0);
    // ceres::LossFunction* loss_function = nullptr;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1;

    ceres::Problem problem;
    ceres::Solver::Summary summary;
    ceres::CostFunction* cost_function = 
        new ceres::NumericDiffCostFunction<ConstrainedICPCost, ceres::FORWARD, ceres::DYNAMIC, 3>(icp_cost, ceres::TAKE_OWNERSHIP, num_points);
    problem.AddResidualBlock(cost_function, loss_function, x);

    ceres::Solve(options, &problem, &summary);
}

std::pair<Eigen::Matrix4d, Eigen::Matrix6d>
pairwise_registration(open3d::PointCloud& source, open3d::PointCloud& target, Eigen::Matrix4d& base_transform)
{   
    std::cout << "Apply constrained point-to-plane ICP..." << std::endl;
    
    Eigen::Matrix4d transform = base_transform;

    double x[3] = {0, 0, 0};
    open3d::PointCloud pcd = source;
    pcd.Transform(base_transform);

    Eigen::Matrix4d I = Eigen::Matrix4d::Identity();

    for (int i = 0; i < 5; ++i)
    {
        constrained_icp_single_step(pcd, target, I, ICP_DIST_COARSE, x);

        double angle = x[0];
        double t0 = x[1];
        double t1 = x[2];

        Eigen::Matrix4d update = Eigen::Matrix4d::Identity();
        Eigen::AngleAxisd angle_axis(angle, ICP_ROTATION_AXIS);
        update.block<3, 3>(0, 0) = angle_axis.matrix();
        update.block<3, 1>(0, 3) = t0 * ICP_TRANSLATION_AXIS0 + t1 * ICP_TRANSLATION_AXIS1;
        pcd.Transform(update);
        transform = update * transform;
    }
    
    for (int i = 0; i < 10; ++i)
    {
        x[0] = x[1] = x[2] = 0.0;
        constrained_icp_single_step(pcd, target, I, ICP_DIST_FINE, x);

        double angle = x[0];
        double t0 = x[1];
        double t1 = x[2];

        // Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d update = Eigen::Matrix4d::Identity();
        Eigen::AngleAxisd angle_axis(angle, ICP_ROTATION_AXIS);
        update.block<3, 3>(0, 0) = angle_axis.matrix();
        update.block<3, 1>(0, 3) = t0 * ICP_TRANSLATION_AXIS0 + t1 * ICP_TRANSLATION_AXIS1;
        pcd.Transform(update);
        transform = update * transform;
    }

    // constrained_icp(source, target, base_transform, ICP_DIST_COARSE, x);
    // constrained_icp(source, target, base_transform, ICP_DIST_FINE, x);

    Eigen::Matrix6d information = open3d::GetInformationMatrixFromPointClouds(source, target, ICP_DIST_FINE, transform);

    return std::move(std::make_pair(transform, information));
}

open3d::PoseGraph build_pose_graph(std::vector<std::shared_ptr<open3d::PointCloud>>& pcds, std::vector<Eigen::Matrix4d>& poses,
    std::vector<std::shared_ptr<const open3d::Geometry>>& geometries)
{
    open3d::PoseGraph pose_graph;
    pose_graph.nodes_.push_back(open3d::PoseGraphNode(poses[0]));

    Eigen::Matrix4d base_transform = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d transform;
    Eigen::Matrix6d information;
    for (int i = 1; i < poses.size(); ++i)
    {
        pose_graph.nodes_.push_back(open3d::PoseGraphNode(poses[i]));

        std::tie(transform, information) = pairwise_registration(*pcds[i-1], *pcds[i], base_transform);

        // Just for debug
        // pcds[i-1]->Transform(transform);
        // auto g = std::vector<std::shared_ptr<const open3d::Geometry>>{geometries[i-1], geometries[i]};
        // open3d::DrawGeometries(g);

        pose_graph.edges_.push_back(open3d::PoseGraphEdge(i-1, i, transform, information, false));
    }

    std::tie(transform, information) = pairwise_registration(*pcds.back(), *pcds[0], base_transform);
    pose_graph.edges_.push_back(open3d::PoseGraphEdge(pcds.size()-1, 0, transform, information, true));

    return std::move(pose_graph);
}

int main(int, char**) {

    std::vector<std::shared_ptr<const open3d::Geometry>> geometries;
    std::vector<std::shared_ptr<open3d::PointCloud>> pcds;

    std::vector<Eigen::Matrix4d> poses = load_cloud_poses("../data/poses.txt");

    for (int i = 0; i < poses.size(); ++i)
    {
        std::shared_ptr<open3d::PointCloud> pcd = std::make_shared<open3d::PointCloud>();
        std::string pcd_file = "../data/pcd_" + std::to_string(i) + ".pcd";
        printf("Reading %s...\n", pcd_file.c_str());
        open3d::ReadPointCloud(pcd_file, *pcd);

        crop_cloud_by_depth(*pcd, MAX_DEPTH);
        remove_cloud_outliers(*pcd, 30, VOXEL_SIZE, 1);
        pcd = open3d::VoxelDownSample(*pcd, VOXEL_SIZE);
        open3d::EstimateNormals(*pcd, open3d::KDTreeSearchParamHybrid(VOXEL_SIZE * 5, 30));
        pcd->Transform(poses[i]);

        geometries.push_back(pcd);
        pcds.push_back(pcd);
    }

    open3d::PoseGraph pose_graph = build_pose_graph(pcds, poses, geometries);
    open3d::GlobalOptimizationOption option(ICP_DIST_FINE, 0.25, 1.0, 0);
    open3d::GlobalOptimization(
        pose_graph, 
        open3d::GlobalOptimizationLevenbergMarquardt(), 
        open3d::GlobalOptimizationConvergenceCriteria(),
        option);
    
    for (int i = 0; i < poses.size(); ++i)
    {
        pcds[i]->Transform(pose_graph.nodes_[i].pose_);
    }

    open3d::DrawGeometries(geometries);
}
