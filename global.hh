#ifndef GLOBAL_HH
#define GLOBAL_HH

#define VOXEL_SIZE 0.02
#define MAX_DEPTH 2
#define ICP_DIST_COARSE (VOXEL_SIZE * 15)
#define ICP_DIST_FINE (VOXEL_SIZE * 5)

#define ICP_ROTATION_AXIS Eigen::Vector3d(0, 0, 1)
#define ICP_TRANSLATION_AXIS0 Eigen::Vector3d(1, 0, 0)
#define ICP_TRANSLATION_AXIS1 Eigen::Vector3d(0, 1, 0)

#endif