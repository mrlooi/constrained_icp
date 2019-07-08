cmake_minimum_required(VERSION 3.0.0)
project(constrained_icp VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common filters)
find_package(Open3D REQUIRED HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
find_package(Ceres REQUIRED)

include_directories(
    ${Open3D_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    "include"
    )
# link_directories must be before add_executable
link_directories(
    ${Open3D_LIBRARY_DIRS}
    ${PCL_LIBRARY_DIRS}
    )

add_definitions(${PCL_DEFINITIONS})
# message(STATUS  sssss->${CERES_LIBRARIES})

include(CTest)
enable_testing()

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES} ${CERES_LIBRARIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
