function(configure_node name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES HEADERS LIBRARIES PACKAGES DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  target_sources(${name}
    PRIVATE ${ARG_SOURCES}
    PRIVATE FILE_SET HEADERS FILES ${ARG_HEADERS})
  target_link_libraries(${name} ${ARG_LIBRARIES} ${ARG_PACKAGES})
  target_compile_definitions(${name}
    PRIVATE
      PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
      PROJECT_NAME="${PROJECT_NAME}"
      NODE_NAME="${name}"
      ${ARG_DEFINITIONS})
  target_compile_options(${name} PRIVATE -Wno-alloc-size-larger-than)
endfunction()

#
# COMMON
#

set(COMMON_SOURCES)

set(COMMON_HEADERS
  classes.h
  util/buffer_node.h
  util/cloud.h
  util/profiler.h
  util/rosargs.h
  util/rosbag.h
  util/stl_vector_eigen.h
  util/tinyply.h
  util/voxel.h)

set(COMMON_LIBRARIES
  Eigen3::Eigen
  tsl::robin_map)

set(COMMON_PACKAGES
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS})

set(COMMON_DEFINITIONS)

#
# SEGMENTATION
#

set(SEGMENTATION_SOURCES
  segmentation/system.cpp)

set(SEGMENTATION_HEADERS
  segmentation/system.h
  segmentation_node_config.h)

set(SEGMENTATION_LIBRARIES
  pybind11::embed)

set(SEGMENTATION_PACKAGES)

set(SEGMENTATION_DEFINITIONS
  GENERATE_CLOUD_PYBIND_MODULE)

add_executable(segmentation_node)
configure_node(segmentation_node
  SOURCES 
    segmentation_node.cpp
    ${COMMON_SOURCES}
    ${SEGMENTATION_SOURCES}
    util/cloud.cpp
    util/rosargs.cpp
  HEADERS
    ${COMMON_HEADERS}
    ${SEGMENTATION_HEADERS}
  LIBRARIES
    ${COMMON_LIBRARIES}
    ${SEGMENTATION_LIBRARIES}
  PACKAGES
    ${COMMON_PACKAGES}
    ${SEGMENTATION_PACKAGES}
  DEFINITIONS
    ${COMMON_DEFINITIONS}
    ${SEGMENTATION_DEFINITIONS})

#
# ODOMETRY
#

set(ODOMETRY_SOURCES
  odometry/system.cpp
  odometry/wrapper.cpp)

set(ODOMETRY_HEADERS
  odometry/system.h
  odometry/wrapper.h
  odometry/frame.h
  odometry/voxel_map.h
  odometry_node_config.h)

set(ODOMETRY_LIBRARIES
  Sophus::Sophus
  TBB::tbb)

set(ODOMETRY_PACKAGES
  tf2_ros::tf2_ros
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS})

set(ODOMETRY_DEFINITIONS)

add_executable(odometry_node)
configure_node(odometry_node
  SOURCES
    odometry_node.cpp
    ${COMMON_SOURCES}
    ${ODOMETRY_SOURCES}
    util/cloud.cpp
    util/rosargs.cpp
  HEADERS
    ${COMMON_HEADERS}
    ${ODOMETRY_HEADERS}
  LIBRARIES
    ${COMMON_LIBRARIES}
    ${ODOMETRY_LIBRARIES}
  PACKAGES
    ${COMMON_PACKAGES}
    ${ODOMETRY_PACKAGES}
  DEFINITIONS
    ${COMMON_DEFINITIONS}
    ${ODOMETRY_DEFINITIONS})

#
# OFFLINE
#

set(OFFLINE_SOURCES
  ${COMMON_SOURCES}
  ${SEGMENTATION_SOURCES}
  ${ODOMETRY_SOURCES}
  dataloader/dataloader.cpp
  dataloader/bag.cpp
  dataloader/wild_scenes.cpp
  util/cloud.cpp
  util/rosargs.cpp
  util/rosbag.cpp 
  util/tinyply.cpp)

set(OFFLINE_HEADERS
  ${COMMON_HEADERS}
  ${SEGMENTATION_HEADERS}
  ${ODOMETRY_HEADERS}
  offline_node.h
  dataloader/dataloader.h)

set(OFFLINE_LIBRARIES
  ${COMMON_LIBRARIES}
  ${SEGMENTATION_LIBRARIES}
  ${ODOMETRY_LIBRARIES})

set(OFFLINE_PACKAGES
  ${COMMON_PACKAGES}
  ${SEGMENTATION_PACKAGES}
  ${ODOMETRY_PACKAGES}
  rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage)

set(OFFLINE_DEFINITIONS
  ${COMMON_DEFINITIONS}
  ${SEGMENTATION_DEFINITIONS}
  ${ODOMETRY_DEFINITIONS})

add_executable(offline_node)
configure_node(offline_node
  SOURCES
    offline_node.cpp
    ${OFFLINE_SOURCES}
  HEADERS
    ${OFFLINE_HEADERS}
  LIBRARIES
    ${OFFLINE_LIBRARIES}
  PACKAGES
    ${OFFLINE_PACKAGES}
  DEFINITIONS
    ${OFFLINE_DEFINITIONS})

add_executable(rerun_node)
configure_node(rerun_node
  SOURCES
    rerun_node.cpp
    ${OFFLINE_SOURCES}
  HEADERS
    ${OFFLINE_HEADERS}
  LIBRARIES
    ${OFFLINE_LIBRARIES}
    rerun_sdk
  PACKAGES
    ${OFFLINE_PACKAGES}
  DEFINITIONS
    ${OFFLINE_DEFINITIONS})

#
# INSTALLATION
#

install(TARGETS segmentation_node odometry_node offline_node rerun_node
  DESTINATION lib/${PROJECT_NAME})