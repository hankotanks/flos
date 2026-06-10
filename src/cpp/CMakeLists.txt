set(SEGMENTATION_SOURCES
  segmentation/node.cpp
  util/cloud.cpp
  util/voxel.cpp)

set(SEGMENTATION_HEADERS
  segmentation/node.h
  util/buffer_node.h
  util/cloud.h
  util/voxel.h
  util/stl_vector_eigen.h
  util/rosargs.h)

set(SEGMENTATION_LIBRARIES
  Eigen3::Eigen
  pybind11::embed
  tsl::robin_map)

set(SEGMENTATION_PACKAGES
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${sensor_msgs_TARGETS})

function(configure_node name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES HEADERS LIBRARIES PACKAGES)
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
      GENERATE_CLOUD_PYBIND_MODULE)
  target_compile_options(${name} PRIVATE -Wno-alloc-size-larger-than)
endfunction()

add_executable(segmentation_node)
configure_node(segmentation_node
  SOURCES 
    ${SEGMENTATION_SOURCES}
    segmentation/node_online.cpp
  HEADERS
    ${SEGMENTATION_HEADERS}
  LIBRARIES
    ${SEGMENTATION_LIBRARIES}
  PACKAGES
    ${SEGMENTATION_PACKAGES})

add_executable(segmentation_node_offline)
configure_node(segmentation_node_offline 
  SOURCES 
    ${SEGMENTATION_SOURCES}
    segmentation/node_offline.cpp
    util/rosbag.cpp 
  HEADERS
    ${SEGMENTATION_HEADERS}
    util/rosbag.h
    util/profiler.h
    util/delayed_publisher.h
  LIBRARIES
    ${SEGMENTATION_LIBRARIES}
  PACKAGES 
    ${SEGMENTATION_PACKAGES}
    ${nav_msgs_TARGETS}
    tf2::tf2 tf2_ros::tf2_ros 
    rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage)

add_executable(odometry_node)
configure_node(odometry_node
  SOURCES
    odom/node.cpp
    odom/voxel_map.cpp
    odom/system.cpp
    util/cloud.cpp
    util/voxel.cpp
  HEADERS
    odom/voxel_map.h
    odom/system.h
    util/cloud.h
    util/voxel.h
    util/profiler.h
    util/rosargs.h
  LIBRARIES
    Eigen3::Eigen
    pybind11::embed
    tsl::robin_map
    Sophus::Sophus
    TBB::tbb
  PACKAGES
    rclcpp::rclcpp
    tf2_ros::tf2_ros
    ${std_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${nav_msgs_TARGETS})

add_executable(recorder_node)
configure_node(recorder_node
  SOURCES
    recorder/node.cpp
  LIBRARIES
    Sophus::Sophus
  PACKAGES
    rclcpp::rclcpp
    ${std_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${nav_msgs_TARGETS})

install(TARGETS segmentation_node segmentation_node_offline odometry_node recorder_node
  DESTINATION lib/${PROJECT_NAME})