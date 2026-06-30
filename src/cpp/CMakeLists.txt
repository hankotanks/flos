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
    segmentation_node.cpp
    segmentation/system.cpp
    util/cloud.cpp
    util/voxel.cpp
  HEADERS
    segmentation/system.h
    util/buffer_node.h
    util/cloud.h
    util/voxel.h
    util/stl_vector_eigen.h
    util/rosargs.h
    util/profiler.h
  LIBRARIES
    Eigen3::Eigen
    pybind11::embed
    tsl::robin_map
  PACKAGES
    rclcpp::rclcpp
    ${std_msgs_TARGETS}
    ${sensor_msgs_TARGETS})

add_executable(odometry_node)
configure_node(odometry_node
  SOURCES
    odometry_node.cpp
    odometry/system.cpp
    odometry/wrapper.cpp
    util/cloud.cpp
    util/voxel.cpp
  HEADERS
    odometry/voxel_map.h
    odometry/system.h
    odometry/frame.h
    odometry/wrapper.h
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

add_executable(offline_node)
configure_node(offline_node
  SOURCES
    offline_node.cpp
    segmentation/system.cpp
    odometry/system.cpp
    odometry/wrapper.cpp
    util/cloud.cpp
    util/voxel.cpp
    util/rosbag.cpp 
  HEADERS
    segmentation/system.h
    odometry/voxel_map.h
    odometry/system.h
    odometry/frame.h
    odometry/wrapper.h
    util/buffer_node.h
    util/cloud.h
    util/voxel.h
    util/stl_vector_eigen.h
    util/profiler.h
    util/rosargs.h
    util/rosbag.h
  LIBRARIES
    Eigen3::Eigen
    pybind11::embed
    tsl::robin_map
    Sophus::Sophus
    TBB::tbb
  PACKAGES
    rclcpp::rclcpp
    tf2_ros::tf2_ros
    rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage
    ${std_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${nav_msgs_TARGETS})

install(TARGETS segmentation_node odometry_node offline_node
  DESTINATION lib/${PROJECT_NAME})