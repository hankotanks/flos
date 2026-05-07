add_executable(segmentation_node segmentation_node.cpp util/base_node.cpp util/cloud.cpp)
target_include_directories(segmentation_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(segmentation_node PUBLIC rclcpp::rclcpp 
  sensor_msgs::sensor_msgs_library)
target_link_libraries(segmentation_node PRIVATE pybind11::embed Eigen3::Eigen)
target_compile_definitions(segmentation_node PRIVATE
  PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
  PROJECT_NAME="${PROJECT_NAME}"
  GENERATE_CLOUD_PYBIND_MODULE)
target_compile_options(segmentation_node PRIVATE -Wno-alloc-size-larger-than)

# TODO: factor common stuff into a function to avoid code duplication
add_executable(segmentation_node_offline segmentation_node_offline.cpp util/rosbag.cpp util/base_node.cpp util/cloud.cpp)
target_include_directories(segmentation_node_offline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(segmentation_node_offline PUBLIC rclcpp::rclcpp 
  sensor_msgs::sensor_msgs_library
  tf2::tf2 tf2_ros::tf2_ros
  rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage)
target_link_libraries(segmentation_node_offline PRIVATE pybind11::embed Eigen3::Eigen)
target_compile_definitions(segmentation_node_offline PRIVATE
  PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
  PROJECT_NAME="${PROJECT_NAME}"
  GENERATE_CLOUD_PYBIND_MODULE)
target_compile_options(segmentation_node_offline PRIVATE -Wno-alloc-size-larger-than)

install(TARGETS segmentation_node segmentation_node_offline DESTINATION lib/${PROJECT_NAME})