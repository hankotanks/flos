add_executable(segmentation_node segmentation_node.cpp forest_segmentor_bindings.cpp util/cloud.cpp)
target_include_directories(segmentation_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(segmentation_node PUBLIC rclcpp::rclcpp sensor_msgs::sensor_msgs_library)
target_link_libraries(segmentation_node PRIVATE pybind11::embed Eigen3::Eigen)

target_compile_definitions(segmentation_node PRIVATE
  PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
  PROJECT_NAME="${PROJECT_NAME}")

install(TARGETS segmentation_node DESTINATION lib/${PROJECT_NAME})