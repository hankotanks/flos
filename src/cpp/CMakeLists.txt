add_executable(segmentation_node segmentation_node.cpp forest_segmentor_bindings.cpp utils.cpp)
target_include_directories(segmentation_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ament_target_dependencies(segmentation_node rclcpp sensor_msgs)
target_link_libraries(segmentation_node pybind11::embed Eigen3::Eigen)

target_compile_definitions(segmentation_node PRIVATE
  PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
  PROJECT_NAME="${PROJECT_NAME}")

install(TARGETS segmentation_node DESTINATION lib/${PROJECT_NAME})
