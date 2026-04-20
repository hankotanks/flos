add_executable(segmentation_node segmentation_node.cpp forest_segmentor_bindings.cpp utils.cpp)
target_include_directories(segmentation_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(segmentation_node
  PRIVATE ${catkin_LIBRARIES} pybind11::embed Eigen3::Eigen)

install(TARGETS segmentation_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})