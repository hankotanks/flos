add_executable(segmentation_node segmentation_node.cpp forest_segmentor_bindings.cpp utils.cpp)
target_include_directories(segmentation_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(segmentation_node
  PRIVATE ${catkin_LIBRARIES} pybind11::embed Eigen3::Eigen)

target_compile_definitions(segmentation_node PRIVATE 
  PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}" 
  PROJECT_NAME="${PROJECT_NAME}")

install(TARGETS segmentation_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})