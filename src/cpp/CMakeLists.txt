add_executable(mapping_node inference_node.cpp pybindings.cpp)
target_include_directories(mapping_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(
  mapping_node
  PRIVATE ${catkin_LIBRARIES}
          pybind11::embed
          Eigen3::Eigen)

install(TARGETS mapping_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
