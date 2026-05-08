set(SHARED_SOURCES
  buffer_node.cpp
  base_node.cpp
  util/cloud.cpp)

set(SHARED_HEADERS
  buffer_node.h
  base_node.h
  util/cloud.h
  util/stl_vector_eigen.h)

set(SHARED_LIBRARIES
  Eigen3::Eigen
  pybind11::embed
  tsl::robin_map)

set(SHARED_PACKAGES
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library)

function(configure_node name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES HEADERS PACKAGES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${name})
  target_sources(${name}
    PRIVATE "${name}.cpp" ${SHARED_SOURCES} ${ARG_SOURCES}
    PRIVATE FILE_SET HEADERS FILES ${SHARED_HEADERS} ${ARG_HEADERS})
  target_link_libraries(${name} 
    PRIVATE ${SHARED_LIBRARIES} ${SHARED_PACKAGES} ${ARG_PACKAGES})
  target_compile_definitions(${name} 
    PRIVATE
      PROJECT_ROOT="${CMAKE_CURRENT_LIST_DIR}"
      PROJECT_NAME="${PROJECT_NAME}"
      GENERATE_CLOUD_PYBIND_MODULE)
  target_compile_options(${name} PRIVATE -Wno-alloc-size-larger-than)
endfunction()

configure_node(segmentation_node)
configure_node(segmentation_node_offline 
  SOURCES 
    util/rosbag.cpp 
  HEADERS
    util/rosbag.h
    util/profiler.h
  PACKAGES 
    tf2::tf2 tf2_ros::tf2_ros 
    rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage)

install(TARGETS segmentation_node segmentation_node_offline DESTINATION lib/${PROJECT_NAME})