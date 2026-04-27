cmake_minimum_required(VERSION 3.16)
project(forest_segmentation VERSION 0.0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(StandardProjectOptions)
include(dependencies)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_custom_target(python_package ALL 
  COMMAND pip3 install --root-user-action=ignore ${CMAKE_CURRENT_SOURCE_DIR}/src/python
  COMMENT "${PROJECT_NAME}: Installing Python package..." 
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(src/cpp)

install(DIRECTORY config models DESTINATION share/${PROJECT_NAME})

ament_package()