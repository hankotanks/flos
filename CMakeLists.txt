cmake_minimum_required(VERSION 3.16)
project(forest_segmentation VERSION 0.0.1 LANGUAGES CXX)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(StandardProjectOptions)
include(dependencies)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)

add_custom_target(python_requirements ALL
  COMMAND python3 -m pip install --root-user-action=ignore -r ${CMAKE_CURRENT_SOURCE_DIR}/src/python/requirements.txt
  COMMENT "${PROJECT_NAME}: Installing Python dependencies..."
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(python_package ALL 
  COMMAND pip3 install --root-user-action=ignore ${CMAKE_CURRENT_SOURCE_DIR}/src/python
  COMMENT "${PROJECT_NAME}: Installing Python package..." 
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(src/cpp)

install(DIRECTORY config models DESTINATION share/${PROJECT_NAME})

ament_package()