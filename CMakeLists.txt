cmake_minimum_required(VERSION 3.16)
project(forest_segmentation VERSION 0.0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(StandardProjectOptions)
include(dependencies)

find_package(
  catkin REQUIRED
  COMPONENTS nav_msgs
             roscpp
             tf2
             std_msgs
             tf2_ros
             sensor_msgs
             geometry_msgs
             rospy)

catkin_package()

add_custom_target(python_package ALL COMMAND pip3 install --user ${CMAKE_CURRENT_SOURCE_DIR}/src/python
                  COMMENT "Installing Python package..." WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(src/cpp)

install(DIRECTORY config launch models DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
