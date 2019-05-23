cmake_minimum_required(VERSION 2.8.3)
project(onnx_tensorrt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  message_filters
  roscpp
  sensor_msgs
  vision_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES onnx_tensorrt_ros
  CATKIN_DEPENDS message_filters roscpp sensor_msgs vision_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)