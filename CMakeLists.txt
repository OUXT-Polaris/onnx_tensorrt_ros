cmake_minimum_required(VERSION 2.8.3)
project(onnx_tensorrt_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  message_filters
  roscpp
  sensor_msgs
  vision_msgs
  pluginlib
  nodelet
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES onnx_tensorrt_ros
  CATKIN_DEPENDS message_filters roscpp sensor_msgs vision_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  third_party/TensorRT-5.1.5.0/include
  third_party/TensorRT-5.1.5.0/samples/common
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

macro(add_nodelet node_name nodelet_cppfile)
  set(NODE_NAME ${node_name})
  set(NODELET_NAME onnx_tensorrt_ros/${node_name})
  configure_file(src/node/standalone_nodelet_exec.cpp.in ${node_name}.cpp @ONLY)
  add_executable(${node_name}_exe ${node_name}.cpp)
  SET_TARGET_PROPERTIES(${node_name}_exe PROPERTIES OUTPUT_NAME ${node_name})
  target_link_libraries(${node_name}_exe ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})
  list(APPEND _onnx_tensorrt_ros_nodelet_cppfiles ${nodelet_cppfile})
  list(APPEND _onnx_tensorrt_ros_nodelet_targets ${node_name}_exe)
endmacro()

add_nodelet(onnx_tensorrt_ros src/nodelet/onnx_tensorrt_ros_nodelet.cpp third_party/TensorRT-5.1.5.0/samples/common/logger.cpp)

add_library(${PROJECT_NAME} SHARED
  src/nodelet/nodelet.cpp
  ${_onnx_tensorrt_ros_nodelet_cppfiles}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)