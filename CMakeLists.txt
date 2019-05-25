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
  image_transport
  cv_bridge
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

link_directories(third_party/TensorRT-5.1.5.0/lib)

macro(add_nodelet node_name nodelet_cppfile)
  set(NODE_NAME ${node_name})
  set(NODELET_NAME onnx_tensorrt_ros/${node_name})
  configure_file(src/node/standalone_nodelet_exec.cpp.in ${node_name}.cpp @ONLY)
  add_executable(${node_name}_exe ${node_name}.cpp)
  SET_TARGET_PROPERTIES(${node_name}_exe PROPERTIES OUTPUT_NAME ${node_name})
  target_link_libraries(${node_name}_exe ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} nvinfer nvparsers nvonnxparser)
  list(APPEND _onnx_tensorrt_ros_nodelet_cppfiles ${nodelet_cppfile})
  list(APPEND _onnx_tensorrt_ros_nodelet_targets ${node_name}_exe)
endmacro()

add_nodelet(onnx_tensorrt_yolov3 src/nodelet/onnx_tensorrt_yolov3_nodelet.cpp third_party/TensorRT-5.1.5.0/samples/common/logger.cpp)

add_library(${PROJECT_NAME} SHARED
  src/nodelet/nodelet.cpp
  ${_onnx_tensorrt_ros_nodelet_cppfiles}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} nvinfer nvparsers nvonnxparser)

set(FILE "${CMAKE_SOURCE_DIR}/data/yolov3.onnx")
message(STATUS "Checking and downloading yolov3.onnx if needed ...")
if (NOT EXISTS "${FILE}")
  message(STATUS "... file does not exist. Downloading now ...")
  execute_process(COMMAND wget https://onnxzoo.blob.core.windows.net/models/opset_10/yolov3/yolov3.onnx -o ${CMAKE_SOURCE_DIR}/data/yolov3.onnx)
endif()

# Install
install(TARGETS ${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS ${_onnx_tensorrt_ros_nodelet_targets}
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch data third_party
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        USE_SOURCE_PERMISSIONS)