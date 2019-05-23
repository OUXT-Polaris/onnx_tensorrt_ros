# onnx_tensorrt_ros package

## Requirements
Ubuntu : 18.04  
ROS : Melodic  
cuda 10.0
cudnn 7.5
nvidia-docker 2.0

## How to Setup
### Download and Install TensorRT

https://developer.nvidia.com/nvidia-tensorrt-5x-download

https://developer.nvidia.com/compute/machine-learning/tensorrt/5.1/ga/tars/TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz

### Put downloaded tar into this package 
put TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz into third_party directory in this package and unzip

```
tar -xvzf TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz
```