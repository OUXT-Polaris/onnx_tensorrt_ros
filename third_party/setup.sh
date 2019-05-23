tar -xvzf TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz 
cd TensorRT-5.1.5.0/lib
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD >> ~/.bashrc
cd ../