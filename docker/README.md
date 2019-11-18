### Running temporal-shift module using the docker image

#### Build docker

`docker build -t temporal_shift_module/pytorch1.2:jiaming_huang -f Dockerfile .`

#### Run docker

We only consider the gpu case, and you have to install nvidia-docker. You can reference to [github](https://github.com/NVIDIA/nvidia-docker).

`docker run --gpus all --name tsm -p 6006:6006 --shm-size 8G 
-v xxx/20bn-jester-v1/:/workspace/datas/jester/20bn-jester-v1 
-it temporal_shift_module/pytorch1.2:jiaming_huang`
