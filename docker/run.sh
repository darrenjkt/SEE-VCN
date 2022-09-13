#!/bin/bash

DATA_PATH="/mnt/big-data/darren/data"
CODE_PATH="/mnt/big-data/darren/code/SEE-VCN"
GPU_ID="0,1"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

# Modify these paths as necessary to mount the data
# Passing in /pcdet might require running python setup.py develop
# every time you switch gpus...
VOLUMES="       --volume=$DATA_PATH/kitti/3d_object_detection:/SEE-VCN/data/kitti
                --volume=$DATA_PATH/nuscenes:/SEE-VCN/data/nuscenes
                --volume=$DATA_PATH/waymo:/SEE-VCN/data/waymo
                --volume=$DATA_PATH/shapenet:/SEE-VCN/data/shapenet
                --volume=$DATA_PATH/custom_data:/SEE-VCN/data/custom_data
                --volume=$CODE_PATH:/SEE-VCN"

# Setup visualization for point cloud demos
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"

# Start docker image
xhost +local:docker

echo "Running the docker image [GPUS: ${GPU_ID}]"
docker_image="darrenjkt/see-vcn:1.0"

docker  run -d -it --rm \
$VOLUMES \
$ENVS \
$VISUAL \
--runtime=nvidia \
--gpus $GPU_ID \
--privileged \
--net=host \
--shm-size=16G \
--workdir=/SEE-VCN \
$docker_image   
