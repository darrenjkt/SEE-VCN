# Viewer-centred Completion Network (VCN)

This is the standalone code for training of the viewer-centred completion network (VCN) in [SEE-VCN](https://github.com/darrenjkt/SEE-VCN). 

VCN is a PointNet based model that can complete the point clouds of object as captured in the wild by a lidar sensor. The coordinates of such objects are relative to the view-point of the sensor frame, which we call viewer-centred coordinates. Given an object's points, we can estimate it's pose and complete the surface without requiring pre-canonicalization like other point cloud completion methods. VCN runs at 0.32ms/car. 

![architecture](/SEE-VCN/docs/vcn_architecture.png)

## Install
```
pip install -e . --user
```

## Datasets
- **VC-ShapeNet** [[download](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EeB6XBooMkdArv1xgRVMja0BcVvt63C2vzHTi-PjAnpQzQ?e=iyhPWj)]: Viewer-centred surface car dataset. Cars were positioned in viewer-centred frame using waymo labels and raycasted to obtain occluded cars in realistic scenes.
- **Lidar test set** [[download](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/Ect6piVGprBJsrymXueeHooBiQAn7z2hxUelpECDQOyS3Q?e=H6Jc58)]: We randomly select 5000 cars from KITTI, nuScenes and Waymo each for evaluation

## Usage
We provide a [Demo notebook](https://github.com/darrenjkt/VCN/blob/main/demo.ipynb) with some demo data for quickstart.

### Training
```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
    
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
For example: 
```
# Train a model with 2 gpus
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/VCN_models/VCN_VC.yaml \
    --exp_name exp01
    
# Resume model training
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/VCN_models/VCN_VC.yaml \
    --exp_name exp01 --resume
```

### Testing
```
bash ./scripts/test.sh 0 \
    --ckpts ./model_zoo/VCN_VC.pth \
    --config ./cfgs/VCN_models/VCN_VC.yaml \
    --exp_name exp01
```

### Acknowledgements
Our code is built on the repository of [PoinTr](https://github.com/yuxumin/PoinTr).
