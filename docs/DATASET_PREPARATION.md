# Dataset Preparation

### Baraja Spectrum-Scan™ Dataset
Please download the [Baraja Spectrum-Scan™ Dataset](https://drive.google.com/file/d/16_azaVGiMVycGH799FX2RyRIWHrslU0R/view?usp=sharing) and organise the downloaded files as follows:
```
SEE-VCN
├── data
│   ├── baraja
│   │   │── ImageSets
│   │   │── test
│   │   │   ├──pcd & masks & image & calib & label
│   │   │── infos
│   │   │   ├──baraja_infos_test.pkl
...
```

### KITTI, Waymo, nuScenes
For KITTI, Waymo, nuScenes datasets, we follow closely to the data preparation of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). 

For training of our networks, we use a subset of the nuScenes and Waymo datasets.
- **nuScenes.** We sort the dataset by number of cars in each scene and choose the top 100 scenes. This give us 4025 frames for training. 
- **Waymo.** We use every 30th frame to reduce the training data size. This gives us 5267 frames.

For testing on nuScenes and KITTI, we use the original validation set.

### KITTI, Waymo, nuScenes
For KITTI, Waymo, nuScenes datasets, we follow closely to the data preparation of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). 

For training of our networks, we use a subset of the nuScenes and Waymo datasets.
- **nuScenes.** We sort the dataset by number of cars in each scene and choose the top 100 scenes. This give us 4025 frames for training. For testing on nuScenes, we use the original validation set.
- **Waymo.** We use every 30th frame to reduce the training data size. This gives us 5267 frames.

### VC-ShapeNet
You don't need this dataset as we provide the pretrained models, however if you'd like to download our VC-ShapeNet dataset, you can download it [here](todo_put_a_link).

You can also download our kitti/nuscenes/waymo test dataset [here](todo_put_a_link).

