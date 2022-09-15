# Dataset Preparation

### Baraja Spectrum-Scan™ Dataset
Please download the [Baraja Spectrum-Scan™ Dataset](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EbBLKPoamxJGh6gmTAAv9hgBqo0w_d7JrHOfCzitZ8xI5Q?e=cP3uwH) and organise the downloaded files as follows:
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

### VC-ShapeNet
You don't need this dataset for SEE-VCN as we provide the pretrained models, however if you'd like to download our VC-ShapeNet dataset, you can download it [here](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/EeB6XBooMkdArv1xgRVMja0BcVvt63C2vzHTi-PjAnpQzQ?e=iyhPWj).
You can also download our kitti/nuscenes/waymo test dataset [here](https://unisyd-my.sharepoint.com/:u:/g/personal/julie_berrioperez_sydney_edu_au/Ect6piVGprBJsrymXueeHooBiQAn7z2hxUelpECDQOyS3Q?e=H6Jc58).