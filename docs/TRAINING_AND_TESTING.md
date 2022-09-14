# Training and Testing

## SEE-VCN Data Augmentation 

First we need to download the pretrained VCN models:
```
bash model_zoo/download_vcn_model.sh
```
### Training
For training data (source domain), please use the `VCN-CN` configuration to complete the objects. VCN-CN uses the source labels to canonicalize the objects before completing them. 
```
cd see/surface_completion && python sc_multiproc.py --cfg_file cfgs/${*}_GT_VCN-CN.yaml
```

### Testing

For test data (target domain), we need to first get the image instance segmentation masks. Pre-trained instance segmentation models can be obtained from the model zoo of [mmdetection](https://github.com/open-mmlab/mmdetection). Our paper uses 
Hybrid Task Cascade ([download](https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth)).

```
cd see/detector2d && bash scripts/htc/${DATASET_MASK}.sh
```
Once we have the masks, we can use them to isolate the points before estimating the pose and completing them. For this, we run the same command as the above, but with the `VCN-VC` configuration.
```
cd see/surface_completion && python sc_multiproc.py --cfg_file cfgs/${*}_DET_VCN-VC.yaml
```
## OpenPCDet Detector
We use [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) v0.5.0 and stick closely to the [usage](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) of the original codebase for the training and testing of the 3D detectors. We've reiterated the commands here for ease.

Make sure to double check the dataset paths in `cfgs/dataset_configs/sc_*_dataset.yaml` for our SEE-VCN approach, or `cfgs/dataset_configs/*_dataset.yaml` for the non SEE-VCN approach.

### Training
Edit the config file to ensure that we have the right infos for the source and target domain. Then train with the following:
```
python train.py --cfg_file ${CONFIG_FILE}
```

### Testing
To test with a pretrained model
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

To test all the saved checkpoints and draw the tensorboard curve, add `--eval_all`
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

## (Example) Testing on Baraja dataset with a pretrained model

### SEE-VCN from start to end
```
# Get instance segmentation masks
cd /SEE-VCN/see/detector2d && bash scripts/htc/baraja_masks.sh

# Complete surfaces of objects
cd /SEE-VCN/see/surface_completion && python sc_multiproc.py --cfg_file cfgs/BAR_DET_VCN-VC.yaml

# Test with pretrained detector (SECOND-IoU)
cd /SEE-VCN/detector3d/tools && python test.py --cfg_file cfgs/source-waymo/second_iou.yaml \
--batch_size 4 --ckpt /SEE-VCN/model_zoo/waymo_secondiou_see_vcn.pth
```

