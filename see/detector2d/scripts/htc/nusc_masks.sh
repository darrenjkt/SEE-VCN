#!/bin/bash

declare -a cameras=('CAM_FRONT' 'CAM_FRONT_RIGHT' 'CAM_FRONT_LEFT' 'CAM_BACK_RIGHT' 'CAM_BACK' 'CAM_BACK_LEFT')
for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-VCN/see/detector2d/generate_masks.py \
	--config "/SEE-VCN/see/detector2d/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-VCN/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-VCN/data/nuscenes/v1.0-trainval" \
	--output_json "/SEE-VCN/data/nuscenes/v1.0-trainval/masks/htc_val/$cam.json" \
	--score_thresh 0.5 \
	--instance_mask \
	--custom_fpaths "/SEE-VCN/data/nuscenes/v1.0-trainval/ImageSets/val_image_paths/$cam.txt"

done