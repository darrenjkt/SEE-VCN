#!/bin/bash

declare -a cameras=('FRONT' 'FRONT_LEFT' 'FRONT_RIGHT' 'SIDE_LEFT' 'SIDE_RIGHT')
for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-VCN/see/generate_masks.py \
	--config "/SEE-VCN/see/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-VCN/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-VCN/data/waymo/custom_1000/image_lidar_projections/image/$cam" \
	--output_json "/SEE-VCN/data/waymo/custom_1000/image_lidar_projections/masks/htc/$cam.json" \
	--instance_mask

done