#!/bin/bash

declare -a cameras=("front")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for camera: $cam"

	python /SEE-VCN/see/detector2d/generate_masks.py \
	--config "/SEE-VCN/see/detector2d/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-VCN/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-VCN/see/ouster/image/$cam" \
	--output_json "/SEE-VCN/see/ouster/masks/htc/$cam.json" \
	--instance_mask

done