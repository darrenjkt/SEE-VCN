#!/bin/bash

# cameras=("image_2" "image_3") if you want to get masks for both cameras
declare -a cameras=("front")

for cam in "${cameras[@]}"
do 
	echo "Getting masks for $cam"

	python /SEE-VCN/see/detector2d/generate_masks.py \
	--config "/SEE-VCN/see/detector2d/mmdetection/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py" \
	--checkpoint "/SEE-VCN/model_zoo/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth" \
	--data_dir "/SEE-VCN/data/custom_data/baraja_subsets/test100_pc0_pc1/test/image/$cam" \
	--output_json "/SEE-VCN/data/custom_data/baraja_subsets/test100_pc0_pc1/test/masks/htc_x101/$cam.json" \
	--score_thresh 0.3 \
	--instance_mask

done