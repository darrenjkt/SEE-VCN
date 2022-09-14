# Training and Testing

1. Complete the point clouds. 

For training data (source domain), please use the `VCN-CN` configuration to complete the objects. VCN-CN uses the source labels to canonicalize the objects before completing them. 
```
cd see/surface_completion && python sc_multiproc.py --cfg_file cfgs/${*}_GT_VCN-CN.yaml
```

For test data (target domain), we need to first get the image instance segmentation masks.
```
cd see/detector2d && bash scripts/htc/${DATASET_MASK}.sh
```
Once we have the masks, we can use them to isolate the points before estimating the pose and completing them. For this, we run the same command as the above, but with the `VCN-VC` configuration.
```
cd see/surface_completion && python sc_multiproc.py --cfg_file cfgs/${*}_DET_VCN-VC.yaml
```

2. Train the 3d detector. Edit the cfg file to ensure that we have the right infos for the source and target domain. Then train with the following:
```
python train.py --cfg_file ${CONFIG_FILE}
```

3. Testing

To test with a pretrained model
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

To test all the saved checkpoints and draw the tensorboard curve, add `--eval_all`
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```
