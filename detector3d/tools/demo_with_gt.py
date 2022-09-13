import _init_path
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader


# # python demo.py --ckpt /SEEv2/detector3d/output/source-waymo/second_iou/100ep/ckpt/checkpoint_epoch_47.pth
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/source-waymo/second_iou.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/SEEv2/detector3d/tools/demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.pcd', help='specify the extension of your point cloud data file')
    parser.add_argument('--min_gt_pts', type=int, default=50, help='show gt for objects with minimum of N points')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG_TAR,
            class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
            batch_size=1, logger=logger, training=False, dist=False
        )
    logger.info(f'Total number of samples: \t{len(test_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            # if idx == 0:

            logger.info(f'Visualized sample index: \t{idx + 1}')

            # print(data_dict['gt_boxes'][0, :, :7].shape )
            # print(data_dict['points'][:, :3].shape)

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)            

            # # filter gt_boxes without points
            # num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
            #     data_dict['points'][:, :3].cpu(),
            #     data_dict['gt_boxes'][0, :, :7].cpu()).numpy().sum(axis=1)

            # mask = (num_points_in_gt >= args.min_gt_pts)
            # print(roiaware_pool3d_utils.points_in_boxes_cpu(
            #     data_dict['points'][:, :3].cpu(),
            #     data_dict['gt_boxes'][0, :, :7].cpu()).numpy())
            # print(mask)
            # print(data_dict['gt_boxes'][0].shape)
            # data_dict['gt_boxes'] = data_dict['gt_boxes'][0][mask]

            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0], ref_boxes=pred_dicts[0]['pred_boxes'], 
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()