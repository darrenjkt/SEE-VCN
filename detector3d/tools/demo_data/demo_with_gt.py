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
    parser.add_argument('--cfg_file', type=str, default='cfgs/source-nuscenes/second_iou.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.pcd', help='specify the extension of your point cloud data file')
    parser.add_argument('--min_gt_pts', type=int, default=50, help='show gt for objects with minimum of N points')
    parser.add_argument('--play', action='store_true', help='if true, we play frame by frame like a video')

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
        if args.play:
            vis = open3d.visualization.Visualizer()
            vis.create_window()    

        for idx, data_dict in enumerate(test_loader):
            if idx > 500:
                break

            logger.info(f'Visualized sample index: \t{idx + 1}')            

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)       
            gt_mask = data_dict['num_points_in_gt'][0] > args.min_gt_pts

            if args.play:
                geom = V.get_geometries(
                        points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'], gt_boxes=data_dict['gt_boxes'][0][gt_mask],
                        ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], draw_origin=False)

                vis.clear_geometries()
                for g in geom:                
                    vis.add_geometry(g)
                    
                ctr = vis.get_view_control()    
                vis.get_render_option().point_size = 2.0
                ctr.set_front([ -0.83801016835646658, 0.020189964528675192, 0.54528095791389741 ])
                ctr.set_lookat([ 20.930189078661069, -0.13003185905090625, -4.5181982607632714 ])
                ctr.set_up([ 0.54532040124707715, -0.0039821110794315264, 0.83821823099660586 ])
                ctr.set_zoom(0.21)
                vis.update_renderer()         
                vis.poll_events()
                Path('demo_data/save_frames').mkdir(parents=True, exist_ok=True)
                vis.capture_screen_image(f'demo_data/save_frames/frame-{idx}.jpg')
            else:
                
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0][gt_mask], ref_boxes=pred_dicts[0]['pred_boxes'], 
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], draw_origin=False, idx=idx)
                
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()