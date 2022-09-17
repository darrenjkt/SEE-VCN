import argparse
import glob
from pathlib import Path
import time
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
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.pcd'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = np.asarray(open3d.io.read_point_cloud(self.sample_file_list[index]).points)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        print(self.dataset_cfg)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            print(f'Adding shift coor of {self.dataset_cfg.SHIFT_COOR}')
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/source-waymo/second_iou.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/SEE-VCN/data/kitti/raw_data/2011_09_26_drive_0093_sync/velodyne_points/vcn_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.pcd', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG_TAR, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    print(demo_dataset.point_cloud_range)
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        

        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            car_mask = (pred_dicts[0]['pred_labels'] == 1)
            scores_mask = (pred_dicts[0]['pred_scores'] > 0.5)
            mask = car_mask & scores_mask

            
            geom = V.get_geometries(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'][mask],
                    ref_scores=pred_dicts[0]['pred_scores'][mask], ref_labels=pred_dicts[0]['pred_labels'][mask])
            print(f'adding {len(geom)} geoms')

            vis.clear_geometries()
            for g in geom:                
                vis.add_geometry(g)
                
            ctr = vis.get_view_control()    
            vis.get_render_option().point_size = 2.0
            ctr.set_front([ -0.82471032579302606, 0.00018059423526749826, 0.56555534292948129 ])
            ctr.set_lookat([ 19.844308125762407, -0.13879944104704953, -6.4286119410902804 ])
            ctr.set_up([ 0.56547045688150521, 0.017591491015727621, 0.82458092497829805 ])
            ctr.set_zoom(0.259)

            vis.update_renderer()         
            vis.poll_events()
               
            Path('demo_data/save_frames').mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(f'demo_data/save_frames/frame-{idx}.jpg')

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
        
    logger.info('Demo done.')


if __name__ == '__main__':
    main()