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
        data_file_list = glob.glob(str(root_path  / 'velodyne_points' / 'vcn_data' / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.calib = self.load_calibration()
        self.image_shape = np.array([375, 1242, 3])

    def load_calibration(self):
        v2c_path = self.root_path / 'calib'/ 'calib_velo_to_cam.txt'
        cam2cam_path = self.root_path / 'calib'/ 'calib_cam_to_cam.txt'
        assert v2c_path.exists(), f'No calib file found at {v2c_path}'
        assert cam2cam_path.exists(), f'No calib file found at {cam2cam_path}'
        
        calib = {}
        with open(v2c_path,"r") as f:
            file = f.readlines()                
            for line in file:
                (key, val) = line.split(':', 1)
                if key == 'R':
                    R = np.fromstring(val, sep=' ')
                    calib['v2c_R'] = R.reshape(3, 3)
                if key == 'T':
                    T = np.fromstring(val, sep=' ')
                    calib['v2c_T'] = T.reshape(3, 1)
        
        with open(cam2cam_path,"r") as f:
            file = f.readlines()      
            for line in file:
                (key, val) = line.split(':', 1)
                if key == ('P_rect_02'): # Use image_02 (colour,rectified) by default
                    P_ = np.fromstring(val, sep=' ')
                    P_ = P_.reshape(3, 4)
                    # erase 4th column ([0,0,0])
                    calib['P'] = P_[:3, :3]
                    
        return calib

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)[:,:3]
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = np.asarray(open3d.io.read_point_cloud(self.sample_file_list[index]).points)
        else:
            raise NotImplementedError

        
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            
        if self.dataset_cfg.get('FOV_POINTS_ONLY', None):
            fov_flag = self.get_fov_flag(points)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def get_fov_flag(self, points, min_dist=1.0):
        pts_2d = self.project_velo_to_imageuv(points)
        IMG_H, IMG_W, _ = self.image_shape
        fov_inds = (pts_2d[:,0]<IMG_W) & (pts_2d[:,0]>=0) & \
            (pts_2d[:,1]<IMG_H) & (pts_2d[:,1]>=0)
        fov_inds = fov_inds & (points[:,0]>min_dist)
        return fov_inds

    def project_velo_to_imageuv(self, points):
        RT_ = np.concatenate((self.calib['v2c_R'], self.calib['v2c_T']), axis=1)
        hom_pts = np.hstack((points, np.ones((points.shape[0],1)))) # (N,4)
        pts_camframe = hom_pts @ RT_.T
        pts_uv = pts_camframe @ self.calib['P'].T
        depth = pts_uv[:,2]
        pts_uv[:,0] /= depth
        pts_uv[:,1] /= depth
        return pts_uv

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/source-nuscenes/second_iou.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/SEE-VCN/data/kitti/raw_data/2011_09_26_drive_0093_sync',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--play', action='store_true', help='if true, we play frame by frame like a video')

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
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.play:
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

            if args.play:
                geom = V.get_geometries(
                        points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'][mask],
                        ref_scores=pred_dicts[0]['pred_scores'][mask], ref_labels=pred_dicts[0]['pred_labels'][mask], draw_origin=False)

                vis.clear_geometries()
                for g in geom:                
                    vis.add_geometry(g)
                    
                ctr = vis.get_view_control()    
                vis.get_render_option().point_size = 2.0
                ctr.set_front([ -0.80399866368441153, 0.0053176952054748141, 0.59460732497286173 ])
                ctr.set_lookat([ 21.374403772027588, -0.41756625208030207, -3.8248644230718032 ])
                ctr.set_up([ 0.59389575957710095, -0.042533534631139798, 0.80341690621253425 ])
                ctr.set_zoom(0.2)
                vis.update_renderer()         
                vis.poll_events()
                Path('demo_data/save_frames').mkdir(parents=True, exist_ok=True)
                vis.capture_screen_image(f'demo_data/save_frames/frame-{idx}.jpg')
            else:
                V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'][mask],
                ref_scores=pred_dicts[0]['pred_scores'][mask], ref_labels=pred_dicts[0]['pred_labels'][mask], draw_origin=False
                )


            if not OPEN3D_FLAG:
                mlab.show(stop=True)
        
    logger.info('Demo done.')


if __name__ == '__main__':
    main()