import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

import open3d as o3d
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..nuscenes.nuscenes_dataset import NuScenesDataset


class SCNuScenesDataset(NuScenesDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def get_completed_lidar(self, index):
        info = self.infos[index]
        completed_lidar_file = self.root_path / info['completed_lidar_path']

        assert completed_lidar_file.exists(), f"No file at: {completed_lidar_file}"
        file_extension = str(completed_lidar_file).split('.')[-1]
        pcd = o3d.io.read_point_cloud(str(completed_lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_completed_lidar(index)
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            
            if self.dataset_cfg.get('MIN_POINTS_OF_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.MIN_POINTS_OF_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = input_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            input_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in input_dict:
            input_dict['gt_boxes'] = input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict