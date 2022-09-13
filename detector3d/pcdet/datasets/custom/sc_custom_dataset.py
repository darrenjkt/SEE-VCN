import copy
import pickle

import numpy as np
from skimage import io
import open3d as o3d
import json

from . import custom_dataset_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..custom.custom_dataset import CustomDataset

class SCCustomDataset(CustomDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def get_completed_lidar(self, idx):
        """
        Loads completed lidar for a sample
        Args:
            idx: int, Sample index
        Returns:
            lidar: (N, 3), 2D np.float32 array
        """
        info_idx = self.get_infos_from_idx(f'{int(idx):06d}')
        info = self.infos[info_idx]
        completed_lidar_file = self.root_split_path / info['completed_lidar_path']
        assert completed_lidar_file.exists(), f"No file at: {completed_lidar_file}"
        
        pcd = o3d.io.read_point_cloud(str(completed_lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)

    def __getitem__(self, index):

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = self.get_image_shape(sample_idx)
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            gt_names = annos['name']
            num_points_in_gt = annos['num_points_in_gt']
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

        if "points" in get_item_list:
            points = self.get_completed_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                fov_flag = custom_dataset_utils.get_fov_flag(points, img_shape, calib)
                points = points[fov_flag]

            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
