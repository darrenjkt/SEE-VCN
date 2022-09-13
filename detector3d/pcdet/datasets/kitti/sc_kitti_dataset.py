import copy
import pickle

import numpy as np
from skimage import io
import torch
import open3d as o3d
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..kitti.kitti_dataset import KittiDataset


class SCKittiDataset(KittiDataset):
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
        info = self.kitti_infos[info_idx]
        completed_lidar_file = self.root_split_path / info['completed_lidar_path']
        assert completed_lidar_file.exists(), f"No file at: {completed_lidar_file}"
        pcd = o3d.io.read_point_cloud(str(completed_lidar_file))
        return np.asarray(pcd.points, dtype=np.float32)

    def get_infos_from_idx(self, index):

        for i, dic in enumerate(self.kitti_infos):        
            if dic['point_cloud']['lidar_idx'] == index:
                return i
        return -1

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_completed_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict