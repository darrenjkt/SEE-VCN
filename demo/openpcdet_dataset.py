import glob 
from pcdet.datasets import DatasetTemplate
from pathlib import Path
import open3d as o3d
import numpy as np
import yaml
from easydict import EasyDict
import torch
from datasets.shared_utils import gtbox_to_corners

def opd_to_o3dbox(pred_dicts, shift_coor, score_thresh=0.1):
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy() 
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()    
    pred_boxes = pred_boxes[pred_scores > score_thresh]
    
    o3d_boxes = []
    for pred_box in pred_boxes:
        box_corners, r_mat = gtbox_to_corners(pred_box)
        boxpts = o3d.utility.Vector3dVector(box_corners)
        o3dbox = o3d.geometry.OrientedBoundingBox().create_from_points(boxpts)
        o3dbox.color = np.array([1,0,0])
        o3dbox.center = pred_box[0:3] - np.array(shift_coor)
        o3dbox.R = r_mat
        o3d_boxes.append(o3dbox)
    return o3d_boxes

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        base_pth = Path('/SEE-VCN/detector3d/tools/') / new_config['_BASE_CONFIG_']
        with open(str(base_pth), 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

class OPD_Demo(DatasetTemplate):
    def __init__(self, 
                dataset_cfg, 
                class_names, 
                training=True, 
                root_path=None, 
                logger=None, 
                shift_coor=None,
                ext='.pcd'):
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
        self.shift_coor = shift_coor

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = np.asarray(o3d.io.read_point_cloud(self.sample_file_list[index]).points)        
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        if self.shift_coor is not None:
            print('shifting coordinate frame by ', self.shift_coor)
            points[:, 0:3] += np.array(self.shift_coor, dtype=np.float32)
        
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict