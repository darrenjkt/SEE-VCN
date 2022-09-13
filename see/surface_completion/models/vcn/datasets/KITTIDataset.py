import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import json
from .build import DATASETS
from pathlib import Path
import open3d as o3d
import pickle
from utils.misc import *
from utils.logger import *

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py
@DATASETS.register_module()
class KITTI(data.Dataset):
    def __init__(self, config):
        self.data_dir = Path(config.DATA_PATH)
        self.subset = config.subset
        assert self.subset in ['val', 'test'], f"{self.subset} is invalid. Please specify val or test."

        self.partial_points_path = str(self.data_dir / self.subset / 'partial' / '%s.pcd')
        self.label_path = str(self.data_dir / self.subset / 'label' / '%s.pkl')        
        self.files_path = self.data_dir / self.subset / 'file_list.txt'        
        self.file_list = self._get_file_list(config)
        self.fixed_input = config.fixed_input
        self.transforms = self._get_transforms() if self.fixed_input else None

    def _get_file_list(self, config):
        """Prepare file list for the dataset"""
        file_list = []
        with open(self.files_path) as f:
            files = [line.split('\n')[0] for line in f.readlines()]
        
        for fname in files:                    
            file_list.append({
                'model_id': fname,
                'partial_path': self.partial_points_path % (fname),
                'label_path': self.label_path % (fname)
            })

        print_log('Complete collecting files. Total files: %d' % (len(file_list)), logger='KITTI_DATASET')
        return file_list

    def _get_transforms(self):
        transform_list = [
            {'callback': 'ResamplePoints',
             'parameters': {'n_points':1024}, # prev was 2048, not sure if better or worse
             'objects': ['partial']}
          ]
        return data_transforms.Compose(transform_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        # Randomly choose a view from the model renderings
        data['partial'] = np.asarray(o3d.io.read_point_cloud(sample['partial_path']).points)
        data['complete'] = None        
        with open(sample['label_path'], 'rb') as f:
            data['label'] = pickle.load(f)
            data['label']['num_pts'] = data['partial'].shape[0]    
        
        if self.transforms is not None:
            data = self.transforms(data)

        # Return taxonomy_id = 02958343 for car category
        return '02958343', sample['model_id'], (data['partial'], data['complete'], data['label'])

    def collate_variable_input(self, batch_list):
        
        model_ids, ids, bbox_pts, gt_boxes, complete, partial, num_pts, dataset_name = [],[],[],[],[],[],[],[]
        for idx, cur_sample in enumerate(batch_list):
            _, model_id, data = cur_sample
            label = data[2]
            bbox_pts.append(label['bbox_pts'])
            gt_boxes.append(label['gtbox'])
            dataset_name.append(label['dataset'])
            partial.append(data[0])
            ids.append(label['pc_id'])
            num_pts.append(label['num_pts'])
            model_ids.append(model_id)

        partial_pcds = torch.from_numpy(np.concatenate(partial, axis=0)).unsqueeze(0).to(torch.float32)
        complete_pcds = torch.zeros([len(ids),1,3], dtype=torch.float32)
        label = {'bbox_pts': torch.from_numpy(np.stack([bbox_pts], axis=0).squeeze().astype(np.float32)),
               'gt_boxes': torch.from_numpy(np.stack([gt_boxes], axis=0).squeeze(0).astype(np.float32)),
               'ids': ids,
               'num_pts': num_pts,
               'dataset': dataset_name}
        return tuple(['02958343' for i in range(len(batch_list))]), tuple(model_ids), (partial_pcds, complete_pcds, label)

    def collate_fixed_input(self, batch_list):
        
        model_ids, ids, bbox_pts, gt_boxes, complete, partial, num_pts, dataset_name = [],[],[],[],[],[],[],[]
        for idx, cur_sample in enumerate(batch_list):
            _, model_id, data = cur_sample
            label = data[2]
            bbox_pts.append(label['bbox_pts'])
            gt_boxes.append(label['gtbox'])
            dataset_name.append(label['dataset'])
            partial.append(data[0])
            ids.append(label['pc_id'])
            num_pts.append(label['num_pts'])
            model_ids.append(model_id)
        
        partial_pcds = torch.from_numpy(np.stack(partial, axis=0)).to(torch.float32)
        complete_pcds = torch.zeros([len(ids),1,3], dtype=torch.float32)
        label = {'bbox_pts': torch.from_numpy(np.stack([bbox_pts], axis=0).squeeze().astype(np.float32)),
               'gt_boxes': torch.from_numpy(np.stack([gt_boxes], axis=0).squeeze(0).astype(np.float32)),
               'ids': ids,
               'num_pts': np.array(num_pts),
               'dataset': dataset_name}       
        return tuple(['02958343' for i in range(len(batch_list))]), tuple(model_ids), (partial_pcds, complete_pcds, label)
