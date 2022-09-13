import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from ..datasets import data_transforms
import random
import os
import json
from .build import DATASETS
from ..utils.logger import *
import open3d as o3d
import pickle
from ..utils.misc import *
from pathlib import Path
import time

@DATASETS.register_module()
class VC(data.Dataset):
    def __init__(self, config):
        self.data_dir = Path(config.DATA_PATH)
        self.subset = config.subset
        self.fixed_input = config.fixed_input
        self.num_inputs = config.get('num_inputs', 1)
        assert self.subset in ['train', 'val', 'test'], f"{self.subset} is invalid. Please specify train, val or test."

        self.partial_points_path = str(self.data_dir / self.subset/ 'partial' / '%s' / '%03d.pcd')
        self.complete_points_path = str(self.data_dir / self.subset / 'surface' / '%s' / '%03d.pcd')        
        
        if (self.data_dir / self.subset / 'label.pkl').exists():
            self.label_path = self.data_dir / self.subset / 'label.pkl'
            with open(self.label_path, 'rb') as fp:
                self.labels = pickle.load(fp)
        else:
            self.label_path = str(self.data_dir / self.subset / 'label' / '%s' / '%03d.pkl')        
            self.labels = None

        self.model_id_path = self.data_dir / self.subset / 'model_ids.txt'        
        self.npoints = config.N_POINTS        
        self.nviews = config.USE_NVIEWS_PER_MODEL if self.subset == 'train' else 4

        with open(self.model_id_path) as f:
            self.model_ids = [line.split('\n')[0] for line in f.readlines()]
        
        self.file_list = self._get_file_list(config)
        self.transforms = self._get_transforms(config.TRANSFORMS)

    def _get_transforms(self, transforms):
        if self.subset == 'train':
            transform_list = transforms.train
        else:
            transform_list = transforms.test

        return data_transforms.Compose(transform_list)

    def _get_file_list(self, config):
        """Prepare file list for the dataset"""
        file_list = []

        for model_id in self.model_ids:
            if self.subset == 'train':
                view_indices = np.random.permutation(config.TOTAL_NVIEWS_PER_MODEL)
            else:
                view_indices = np.arange(config.TOTAL_NVIEWS_PER_MODEL)

            for i in view_indices[:self.nviews]:     
                data_input = {
                    'model_id': model_id,
                    'partial_path': self.partial_points_path % (model_id, i),
                    'complete_path': self.complete_points_path % (model_id, i),
                    'label_path': self.label_path % (model_id, i) if self.labels is None else (model_id, i)
                }           
                if self.num_inputs > 1:
                    for i in range(1, self.num_inputs):
                        rand_view = np.random.randint(0,40)
                        data_input[f'partial_path_{i}'] = self.partial_points_path % (model_id, rand_view)
                        data_input[f'complete_path_{i}'] = self.complete_points_path % (model_id, rand_view)
                        data_input[f'label_path_{i}'] = self.label_path % (model_id, rand_view)  if self.labels is None else (model_id, rand_view)
                file_list.append(data_input)

        print_log('Complete collecting files for %s. Total views: %d' % (self.subset, len(file_list)), logger='VC_DATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        t0 = time.time()
        # Randomly choose a view from the model renderings
        data['partial_0'] = np.asarray(o3d.io.read_point_cloud(sample['partial_path']).points)
        data['complete_0'] = np.asarray(o3d.io.read_point_cloud(sample['complete_path']).points)        
        t1 = time.time()

        if self.labels is None:
            with open(sample['label_path'], 'rb') as f:
                data['label_0'] = pickle.load(f)
                data['label_0']['num_pts'] = data['partial_0'].shape[0]
        else:
            mid = sample['label_path'][0]
            viewidx = sample['label_path'][1]
            data['label_0'] = self.labels[mid][viewidx]
            data['label_0']['num_pts'] = data['partial_0'].shape[0]

        t2 = time.time()

        # print(f'loaded one set - partial/complete in {t1-t0:0.3f}s, label in {t2-t1:0.3f}s')

        if self.num_inputs > 1:
            for i in range(1, self.num_inputs):
                data[f'partial_{i}'] = np.asarray(o3d.io.read_point_cloud(sample[f'partial_path_{i}']).points)
                data[f'complete_{i}'] = np.asarray(o3d.io.read_point_cloud(sample[f'complete_path_{i}']).points)

                
                if self.labels is None:
                    with open(sample[f'label_path_{i}'], 'rb') as f:
                        data[f'label_{i}'] = pickle.load(f)
                        data[f'label_{i}']['num_pts'] = data[f'partial_{i}'].shape[0]
                else:
                    mid = sample[f'label_path_{i}'][0]
                    viewidx = sample[f'label_path_{i}'][1]
                    data[f'label_{i}'] = self.labels[mid][viewidx]
                    data[f'label_{i}']['num_pts'] = data[f'partial_{i}'].shape[0]
                    
        t3 = time.time()
        partial_ins = [k for k in data.keys() if k.startswith('partial_')]
        complete_ins = [k for k in data.keys() if k.startswith('complete_')]
        label_ins = [k for k in data.keys() if k.startswith('label_')]
        area_ins = [k for k in data.keys() if k.startswith('area_')]

        if self.num_inputs == 1:
            data['partial'] = data['partial_0']
            data['complete'] = data['complete_0']
            data['label'] = data['label_0']

        if self.transforms is not None:
            data = self.transforms(data)

        t4 = time.time()
        
        if len(partial_ins) > 1:
            data['partial'] = [data[k] for k in partial_ins]
            data['complete'] = [data[k] for k in complete_ins]
            data['label'] = [data[k] for k in label_ins]

        # print("data['label'] = ", data['label'])
        # print(f'transforms in {t4-t3:0.3f}s, assignment in {time.time() - t4:0.3f}')
        return '02958343', sample['model_id'], (data['partial'], data['complete'], data['label'])

    def __len__(self):
        return len(self.file_list)

    def collate_fixed_input(self, batch_list):
        t0 = time.time()

        model_ids, ids, bbox_pts, gt_boxes, complete, partial, area, cn_area, num_pts = [],[],[],[],[],[],[],[],[]
        for idx, cur_sample in enumerate(batch_list):
            _, model_id, data = cur_sample
            label = data[2]
            if isinstance(label, list):
                bbox_pts.append([l['bbox_pts'] for l in label])
                gt_boxes.append([l['gtbox'] for l in label])
                ids.append([l['pc_id'] for l in label])
                num_pts.append([l['num_pts'] for l in label])
                # area.append([l['surface_area'] for l in label])
                # cn_area.append([l['canonical_area'] for l in label])
            else:
                bbox_pts.append(label['bbox_pts'])
                gt_boxes.append(label['gtbox'])
                ids.append(label['pc_id'])
                num_pts.append(label['num_pts'])
                # area.append(label['surface_area'])
                # cn_area.append(label['canonical_area'])
            
            model_ids.append(model_id)
            partial.append(data[0])
            complete.append(data[1])

        partials = torch.from_numpy(np.array(partial)).to(torch.float32)
        completes = torch.from_numpy(np.array(complete)).to(torch.float32)
        if isinstance(batch_list[0][2][0], list):
            partial_pcds, complete_pcds = [], []
            num_inputs = len(batch_list[0][2][0])
            for i in range(num_inputs):
                partial_pcds.append(partials[:,i,:,:])
                complete_pcds.append(completes[:,i,:,:])
        else:
            partial_pcds = partials
            complete_pcds = completes

        # partial_pcds = torch.from_numpy(np.stack(partial, axis=0)).to(torch.float32)
        # complete_pcds = torch.from_numpy(np.stack(complete, axis=0)).to(torch.float32)

        if isinstance(batch_list[0][2][2], list):
            num_inputs = len(batch_list[0][2][2]) # length of single getitem label list
            label = []
            for i in range(num_inputs):
                label_dict = {  'bbox_pts': torch.from_numpy(np.array([b[i] for b in bbox_pts]).astype(np.float32)),
                                'gt_boxes': torch.from_numpy(np.array([g[i] for g in gt_boxes]).astype(np.float32)),
                                'ids': [pcid[i] for pcid in ids],
                                'num_pts': np.array([npts[i] for npts in num_pts])}
                                # 'areas': np.array([a[i] for a in area]),
                                # 'cn_areas': np.array([a[i] for a in cn_area])}
                label.append(label_dict)

        else:
            label = {'bbox_pts': torch.from_numpy(np.stack([bbox_pts], axis=0).squeeze().astype(np.float32)),
                   'gt_boxes': torch.from_numpy(np.stack([gt_boxes], axis=0).squeeze(0).astype(np.float32)),
                   'ids': ids,
                   'num_pts': np.array(num_pts)}
                   # 'areas': np.array(area),
                   # 'cn_areas': np.array(cn_area)}

        # print(f'collated in {time.time() - t0:0.3f}s')
        return tuple(['02958343' for i in range(len(batch_list))]), tuple(model_ids), (partial_pcds, complete_pcds, label)


    def collate_variable_input(self, batch_list):
        
        model_ids, ids, bbox_pts, gt_boxes, complete, partial, num_pts = [],[],[],[],[],[],[]
        for idx, cur_sample in enumerate(batch_list):
            _, model_id, data = cur_sample
            label = data[2]
            bbox_pts.append(label['bbox_pts'])
            gt_boxes.append(label['gtbox'])            
            ids.append(label['pc_id'])
            num_pts.append(label['num_pts'])
            model_ids.append(model_id)
            
            if not isinstance(data[0], list):
                partial.append(data[0])
                complete.append(data[1])

        partial_pcds = torch.from_numpy(np.concatenate(partial, axis=0)).unsqueeze(0).to(torch.float32)
        complete_pcds = torch.from_numpy(np.stack(complete, axis=0)).to(torch.float32)
        label = {'bbox_pts': torch.from_numpy(np.stack([bbox_pts], axis=0).squeeze().astype(np.float32)),
               'gt_boxes': torch.from_numpy(np.stack([gt_boxes], axis=0).squeeze(0).astype(np.float32)),
               'ids': ids,
               'num_pts': np.array(num_pts)}
        return tuple(['02958343' for i in range(len(batch_list))]), tuple(model_ids), (partial_pcds, complete_pcds, label)