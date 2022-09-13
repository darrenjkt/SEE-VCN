# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import open3d
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.iou3d_nms import iou3d_nms_utils
from utils.losses import geodesic_distance, angle_between_vectors, points_outside_hull, get_oob_error
from utils.transform import rot_from_heading, rotm_to_heading
from sklearn.decomposition import PCA
from utils import misc
from utils.sampling import get_partial_mesh_batch
import numpy as np

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': False,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL1_PARTIAL',
        'enabled': False,
        'eval_func': 'cls._get_chamfer_distancel1_partial',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2_PARTIAL',
        'enabled': False,
        'eval_func': 'cls._get_chamfer_distancel2_partial',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'OUT_OF_BOX',
        'enabled': True,
        'eval_func': 'cls._get_oob_error',
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'IOU_3D_Error',
        'enabled': False,
        'eval_func': 'cls._get_iou_3d_error',
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'IOU_3D',
        'enabled': True,
        'eval_func': 'cls._get_box_iou3d',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'IOU_BEV',
        'enabled': False,
        'eval_func': 'cls._get_box_iou_bev',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Rotation_Error',
        'enabled': True,
        'eval_func': 'cls._get_rotation_error',
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'Translation_Error',
        'enabled': True,
        'eval_func': 'cls._get_translation_error',
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'Axis_Alignment',
        'enabled': False,
        'eval_func': 'cls._get_axis_alignment',
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'Coherence',
        'enabled': False,
        'eval_func': 'cls._get_coherence',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'Cosine_Similarity',
        'enabled': False,
        'eval_func': 'cls._get_cosine_similarity',
        'is_greater_better': True,
        'init_value': 0
    }]
    levels = {}
    levels['L1'] = {'min':201, 'max': 1000000} # arbitrary max
    levels['L2'] = {'min':81, 'max':200}
    levels['L3'] = {'min':31, 'max':80}
    levels['L4'] = {'min':5, 'max':30}

    @classmethod
    def get(cls, pred, gt, eval_by_num_pts=True):
        _items = cls.items()
        _values = []
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            if eval_by_num_pts:
                if 'min_pts' in item:
                    _values.append(eval_func(pred, gt, min_pts=item['min_pts'], max_pts=item['max_pts']))
                else:
                    _values.append(eval_func(pred, gt, min_pts=None, max_pts=None))                    
            else:
                _values.append(eval_func(pred, gt))

        return _values

    @classmethod
    def items(cls, eval_by_num_pts=True):
        if eval_by_num_pts:
            items = []
            for it in cls.ITEMS:
                if it['enabled']:
                    items.append(it)
                    base_name = it['name']                    
                    for level in cls.levels.keys():
                        it_copy = {key: value for key, value in it.items()}                        
                        it_copy['name'] = base_name + f'_{level}'
                        it_copy['min_pts'] = cls.levels[level]['min']
                        it_copy['max_pts'] = cls.levels[level]['max']
                        items.append(it_copy)
            return items
        else:
            return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls, eval_by_num_pts=True):
        # if eval_by_num_pts:
        #     _items = cls.items()
        #     metric_names = []
        #     for i in _items:
        #         metric_names.append(i['name'])
        #         for level in cls.levels.keys():
        #             name = i['name'] + f'_{level}'
        #             metric_names.append(name)                
        #     return metric_names
        # else:
        _items = cls.items()

        return [i['name'] for i in _items]


    @classmethod
    def _get_chamfer_distancel1(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        if in_dict['complete'].shape[1] == 1:
            return -1

        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)

            pred = ret_dict['coarse'][mask]
            gt = in_dict['complete'][mask]   
            if len(pred) == 0:
                return -1
        else:
            pred = ret_dict['coarse']
            gt = in_dict['complete']
        
        try:
            chamfer_distance = cls.ITEMS[1]['eval_object']
            return chamfer_distance(pred, gt).item() * 1000
        except:
            return -1

    @classmethod
    def _get_chamfer_distancel2(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        if in_dict['complete'].shape[1] == 1:
            return -1

        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred = ret_dict['coarse'][mask]
            gt = in_dict['complete'][mask]   
            if len(pred) == 0:
                return -1
        else:
            pred = ret_dict['coarse']
            gt = in_dict['complete']
        
        try:
            chamfer_distance = cls.ITEMS[2]['eval_object']
            return chamfer_distance(pred, gt).item() * 1000
        except:
            return -1

    @classmethod
    def _get_chamfer_distancel1_partial(cls, ret_dict, in_dict, sel_k=30, min_pts=None, max_pts=None):
        if in_dict['complete'].shape[1] == 1:
            return -1

        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred = ret_dict['coarse'][mask]
            gt = in_dict['complete'][mask]   
            inpc = in_dict['input'][mask]   
            
            if len(pred) == 0:
                return -1
        else:
            pred = ret_dict['coarse']
            gt = in_dict['complete']
            inpc = in_dict['input']            

        try:
            ds_complete = misc.fps(gt, pred.shape[1])
            pred_surface = get_partial_mesh_batch( inpc, pred, k=sel_k)
            gt_surface = get_partial_mesh_batch( inpc, ds_complete, k=sel_k)

            chamfer_distance = cls.ITEMS[1]['eval_object']
            return chamfer_distance(pred_surface, gt_surface).item() * 1000
        except:
            return -1

    @classmethod
    def _get_chamfer_distancel2_partial(cls, ret_dict, in_dict, sel_k=30, min_pts=None, max_pts=None):
        if in_dict['complete'].shape[1] == 1:
            return -1

        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred = ret_dict['coarse'][mask]
            gt = in_dict['complete'][mask]   
            inpc = in_dict['input'][mask]   
            
            if len(pred) == 0:
                return -1
        else:
            pred = ret_dict['coarse']
            gt = in_dict['complete']
            inpc = in_dict['input']  
        
        try:
            ds_complete = misc.fps(gt, pred.shape[1])
            pred_surface = get_partial_mesh_batch( inpc, pred, k=sel_k)
            gt_surface = get_partial_mesh_batch( inpc, ds_complete, k=sel_k)

            chamfer_distance = cls.ITEMS[2]['eval_object']
            return chamfer_distance(pred_surface, gt_surface).item() * 1000
        except:
            return -1

    @classmethod
    def _get_oob_error(cls, ret_dict, in_dict, ds_pts=200, min_pts=None, max_pts=None):
        
        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred = ret_dict['coarse'][mask]
            gtboxes = in_dict['gt_boxes'][mask]   
            
            if len(pred) == 0:
                return -1
        else:
            pred = ret_dict['coarse']
            gtboxes = in_dict['gt_boxes']
        
        return get_oob_error(pred, gtboxes).mean().item()

        # try:
        #     return points_outside_hull(pred, gt)
        # except:
        #     return -1

    @classmethod
    def _get_iou_3d_error(cls, ret_dict, in_dict, min_pts=None, max_pts=None):        
        """
        Get iou3d mse error for predicted iou vs actual iou
        """
        if 'reg_iou_3d' in ret_dict.keys():
            if (min_pts is not None) and (max_pts is not None):
                mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
                pred_bbox = ret_dict['pred_box'][mask]
                gt_bbox = in_dict['gt_boxes'][mask]            
                if len(pred_bbox) == 0:
                    return -1
            else:
                pred_bbox = ret_dict['pred_box']
                gt_bbox = in_dict['gt_boxes']
                        
            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_bbox, gt_bbox).diag() # B
            pred_iou = ret_dict['reg_iou_3d'] # B

            return torch.abs(iou3d - pred_iou).mean().item()
        else:
            return -1

    @classmethod
    def _get_translation_error(cls, ret_dict, in_dict, min_pts=None, max_pts=None):        
        """
        Get rotation error using geodesic distance.

        Range of this geodesic error is from: [0, 3.14] 
        """
        if ('reg_centre' in ret_dict.keys()) or ('reg_centre_0' in ret_dict.keys()):
            reg_centre = ret_dict['reg_centre'] if 'reg_centre' in ret_dict.keys() else ret_dict['reg_centre_0']
            if (min_pts is not None) and (max_pts is not None):
                mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
                pred_centre = reg_centre[mask]
                gt_centre = in_dict['gt_boxes'][:,:3][mask]         
                if len(pred_centre) == 0:
                    return -1
            else:
                pred_centre = reg_centre
                gt_centre = in_dict['gt_boxes'][:,:3]

            return torch.abs(pred_centre - gt_centre).mean().item()
        else:
            return -1

    @classmethod
    def _get_rotation_error(cls, ret_dict, in_dict, min_pts=None, max_pts=None):        
        """
        Get rotation error using geodesic distance.

        Range of this geodesic error is from: [0, 3.14] 
        """
        if ('reg_rot' in ret_dict.keys()) or ('reg_rot_0' in ret_dict.keys()):
            reg_rot = ret_dict['reg_rot'] if 'reg_rot' in ret_dict.keys() else ret_dict['reg_rot_0']
            if (min_pts is not None) and (max_pts is not None):
                mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
                pred_rmat = reg_rot[mask]
                gt_headings = in_dict['gt_boxes'][:,-1][mask]         
                if len(pred_rmat) == 0:
                    return -1
            else:
                pred_rmat = reg_rot
                gt_headings = in_dict['gt_boxes'][:,-1]

            pred_heading = rotm_to_heading(pred_rmat)
            return torch.abs(pred_heading - gt_headings).median().item()
        else:
            return -1

    @classmethod
    def _get_axis_alignment(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        """
        Check if generated car is parallel with gt bounding box.
        Use PCA to determine major axis of car. We don't check
        heading here since some methods don't predict it.
        """
        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)        
            heading_gt = in_dict['gt_boxes'][:,-1][mask]            
            pc_bev = ret_dict['coarse'][:,:,:2][mask].cpu().numpy()
            if len(pc_bev) == 0:
                return -1
        else:
            heading_gt = in_dict['gt_boxes'][:,-1]           
            pc_bev = ret_dict['coarse'][:,:,:2].cpu().numpy()

        # PCA to predict major axis of coarse car            
        gt_vec = torch.stack([ torch.cos(heading_gt), torch.sin(heading_gt)], dim=1) # B 2
        pca_init = [PCA(n_components=1) for i in range(pc_bev.shape[0])]
        pcas = [pca.fit(pc_bev[idx]) for idx, pca in enumerate(pca_init)]
        major_axis = torch.stack([torch.from_numpy(pca.components_.squeeze(0)) for pca in pcas]).cuda() # B 2

        angle1 = angle_between_vectors(gt_vec, major_axis)
        angle2 = angle_between_vectors(-gt_vec, major_axis)
        angle_to_vehicle_axis = torch.min(angle1, angle2)

        return angle_to_vehicle_axis.mean().item()

    @classmethod
    def _get_coherence(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        """
        Coherence describes how much noise there is in the shape generation.
        Incoherent (i.e. noisy) points lead to low explained variance by PCA.

        We impose an upper bound of 1 as some model of cars have much higher 
        explained variance.
        """
        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)        
            pc_bev = ret_dict['coarse'][:,:,:2][mask].cpu().numpy()
            if len(pc_bev) == 0:
                return -1
        else:
            pc_bev = ret_dict['coarse'][:,:,:2].cpu().numpy()

        # PCA to predict major axis of coarse car
        pca_init = [PCA(n_components=1) for i in range(pc_bev.shape[0])]
        pcas = [pca.fit(pc_bev[idx]) for idx, pca in enumerate(pca_init)]
        explained_var = torch.tensor([pca.explained_variance_.item() for pca in pcas]).mean()

        return torch.clamp(explained_var, max=1.0).item()

    @classmethod
    def _get_box_iou3d(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred_bbox = ret_dict['pred_box'][mask]
            gt_bbox = in_dict['gt_boxes'][mask]            
            if len(pred_bbox) == 0:
                return -1
        else:
            pred_bbox = ret_dict['pred_box']
            gt_bbox = in_dict['gt_boxes']
                    
        iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_bbox, gt_bbox)
        return iou3d.diag().mean().item()

    @classmethod
    def _get_box_iou_bev(cls, ret_dict, in_dict, min_pts=None, max_pts=None):
        if (min_pts is not None) and (max_pts is not None):
            mask = (in_dict['num_pts'] >= min_pts) & (in_dict['num_pts'] <= max_pts)
            pred_bbox = ret_dict['pred_box'][mask]
            gt_bbox = in_dict['gt_boxes'][mask]
            if len(pred_bbox) == 0:
                return -1
        else:
            pred_bbox = ret_dict['pred_box']
            gt_bbox = in_dict['gt_boxes']
                    
        iou_bev = iou3d_nms_utils.boxes_iou_bev(pred_bbox, gt_bbox)
        return iou_bev.diag().mean().item()

    @classmethod
    def _get_cosine_similarity(cls, pred_bbox, gt_bbox):
        pred_heading = pred_bbox[:,-1]
        gt_heading = gt_bbox[:,-1]
        heading_err = abs(gt_heading - pred_heading)
        # print('----')
        # print(f'pred_heading = {pred_heading}')
        # print(f'gt_heading = {gt_heading}')
        # print(f'cos sim = {(1.0 + torch.cos(heading_err))/2.0}')
        
        return (1.0 + torch.cos(heading_err))/2.0

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):

        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud    

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
