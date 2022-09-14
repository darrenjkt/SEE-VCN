import os
import glob
import time
import torch 
import numpy as np
from pathlib import Path
import open3d as o3d
from datasets.shared_utils import convert_to_o3dpcd, populate_gtboxes

from models.vcn.tools import builder
from models.vcn.utils.sampling import get_partial_mesh_batch, get_largest_cluster_batch
from models.vcn.datasets.data_transforms import ResamplePoints

class VCN:
    def __init__(self, cfg, gpu_id=0):
        self.cfg = cfg

        # DL Models     
        self.device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id) 
        self.model_init()       

    def model_init(self):
        """ 
        Initialize model
        """
        self.norm_with_gt = self.cfg.NORM_WITH_GT # Normalize pose with gt boxes orientation
        self.surface_sel_k = self.cfg.SEL_K_NEAREST
        self.cluster_eps = self.cfg.CLUSTER_EPS

        self.batch_size_limit = self.cfg.get('BATCH_SIZE_LIMIT', None)
        assert Path(self.cfg.CKPT_PATH).exists(), f"No ckpt found at {self.cfg.CKPT_PATH}"                

        self.model = builder.model_builder({'NAME': self.cfg.MODEL})        
        state_dict = torch.load(self.cfg.CKPT_PATH, map_location=self.device)
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        self.model.load_state_dict(base_ckpt)
        self.model.to(self.device)
        self.model.eval()
        
        print(f'{self.cfg.MODEL} initialised, batch_size_limit: {self.batch_size_limit}')

    def inference(self, pts, gtboxes=None, batch_size_limit=None, resample_num=1024, k=30, eps=0.4):
        """
        pts: np.array (N, 3) or list(np.array) of shape (N,3)
        gtboxes: list(np.array) each of shape (7)
        ret_dict: 
            coarse: torch.Tensor (B, 1024, 3), whole completed surface of car
            surface: torch.Tensor (B, N, 3),k nearest completed points for each input lidar point
            cluster: torch.Tensor (B, N, 3), largest cluster from the surface pts.
        """
        resample = ResamplePoints({'n_points': resample_num})
        if type(pts) == list:
            resampled = np.concatenate([resample(pc)[np.newaxis,...] for pc in pts], axis=0)
            if batch_size_limit is not None:
                num_objs = resampled.shape[0]
                pad_num = batch_size_limit*np.ceil(num_objs/batch_size_limit) - num_objs
                pad_batch = np.concatenate([resampled, np.zeros((int(pad_num),resample_num,3))], axis=0)
                resampled_batch = np.split(pad_batch, pad_batch.shape[0]//batch_size_limit)
                if self.norm_with_gt:
                    gt = np.vstack(gtboxes)[:,:7]
                    gt_batch = np.concatenate([gt, np.zeros((int(pad_num),7))], axis=0)
                    gt_batch_list = np.split(gt_batch, pad_batch.shape[0]//batch_size_limit)
        else:
            resampled = resample(pts)[np.newaxis,...]

        out_dict = {}
        in_pc = torch.from_numpy(resampled).float().to(self.device)
        
        if batch_size_limit is not None:
            coarse = []
            t0 = time.time()
            for idx in range(len(resampled_batch)):
                in_dict = {}
                in_dict['input'] = torch.from_numpy(resampled_batch[idx]).float().to(self.device)
                if self.norm_with_gt:
                    in_dict['gt_boxes'] = torch.from_numpy(gt_batch_list[idx]).float().to(self.device)
                
                ret_dict = self.model(in_dict)
                
                coarse.append(ret_dict['coarse'])

            output = torch.cat(coarse, dim=0)[:num_objs,:,:]           
        else:
            in_dict = {}
            in_dict['input'] = in_pc
            if self.norm_with_gt:
                gt = np.vstack(gtboxes)[:,:7]
                in_dict['gt_boxes'] = torch.from_numpy(gt).float().to(self.device)

            ret_dict = self.model(in_dict)
            output = ret_dict['coarse']

        pred_surface = get_partial_mesh_batch(in_pc, output, k=k)    
        pred_cluster = get_largest_cluster_batch( pred_surface, 
                                            eps=eps, 
                                            min_points=2, 
                                            total_pts=output.shape[1])

        out_dict['input'] = in_pc.cpu().numpy()
        out_dict['surface'] = pred_surface
        out_dict['clustered'] = pred_cluster
        out_dict['coarse'] = output.detach().cpu().numpy()
        return out_dict