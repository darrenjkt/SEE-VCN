import os
import json
import datasets.shared_utils as shared_utils
from PIL import Image, ImageEnhance
from pathlib import Path
import glob
import pickle
import time
import open3d as o3d
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import math
import itertools
from tqdm import tqdm
tf.enable_eager_execution()
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# Order of enum taken from dataset.proto on official waymo-dataset github
#LIDAR_CHANNELS = ['TOP','FRONT','SIDE_LEFT','SIDE_RIGHT','REAR']
#CAMERA_CHANNELS = ['FRONT','FRONT_LEFT','FRONT_RIGHT','SIDE_LEFT','SIDE_RIGHT']
#idx2lidar = {v+1:k for v, k in enumerate(LIDAR_CHANNELS)}
#idx2camera = {v+1:k for v, k in enumerate(CAMERA_CHANNELS)}
class2idx = {'Pedestrian':0, 'Car':2}

class WaymoObjects:

    def __init__(self, cfg, cfg_path):
        self.cfg = cfg.DATASET
        self.dataset_name = self.cfg.NAME
        self.classes = self.cfg.CLASSES
        self.root_dir = Path(self.cfg.DATA_DIR)
        self.processed_data_dir = self.root_dir / 'waymo_processed_data_v0_5_0'        
        self.config_tag = f'{Path(cfg_path).resolve().stem}_{cfg.EXTRA_TAG}' if cfg.EXTRA_TAG != '' else Path(cfg_path).resolve().stem
        self.save_dir = self.root_dir / f'vcn_{self.config_tag}'                        

        if cfg.get('PC_ISOLATION', False):
            self.camera_channels = cfg.PC_ISOLATION.IMG_DET.get('CAMERA_CHANNELS', [])        
            self.mask_dir = self.root_dir / 'image_lidar_projections' / 'masks' / cfg.PC_ISOLATION.IMG_DET.MODEL
            self.masks = self.load_masks()
            self.idx2tokens = self.load_tokens()

            # This might not be necessary anymore if we use lidar sem seg
            self.shrink_mask_percentage = cfg.PC_ISOLATION.IMG_DET.get('SHRINK_MASK_PERCENTAGE', 0)                
            

        self.record_files = glob.glob(str(self.root_dir.parent / "raw_data/*.tfrecord"))
        self.infos = []
        self.split = self.cfg.SPLIT
        self.load_infos()

    def load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks

    def __len__(self):
        return len(self.infos)

    def load_infos(self):
        # We only use this for training
        split_dir = self.root_dir / 'ImageSets' / f'{self.split}.txt'
        self.sample_sequence_list = [os.path.splitext(x.strip())[0] for x in open(split_dir).readlines()]
        waymo_infos = []
        for sequence_name in self.sample_sequence_list:
            info_path = self.processed_data_dir / sequence_name /('%s.pkl' % sequence_name)
            assert info_path.exists(), f"info_path at {info_path} does not exist."
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)
        print('Total samples for Waymo dataset: %d' % len(waymo_infos))

        if self.cfg.SAMPLED_INTERVAL > 1:
            sampled_waymo_infos = []
            for k in range(0, len(waymo_infos), self.cfg.SAMPLED_INTERVAL):
                sampled_waymo_infos.append(waymo_infos[k])
            self.infos = sampled_waymo_infos
        else:
            self.infos = waymo_infos
        print('Total sampled samples for Waymo dataset: %d' % len(self.infos))
    
    def load_tokens(self):
        idx2tokens = {}
        for camera in self.camera_channels:
            tokens = self.masks[camera].getImgIds(catIds=[])
            idx2tokens[camera] = {k:v for k,v in enumerate(tokens)}
        return idx2tokens

    def get_save_fname(self, idx):
        info = self.get_infos(idx)
        sequence_name = info['point_cloud']['lidar_sequence']
        sample_idx = info['point_cloud']['sample_idx']
        
        return str(self.save_dir / sequence_name /f'{sample_idx:04}')

    def get_infos(self, idx):
        return self.infos[idx]

    def find_info_idx(self, seq, fid):
        
        for i, dic in enumerate(self.infos):
            if dic['point_cloud']['lidar_sequence'] == seq and int(dic['point_cloud']['sample_idx']) == int(fid):
                return i
        return -1

    def update_infos(self):
        for sequence_name in tqdm(self.sample_sequence_list, total=len(self.sample_sequence_list), desc='Updating infos'):

            saved_pcds = glob.glob(f'{str(self.save_dir)}/{sequence_name}/*.pcd')
            if not saved_pcds:
                continue
                
            seq_infos = []
            for pcd_file in saved_pcds:
                frame_id = int(Path(pcd_file).stem) 
                rel_path = '/'.join(pcd_file.split('/')[-3:])
                infos_idx = self.find_info_idx(sequence_name, frame_id)
                self.infos[infos_idx]['completed_lidar_path'] = rel_path
                seq_infos.append(self.infos[infos_idx])

            seq_info_path = self.save_dir / sequence_name / f'{sequence_name}.pkl'
            with open(seq_info_path, 'wb') as f:
                pickle.dump(seq_infos, f)
        
        savepath = self.root_dir / f'infos_{self.config_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        
        master_infos_path = savepath / f'waymo_infos_{self.split}.pkl'
        with open(master_infos_path, 'wb') as f:
            pickle.dump(self.infos, f)        

    def get_pointcloud(self, idx, disable_nlz_flag=False, tanhnorm=False):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']
        path = self.processed_data_dir / sequence_name /f'{sample_idx:04}.npy'
        point_features = np.load(path)
        points_all, NLZ_flag = point_features[:,0:5], point_features[:, 5]
        if disable_nlz_flag:            
            points_all = points_all[NLZ_flag == -1]            
        if tanhnorm:
            points_all[:, 3] = np.tanh(points_all[:,3])

        return points_all[:,:3]

    
    def get_image(self, idx, camera_channel, brightness=1):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']

        img_path = self.root_dir / 'image_lidar_projections' / 'image' / camera_channel / f'{sequence_name}_{sample_idx:04}.png'
        img = Image.open(img_path).convert("RGB")

        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def map_pointcloud_to_image(self, idx, camera_channel):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']

        imgpc_path = self.root_dir / 'image_lidar_projections' / 'image_pc' / camera_channel / f'{sequence_name}_{sample_idx:04}.npy'
        pts_img = np.load(imgpc_path)
        fovinds_path = self.root_dir / 'image_lidar_projections' / 'fov_inds' / camera_channel / f'{sequence_name}_{sample_idx:04}.npy'
        fov_inds = np.load(fovinds_path)

        pc_lidar = self.get_pointcloud(idx)[fov_inds,:]

        imgfov = {"pc_lidar": pc_lidar,
                  "pts_img": pts_img,
                  "pc_cam": None,
                  "fov_inds": fov_inds}
        return imgfov
    
    def get_camera_instances(self, idx, channel):
        infos = self.get_infos(idx)
        sequence_name = infos['point_cloud']['lidar_sequence']
        sample_idx = infos['point_cloud']['sample_idx']
        image_id = f'{sequence_name}_{sample_idx:04}'
        ann_ids = self.masks[channel].getAnnIds(imgIds=[image_id], catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances

    def get_det_instances(self, idx, camera_channel, use_bbox=True):
        image = self.get_image(idx, camera_channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(idx, camera_channel=camera_channel)
        instances = self.get_camera_instances(idx, camera_channel)
        instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                    instances, 
                                                    imgfov, 
                                                    use_bbox=use_bbox)
        filtered_icloud = [x for x in instance_pts['pointcloud'] if len(x) != 0]
        return filtered_icloud
    
    def render_pointcloud_in_image(self, idx, camera_channel, mask=False, use_bbox=False, min_dist=1.0, point_size=5, brightness=1):
        
        image = self.get_image(idx, camera_channel=camera_channel)
        imgfov = self.map_pointcloud_to_image(idx, camera_channel=camera_channel)
        imgfov['img_shape'] = img.shape[:2] # H, W
        if mask == True:
            instances = self.get_camera_instances(idx, camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        use_bbox=use_bbox)
            try:
                # For waymo we already concatenated the depth
                all_instance_uvd = np.vstack(instance_pts['img_uv'])
                shared_utils.draw_lidar_on_image(all_instance_uvd, image, instances=instances, clip_distance=min_dist, point_size=point_size)
            except:
                print('No points in mask; drawing whole pointcloud instead')
                shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
        else:                        
            shared_utils.draw_lidar_on_image(imgfov['pts_img'], image, instances=None, clip_distance=min_dist, point_size=point_size)
    

