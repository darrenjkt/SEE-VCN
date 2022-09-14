import os
import time
import numpy as np
import pickle
import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import datasets.shared_utils as shared_utils
import open3d as o3d

CAMERA_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
class2idx = {'pedestrian':0, 'car':2, 'truck': 7}

class NuscenesObjects:
    def __init__(self, cfg, cfg_path):
        self.cfg = cfg.DATASET        
        self.root_dir = Path(self.cfg.DATA_DIR)
        self.config_tag = f'{Path(cfg_path).resolve().stem}_{cfg.EXTRA_TAG}' if cfg.EXTRA_TAG != '' else Path(cfg_path).resolve().stem
        self.save_dir = self.root_dir / 'samples'  / f'vcn_{self.config_tag}'                
        self.split = self.cfg.SPLIT
        
        self.version = os.path.basename(self.root_dir)
        if not self.version in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']:
            self.version = 'v1.0-trainval' # assume a custom subset of the trainval set

        self.nusc = NuScenes(version=self.version, dataroot=self.root_dir, verbose=True)    
        self.sample_records = self.get_sample_records()
        self.infos = self.load_infos()
        self.classes = self.cfg.CLASSES
        self.nsweeps = self.cfg.LIDAR_NSWEEPS
        self.dataset_name = self.cfg.NAME        

        if cfg.get('PC_ISOLATION', False):
            self.camera_channels = cfg.PC_ISOLATION.IMG_DET.get('CAMERA_CHANNELS', [])        
            if type(self.camera_channels) is not list:
                self.camera_channels = [self.camera_channels]
            self.mask_dir = self.root_dir / 'masks' / cfg.PC_ISOLATION.IMG_DET.MODEL
            self.masks = self.load_masks()
            self.shrink_mask_percentage = cfg.PC_ISOLATION.IMG_DET.get('SHRINK_MASK_PERCENTAGE', 0)        
                    
    def __len__(self):
        """
        Returns number of samples in the dataset. This is the number of
        all scenes joined together
        """
        return len(self.sample_records)
    
    def load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks

    def get_sample_records(self):
        from nuscenes.utils import splits
        """
        Get sample records from the given scenes. 

        For custom scenes, we define a set of scenes where there are 
        more than 250 cars in the scene with at least 50 points..        
        """
        print("Loading sample records...")
        if self.split == 'train':
            if self.cfg.get('CUSTOM_SCENES', False):
                print("Using custom scenes...")
                ftrain = open(self.root_dir / 'ImageSets' / 'custom_train_scenes.txt', 'r').read()            
                scene_names = ftrain.split('\n')
                self.scenes = [nusc_scene for nusc_scene in self.nusc.scene if nusc_scene['name'] in scene_names]
            else: 
                self.scenes = [nusc_scene for nusc_scene in self.nusc.scene if nusc_scene['name'] in splits.train]
        else:
            self.scenes = [nusc_scene for nusc_scene in self.nusc.scene if nusc_scene['name'] in splits.val]            

        frame_num = 0
        sample_records = {}
        for scene in self.scenes:
            current_sample_token = scene['first_sample_token']
            while(current_sample_token != ''):
                current_sample = self.nusc.get('sample', current_sample_token)
                sample_records[frame_num] = current_sample
                current_sample_token = current_sample['next']
                frame_num += 1
        
        print(f'Total number of samples = {len(sample_records)}')
        return sample_records
    
    def load_infos(self):
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / f'nuscenes_infos_10sweeps_train.pkl'), 'rb') as f:
            train_infos = pickle.load(f)
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / f'nuscenes_infos_10sweeps_val.pkl'), 'rb') as f:
            val_infos = pickle.load(f)

        return {'train': train_infos, 'val': val_infos}

    def get_save_fname(self, idx):
        sample_token = self.get_sample_token_from_idx(idx)
        return str(self.save_dir / f'{sample_token}#{idx:06}')
    
    def find_info_idx(self, infos, token):
        for i, dic in enumerate(infos):        
            if dic['token'] == token:
                return i
        return -1

    def export_image_fnames_for_split():
        for cam in self.camera_channels:
            save_dir = self.root_dir / 'ImageSets' / f'{self.split}_image_paths'
            save_dir.mkdir(parents=True, exist_ok=True)

            f = open(str(save_dir / f'{cam}.txt'), 'w')
            for i in self.sample_records.keys():
                srec = self.sample_records[i]        
                rel_campath = '/'.join(self.nusc.get_sample_data_path(srec['data'][cam]).split('/')[-3:])        
                f.write(rel_campath + '\n')                
            f.close()

    def get_infos(self, idx):
        sample_record = self.sample_records[idx]
        sample_token = sample_record['token']

        infos_idx = self.find_info_idx(self.infos[self.split], sample_token)
        if infos_idx == -1:
            splits = list(self.infos.keys())
            splits.remove(self.split)
            infos_idx = self.find_info_idx(self.infos[splits[0]], sample_token)
            infos = self.infos[splits[0]][infos_idx]
            if infos_idx == -1:
                print("Sample token not found in infos")
                return None
        else:
            infos = self.infos[self.split][infos_idx]

        return infos

    def update_infos(self):
            
        saved_files = glob.glob(f'{str(self.save_dir)}/*')
        
        s_tok = [pcd.split('#')[-2].split('/')[-1] for pcd in saved_files]
        rel_pcds = ['/'.join(pcd.split('/')[-3:]) for pcd in saved_files]
        tok_path = dict(zip(s_tok, rel_pcds))
        
        new_infos = []
        for sample_token, path in tqdm(tok_path.items(), total=len(rel_pcds), desc="Updating infos"):            
            infos_idx = self.find_info_idx(self.infos[self.split], sample_token)            
            if infos_idx == -1:
                splits = list(self.infos.keys())
                splits.remove(self.split)
                infos_idx = self.find_info_idx(self.infos[splits[0]], sample_token)
                infos = self.infos[splits[0]]
            else:                  
                infos = self.infos[self.split]

            new_info = infos[infos_idx].copy()
            new_info['completed_lidar_path'] = path
            gt_boxes = infos[infos_idx]['gt_boxes']
            opcd = o3d.io.read_point_cloud(str(self.root_dir / path))
            o3dboxes = [shared_utils.boxpts_to_o3dbox(shared_utils.opd_to_boxpts(box)) for box in gt_boxes]
            objs = [opcd.crop(o3dbox) for o3dbox in o3dboxes]    
            num_pts = [len(obj.points) for obj in objs]
            new_info['num_completed_lidar_pts'] = np.array(num_pts)

            new_infos.append(new_info)

        savepath = self.root_dir / f'infos_{self.config_tag}'
        savepath.mkdir(parents=True, exist_ok=True)

        info_path = str(savepath / f'nuscenes_infos_{self.split}.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(new_infos, f)
            print(f"Saved updated train infos: {info_path}")

        print(f'Complete: {len(saved_files)} processed')        

    def get_image(self, idx, channel, brightness=1, return_token=False):
        sample_record = self.sample_records[idx]
        img_token = sample_record['data'][channel]
        img_file = self.nusc.get_sample_data_path(img_token)
        img = Image.open(img_file).convert("RGB")
        
        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if return_token:
            return np.array(img), img_token
        else:
            return np.array(img)
    
    def get_pointcloud(self, idx, nsweeps=0, return_token=False, as_nparray=True):
        if nsweeps == 0:
            nsweeps = self.nsweeps
        
        sample_record = self.sample_records[idx]
        pc_sweeps, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample_record, chan='LIDAR_TOP', ref_chan='LIDAR_TOP', nsweeps=nsweeps)
        if as_nparray:
            pc_sweeps = pc_sweeps.points.T[:,:3]
        
        if return_token:
            lidar_token = sample_record['data']['LIDAR_TOP']
            return pc_sweeps, lidar_token
        else:
            return pc_sweeps
    
    def get_lidar_token_from_idx(self, idx):
        sample_record = self.sample_records[idx]
        return sample_record['data']['LIDAR_TOP']
    
    def get_camera_token_from_idx(self, idx, channel):
        sample_record = self.sample_records[idx]
        return sample_record['data'][channel]
    
    def get_sample_token_from_idx(self, idx):
        sample_record = self.sample_records[idx]
        return sample_record['token']
    
    def get_camera_instances(self, idx, channel):
        sample_record = self.sample_records[idx]
        image_tok = sample_record['data'][channel]
        img_fpath = self.nusc.get_sample_data_path(image_tok)
        img_fname = Path(img_fpath).stem
        
        ann_ids = self.masks[channel].getAnnIds(imgIds=[img_fname], catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True)
        return instances

    def map_pointcloud_to_image(self, idx, camera_channel, nsweeps=None, min_dist=1.0):
        """
        Code mostly adapted from the original nuscenes-devkit:
        https://github.com/nutonomy/nuscenes-devkit/blob/cbca1b882aa4fbaf8714ea0d0897457e60a5caae/python-sdk/nuscenes/nuscenes.py#L834

        The difference is that the outputs are modified to fit within
        the existing code structure.
        """
        if nsweeps is None:
            nsweeps = self.nsweeps

        img, camera_token = self.get_image(idx, camera_channel, return_token=True)
        pc, lidar_token = self.get_pointcloud(idx, nsweeps=nsweeps, return_token=True, as_nparray=False)
        pc_lidar = np.array(pc.points, copy=True)

        cam = self.nusc.get('sample_data' , camera_token)
        pointsensor = self.nusc.get('sample_data', lidar_token)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: project points from camera frame to image using camera intrinsics
        pc_cam = pc.points[:3, :]
        depths = pc.points[2, :]
        pts_2d = view_points(pc_cam, np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        fov_inds = np.ones(depths.shape[0], dtype=bool)
        fov_inds = np.logical_and(fov_inds, depths > min_dist)
        fov_inds = np.logical_and(fov_inds, pts_2d[0, :] > 0)
        fov_inds = np.logical_and(fov_inds, pts_2d[0, :] < img.shape[1])
        fov_inds = np.logical_and(fov_inds, pts_2d[1, :] > 0)
        fov_inds = np.logical_and(fov_inds, pts_2d[1, :] < img.shape[0])

        # Restrict points to the camera's fov        
        imgfov = {"pc_lidar": pc_lidar[:3, fov_inds].T,
                  "pc_cam": pc_cam[:3, fov_inds].T,
                  "pts_img": np.floor(pts_2d[:2, fov_inds]).astype(int).T,
                  "fov_inds": fov_inds,
                  "img_shape": img.shape[:2] }
        return imgfov
    
    # def get_det_instances(self, idx, camera_channels=None, min_dist=1.0, use_bbox=False, shrink_percentage=0):
    #     """
    #     Returns the pointclouds of the individual objects, bounded by the 
    #     segmentation mask or bbox.
    #     """
    #     start = time.time()        
        
    #     if camera_channels is None:
    #         camera_channels = self.camera_channels

    #     if type(camera_channels) is not list:
    #         camera_channels = [camera_channels]

    #     if shrink_percentage == None:
    #         shrink_percentage = self.shrink_mask_percentage

    #     proj_clouds = []
    #     for camera_channel in camera_channels:
    #         camera_token = self.get_camera_token_from_idx(idx, channel=camera_channel)
    #         lidar_token = self.get_lidar_token_from_idx(idx)
    #         img = self.get_image(idx, channel=camera_channel)

    #         imgfov = self.map_pointcloud_to_image(idx, camera_channel, self.nsweeps, min_dist=min_dist)

    #         instances = self.get_camera_instances(idx, channel=camera_channel)
    #         instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
    #                                                     instances, 
    #                                                     imgfov,
    #                                                     shrink_percentage=shrink_percentage, 
    #                                                     use_bbox=use_bbox)

    #         proj_clouds.append(proj_dict)

    #     return i_clouds
        
    def render_pointcloud_in_image(self, idx, 
                                    camera_channel, 
                                    mask=False, 
                                    nsweeps=None, 
                                    min_dist=1.0, 
                                    point_size=15, 
                                    brightness=1, 
                                    shrink_percentage=0):
        """
        Project LiDAR points to image and draw
        """        
        if nsweeps is None:
            nsweeps = self.nsweeps
            
        camera_token = self.get_camera_token_from_idx(idx, channel=camera_channel)
        lidar_token = self.get_lidar_token_from_idx(idx)
        img = self.get_image(idx, channel=camera_channel, brightness=brightness)
        
        imgfov = self.map_pointcloud_to_image(idx, camera_channel, nsweeps=nsweeps, min_dist=min_dist)
        imgfov['img_shape'] = img.shape[:2] # H, W
        if mask == True:
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        use_bbox=False,
                                                        shrink_percentage=shrink_percentage)
            
            try:
                all_instance_uv = np.vstack(instance_pts['img_uv'])
                all_instance_cam = np.vstack(instance_pts['cam_xyz'])
                projected_points = np.hstack((all_instance_uv[:,:2], all_instance_cam[:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=instances, clip_distance=min_dist, point_size=point_size)
            except:
                print('No mask; drawing whole pointcloud instead')
                projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
        else:         
            projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
            shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
            
