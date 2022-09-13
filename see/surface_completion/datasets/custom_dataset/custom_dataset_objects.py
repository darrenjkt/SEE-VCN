import cv2
import numpy as np
import matplotlib.pyplot as plt
import datasets.shared_utils as shared_utils
import pickle
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
import open3d as o3d
import json

# For the instance segmentation masks
class2idx = {'Pedestrian':0, 'Car':2, 'Truck': 7}

class CustomDatasetObjects:
    def __init__(self, cfg, cfg_path):
        self.cfg = cfg.DATASET
        self.dataset_name = self.cfg.NAME
        self.classes = self.cfg.CLASSES
        self.root_dir = Path(self.cfg.DATA_DIR)
        self.config_tag = f'{Path(cfg_path).resolve().stem}_{cfg.EXTRA_TAG}' if cfg.EXTRA_TAG != '' else Path(cfg_path).resolve().stem        
        self.split = self.cfg.SPLIT
        self.save_dir = self.root_dir / self.split / f'partialsc_{self.config_tag}'                        
        self.infos = self.load_infos()
        self.frame_ids = [self.infos[i]['point_cloud']['lidar_idx'] for i in range(len(self.infos))]

        if cfg.get('PC_ISOLATION', False):
            self.camera_channels = cfg.PC_ISOLATION.IMG_DET.get('CAMERA_CHANNELS', [])        
            if type(self.camera_channels) is not list:
                self.camera_channels = [self.camera_channels]
            self.mask_dir = self.root_dir / self.split / 'masks' / cfg.PC_ISOLATION.IMG_DET.MODEL
            self.masks = self.load_masks()
            self.shrink_mask_percentage = cfg.PC_ISOLATION.IMG_DET.get('SHRINK_MASK_PERCENTAGE', 0)        

    def __len__(self):
        """
        Returns number of samples in the dataset
        """
        return len(self.infos)

    def load_masks(self):
        print("Loading masks...")
        masks = {}
        for channel in self.camera_channels:
            mask_json = self.mask_dir / f"{channel}.json"
            masks[channel] = COCO(mask_json)
        return masks
    
    def load_infos(self):
        with open(str(self.root_dir / 'infos' / f'baraja_infos_{self.split}.pkl'), 'rb') as pkl:
            infos = pickle.load(pkl)
        return infos
    
    def find_info_idx(self, infos, frame_id):
        for i, dic in enumerate(infos):        
            if dic['point_cloud']['lidar_idx'] == frame_id:
                return i
        return -1
    
    def update_infos(self):
        saved_files = glob.glob(f'{str(self.save_dir)}/*')
        
        frame_ids = [Path(fname).stem for fname in saved_files]
        rel_pcds = ['/'.join(fname.split('/')[-2:]) for fname in saved_files]
        id_path = dict(zip(frame_ids, rel_pcds))
        
        new_infos = []
        for frame_id, rel_path in tqdm(id_path.items(), total=len(id_path.items()), desc='Updating infos'):
            
            infos_idx = self.find_info_idx(self.infos, frame_id)
            self.infos[infos_idx]['completed_lidar_path'] = rel_path
            new_infos.append(self.infos[infos_idx])

            
        savepath = self.root_dir / f'infos_{self.config_tag}'
        savepath.mkdir(parents=True, exist_ok=True)
        
        infopath = str(savepath / f'baraja_infos_{self.split}.pkl')
        with open(infopath, 'wb') as f:
            pickle.dump(new_infos, f)
            print(f"Saved updated infos: {infopath}")

        print(f'Complete: {len(saved_files)} processed')  
        
    def get_save_fname(self, idx):
        return str(self.save_dir / f'{self.frame_ids[idx]}')

    def get_infos(self, idx):
        frame_id = f'{idx:06}'
        infos_idx = self.find_info_idx(self.infos, frame_id)
        if infos_idx != -1:
            return self.infos[infos_idx]        
        else:
            print(f"frame_id: {frame_id}, not found in infos")
            return None
    
    # Loading methods
    def get_image(self, idx, channel='front', brightness=1):
        
        img_file = self.root_dir / self.split / 'image' / channel / f'{idx:06}.jpg'
        img = Image.open(img_file).convert("RGB")

        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def get_pointcloud(self, idx):        
        lidar_file = self.root_dir / self.split / 'pcd' / f'{idx:06}.pcd'
        assert lidar_file.exists(), f'No lidar file found at {lidar_file}'
        pcd = o3d.io.read_point_cloud(str(lidar_file))
        return np.asarray(pcd.points)
    
    def get_calibration(self, idx):
        calib_file = self.root_dir / self.split / 'calib'/ f'{idx:06}.json'
        assert calib_file.exists(), f'No calib file found at {calib_file}'
        with open(calib_file) as f:
            return json.load(f)                           
        
    def get_camera_instances(self, idx, channel):
        """
        Returns all instances detected by the instance detection for the particular requested sequence
        """
        img_ids=[self.frame_ids[idx]]
        ann_ids = self.masks[channel].getAnnIds(imgIds=img_ids, catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        instances = sorted(instances, key=lambda x: x['area'], reverse=True) 
        return instances

    def map_pointcloud_to_image(self, idx, camera_channel, min_dist=1.0):
        """
        Filter lidar points, keep those in image FOV. Set min_dist=1m to prevent
        getting points on the car bonnet
        """
        points = self.get_pointcloud(idx)
        img = self.get_image(idx, channel=camera_channel)
        calib = self.get_calibration(idx)

        IMG_H, IMG_W, _ = img.shape
        cameramat = np.array(calib['intrinsic']).reshape((3,3))
        camera2sensorframe = np.array(calib['extrinsic']).reshape((4,4))

        pts_3d_hom = np.hstack((points, np.ones((points.shape[0],1)))).T # (4,N)
        pts_imgframe = np.dot(camera2sensorframe[:3], pts_3d_hom) # (3,4) * (4,N) = (3,N)
        image_pts = np.dot(cameramat, pts_imgframe).T # (3,3) * (3,N)

        image_pts[:,0] /= image_pts[:,2]
        image_pts[:,1] /= image_pts[:,2]
        uv = image_pts.copy()
        fov_inds =  (uv[:,0] > 0) & (uv[:,0] < IMG_W -1) & \
                    (uv[:,1] > 0) & (uv[:,1] < IMG_H -1)     
        fov_inds = fov_inds & (points[:,0]>min_dist)   

        imgfov = {"pc_lidar": points[fov_inds,:],
                  "pc_cam": image_pts[fov_inds,:], # same as pts_img, just here to keep it consistent across datasets
                  "pts_img": np.round(uv[fov_inds,:],0).astype(int),
                  "fov_inds": fov_inds,
                  "img_shape": img.shape[:2] }
        return imgfov
        

    def get_mask_instance_clouds(self, idx, use_bbox=False, min_dist=1.0, shrink_percentage=None):
        """
        Returns the individual clouds for each mask instance. 
        
        Return: list of (N,4) np arrays (XYZL), each corresponding to one object instance (XYZ points) with a label (L)
        """
        if type(self.camera_channels) is not list:
            self.camera_channels = [self.camera_channels]

        if shrink_percentage == None:
            shrink_percentage = self.shrink_mask_percentage

        points = self.get_pointcloud(idx)
        calib = self.get_calibration(idx)

        i_clouds = []
        for camera_channel in self.camera_channels:                    

            # We only limit the point cloud range when getting instance points. 
            # Once we got the instances, we concat it with the whole range pcd
            img = self.get_image(idx, channel=camera_channel)

            # Project to image
            imgfov = self.map_pointcloud_to_image(points, calib, img, min_dist=min_dist)        
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        shrink_percentage=shrink_percentage,
                                                        use_bbox=use_bbox,
                                                        labelled_pcd=False)

            filtered_icloud = [x for x in instance_pts['lidar_xyzls'] if len(x) != 0]
            i_clouds.extend(filtered_icloud)
        
        return i_clouds   

    def render_pointcloud_in_image(self, idx, 
                                    camera_channel, 
                                    mask=False, 
                                    draw_bbox=False,
                                    min_dist=1.0, 
                                    point_size=2, 
                                    brightness=1, 
                                    shrink_percentage=0):
        """
        Project LiDAR points to image and draw
        """        
        imgfov = self.map_pointcloud_to_image(idx, camera_channel=camera_channel)
        img = self.get_image(idx, channel=camera_channel)
        if mask == True:
            instances = self.get_camera_instances(idx, channel=camera_channel)
            instance_pts = shared_utils.get_pts_in_mask(self.masks[camera_channel], 
                                                        instances, 
                                                        imgfov,
                                                        shrink_percentage=shrink_percentage)
            
            try:
                all_instance_uv = np.vstack(instance_pts['img_uv'])
                all_instance_cam = np.vstack(instance_pts['cam_xyz'])
                projected_points = np.hstack((all_instance_uv[:,:2], all_instance_cam[:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=instances, clip_distance=min_dist, point_size=point_size, shrink_percentage=shrink_percentage, instance_mask=mask, draw_bbox=draw_bbox)
            except Exception as e:
                # Some frames don't have a mask
                print(e)
                print('No points in mask; drawing whole pointcloud instead')
                projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
                shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)
        else:
            projected_points = np.hstack((imgfov['pts_img'][:,:2], imgfov['pc_cam'][:,2][:,np.newaxis]))
            shared_utils.draw_lidar_on_image(projected_points, img, instances=None, clip_distance=min_dist, point_size=point_size)