import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import kitti_utils
import datasets.shared_utils as shared_utils
import pickle
import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO

# CAMERA_CHANNELS = ['image_2', 'image_3']
class2idx = {'Pedestrian':0, 'Car':2}

class KittiObjects:
    def __init__(self, cfg, cfg_path):

        self.cfg = cfg.DATASET
        self.dataset_name = self.cfg.NAME
        self.classes = self.cfg.CLASSES
        self.root_dir = Path(self.cfg.DATA_DIR)
        self.config_tag = f'{Path(cfg_path).resolve().stem}_{cfg.EXTRA_TAG}' if cfg.EXTRA_TAG != '' else Path(cfg_path).resolve().stem
        self.save_dir = self.root_dir / 'training' / f'vcn_{self.config_tag}'                        
        self.split = self.cfg.SPLIT
        self.infos = self.load_infos()
        self.frame_ids = [self.infos[i]['image']['image_idx'] for i in range(len(self.infos))]

        if cfg.get('PC_ISOLATION', False):
            self.camera_channels = cfg.PC_ISOLATION.IMG_DET.get('CAMERA_CHANNELS', [])        
            if type(self.camera_channels) is not list:
                self.camera_channels = [self.camera_channels]
            self.mask_dir = self.root_dir / 'training' / 'masks' / cfg.PC_ISOLATION.IMG_DET.MODEL
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
        with open(str(self.root_dir / 'infos_openpcdetv0.3.0' / f'kitti_infos_{self.split}.pkl'), 'rb') as f:
            return pickle.load(f)

    def get_save_fname(self, idx):
        return str(self.save_dir / f'{self.frame_ids[idx]}')

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

        infopath = str(savepath / f'kitti_infos_{self.split}.pkl')
        with open(infopath, 'wb') as f:
            pickle.dump(new_infos, f)
            print(f"Saved updated infos: {infopath}")

        print(f'Complete: {len(saved_files)} processed')  
    
    def get_infos(self, idx):
        return self.infos[idx]
        
    # Loading methods
    def get_image(self, idx, channel, brightness=1):
        
        img_file = self.root_dir / 'training' / f'{channel}/{self.frame_ids[idx]}.png'
        img = Image.open(img_file).convert("RGB")
        # change brightness if desired. 1 is to keep as original
        if brightness != 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)

        return np.array(img)

    def get_pointcloud(self, idx, append_labels=False, add_ground_lift=False): 
        lidar_file = self.root_dir / 'training' / f'velodyne/{self.frame_ids[idx]}.bin'
        xyz_pts = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,4))[:,:3]

        if append_labels:
            # Get labels
            sample_infos = self.get_infos(idx)
            pcd_gtboxes = shared_utils.populate_gtboxes(sample_infos, "kitti", self.classes, add_ground_lift=add_ground_lift)

            ## (X,Y,Z,OBJ_ID,CLASS)
            # Obj ID is [0,N] for objs, and -1 for stuff
            # Class is 1 for car, 0 for others
            o3dpcd = shared_utils.convert_to_o3dpcd(xyz_pts)
            obj_pcds = [o3dpcd.crop(gtbox) for gtbox in pcd_gtboxes['gt_boxes']]
            obj_pts = np.concatenate([np.asarray(obj.points) for obj in obj_pcds])
            dists = [o3dpcd.compute_point_cloud_distance(obj) for obj in obj_pcds]
            cropped_inds = np.concatenate([np.where(np.asarray(d) < 0.01)[0] for d in dists])
            pcd_without_objects = np.asarray(o3dpcd.select_by_index(cropped_inds, invert=True).points)

            
            obj_np = [np.array(obj.points) for obj in obj_pcds]
            obj_np_ids = np.vstack([np.hstack([obj, np.ones((len(obj),1))*(o_id+1)]) for o_id, obj in enumerate(obj_np)])
            obj_np_ids_carlabel = np.hstack([obj_np_ids, np.ones((len(obj_np_ids), 1))])
            pcd_without_objects_id = np.hstack([pcd_without_objects, -1*np.ones((len(pcd_without_objects), 1))])
            pcd_without_objects_label = np.hstack([pcd_without_objects_id, np.zeros((len(pcd_without_objects), 1))])
            labelled_pcd = np.vstack([obj_np_ids_carlabel, pcd_without_objects_label])

            return labelled_pcd
        else:
            return xyz_pts


    def get_calibration(self, idx):
        calib_file = self.root_dir / 'training' / f'calib/{self.frame_ids[idx]}.txt'
        return kitti_utils.Calibration(calib_file)
        
    def get_camera_instances(self, idx, channel):
        """
        Returns all instances detected by the instance detection for the particular requested sequence
        """
        # if htc mask, then convert to int.
        img_ids=[int(self.frame_ids[idx])]

        # For non-htc masks. Moving forward, this will be the norm since nuscenes/waymo are not integer frame ids
        # img_ids=[self.frame_ids[idx]]
        ann_ids = self.masks[channel].getAnnIds(imgIds=img_ids, catIds=[class2idx[c] for c in self.classes])
        instances = self.masks[channel].loadAnns(ann_ids)
        return instances

    def map_pointcloud_to_image(self, idx, camera_channel, min_dist=1.0):
        """
        Filter lidar points, keep those in image FOV
        """
        pc_velo = self.get_pointcloud(idx)
        calib = self.get_calibration(idx)
        image = self.get_image(idx, channel=camera_channel)
        pts_2d = calib.project_velo_to_imageuv(pc_velo)
        
        # We keep the indices to associate any processing in image domain with the original lidar points
        # min_dist is to keep any lidar points further than that minimum distance
        IMG_H, IMG_W, _ = image.shape
        fov_inds = (pts_2d[:,0]<IMG_W) & (pts_2d[:,0]>=0) & \
            (pts_2d[:,1]<IMG_H) & (pts_2d[:,1]>=0)
        fov_inds = fov_inds & (pc_velo[:,0]>min_dist)
        imgfov_pc_velo = pc_velo[fov_inds,:]
        imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

        imgfov = {"pc_lidar": imgfov_pc_velo,
                  "pc_cam": imgfov_pc_rect,
                  "pts_img": np.floor(pts_2d[fov_inds,:]).astype(int) ,
                  "fov_inds": fov_inds,
                  "img_shape": image.shape[:2] }
        return imgfov
        

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
        imgfov['img_shape'] = img.shape[:2] # H, W
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