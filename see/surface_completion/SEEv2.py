import os
import glob
import time
import torch 
import numpy as np
from pathlib import Path
import open3d as o3d
from scipy.spatial import cKDTree

from datasets.nuscenes.nuscenes_objects import NuscenesObjects
from datasets.waymo.waymo_objects import WaymoObjects
from datasets.kitti.kitti_objects import KittiObjects
from datasets.custom_dataset.custom_dataset_objects import CustomDatasetObjects
from datasets.shared_utils import convert_to_o3dpcd, populate_gtboxes, db_scan, get_pts_in_mask

from models.PartialSC import PartialSC
# from models.Cluster import Cluster

__DATASETS__ = {
    'nuscenes': NuscenesObjects,
    'kitti': KittiObjects,
    'waymo': WaymoObjects,
    'custom': CustomDatasetObjects
}

class SEEv2:
    def __init__(self, cfg, cfg_path, gpu_id=0):
        
        self.data_obj = __DATASETS__[cfg.DATASET.NAME]( cfg=cfg, cfg_path=cfg_path)                
        
        if cfg.get('PC_ISOLATION', False):
            self.vres = cfg.PC_ISOLATION.VRES
            self.eps_scaling = cfg.PC_ISOLATION.EPS_SCALING
            self.max_eps = cfg.PC_ISOLATION.MAX_EPS
            self.min_eps = cfg.PC_ISOLATION.MIN_EPS

        self.min_lidar_pts = cfg.SURFACE_COMPLETION.MIN_LIDAR_PTS
        self.replace_distance_thresh = cfg.SURFACE_COMPLETION.REPLACE_DISTANCE_THRESH
        if cfg.SURFACE_COMPLETION.get('USE_SEEV1', False):
            self.use_seev1 = cfg.SURFACE_COMPLETION.USE_SEEV1
            self.vres = cfg.SURFACE_COMPLETION.VRES
        else:
            self.use_seev1 = False
            self.partialsc = PartialSC(cfg=cfg.SURFACE_COMPLETION.PARTIALSC, gpu_id=gpu_id)

    def get_pcd_gtboxes(self, idx, add_ground_lift=True, ground_lift_height=0.1):
        """
        Get the pcd and boxes for a single frame. 
        add_ground_lift: adds 20cm to box centroid to lift it off the ground to avoid getting ground points
        """
        sample_infos = self.data_obj.get_infos(idx)
        if sample_infos == 'ignore':
            return None

        o3dpcd = convert_to_o3dpcd(self.data_obj.get_pointcloud(idx))            
        pcd_gtboxes = populate_gtboxes(sample_infos, self.data_obj.dataset_name, self.data_obj.classes, add_ground_lift=add_ground_lift, ground_lift_height=ground_lift_height)        
        pcd_gtboxes['pcd'] = o3dpcd            

        return pcd_gtboxes

    def isolate_gt_pts(self, pcd_gtboxes):
        pcds, gt_labels = [], []
        for object_id in range(len(pcd_gtboxes['gt_boxes'])):

            # Crop pcd to get pts in gt box
            gt_box = pcd_gtboxes['gt_boxes'][object_id]
            obj_pcd = pcd_gtboxes['pcd']

            try:
                cropped_pcd = obj_pcd.crop(gt_box)
                if len(cropped_pcd.points) >= self.min_lidar_pts:
                    pcds.append(np.asarray(cropped_pcd.points))
                    if self.use_seev1:
                        gt_labels.append(pcd_gtboxes['gt_boxes'][object_id])
                    else:
                        gt_labels.append(pcd_gtboxes['xyzlwhry_gt_boxes'][object_id])
            except Exception as e:                
                print(e)          
                print('Lifting the ground height can cause some weird box dimensions, but those are not cars so we skip them')
                print('gt_box = ', gt_box)      

        return pcds, gt_labels


    def complete_gt_pts(self, isolated_pts, gt_labels):
        """
        Given the ground truth boxes and their pcds, extract the points in each gt bbox.

        Complete the surface of the isolated points.        
        """            
        
        if not isolated_pts:
            return {'all_instances': None}
        else:
            if self.use_seev1:
                sc_model_ret = {}
                sc_model_ret['coarse'] = []
                isolated_o3dpcd = convert_to_o3dpcd(isolated_pts)    
                for obj_id, pcd in enumerate(isolated_o3dpcd):
                    bp_mesh = self.seev1_ball_pivoting(pcd)
                    sampled_pcd = self.seev1_sampling(bp_mesh, len(pcd.points), gt_box=gt_labels[obj_id])
                    if sampled_pcd:
                        sc_model_ret['coarse'].append(np.asarray(sampled_pcd.points))

                sc_model_ret['all_instances'] = np.vstack(sc_model_ret['coarse'])
            else:
                sc_model_ret = self.partialsc.inference(isolated_pts, 
                                                        gtboxes=gt_labels, 
                                                        batch_size_limit=self.partialsc.batch_size_limit,                                         
                                                        k=self.partialsc.surface_sel_k,
                                                        eps=self.partialsc.cluster_eps)

                sc_model_ret['all_instances'] = np.unique(np.vstack(sc_model_ret['clustered']), axis=0)

            return sc_model_ret
    
    def get_det_instances(self, 
                        idx, 
                        min_dist=1.0, 
                        shrink_percentage=None, 
                        use_bbox=False,
                        camera_channels=None):

        if shrink_percentage is None:
            shrink_percentage = self.data_obj.shrink_mask_percentage
        if camera_channels is None:
            camera_channels = self.data_obj.camera_channels

        proj_clouds = []
        for camera_channel in camera_channels:

            # Project to image
            imgfov = self.data_obj.map_pointcloud_to_image(idx, camera_channel=camera_channel)        
            instances = self.data_obj.get_camera_instances(idx, channel=camera_channel)
            proj_dict = get_pts_in_mask(self.data_obj.masks[camera_channel], 
                                        instances, 
                                        imgfov,
                                        shrink_percentage=shrink_percentage, 
                                        use_bbox=use_bbox)

            proj_clouds.append(proj_dict)
        return proj_clouds   

    def isolate_det_pts(self, in_proj_dict, min_cluster=10):
        # If more than one camera, we merge the instances lists
        # Might need to tinker here for nuscenes due to low res.
        proj_dict = {}
        for key in in_proj_dict[0].keys():
            for pd in in_proj_dict:
                proj_dict.setdefault(key, [])
                proj_dict[key].extend(pd[key])

        instances = []
        for idx in range(len(proj_dict['lidar_xyz'])):
            uv = proj_dict['img_uv'][idx]
            xyz = proj_dict['lidar_xyz'][idx]
            if xyz.shape[0] > min_cluster:                

                # # Find the K nearest points to the centre of the image mask
                # img_label = proj_dict['img_labels'][idx]
                # mask_centroid = np.flip(np.argwhere(img_label['bin_mask'] == 1).mean(axis=0))
                # kd = cKDTree(uv)                 
                # knn_query_num = int(np.ceil(min(self.mask_query_max, self.mask_query_pct * len(xyz))))
                # knn_idx = kd.query(mask_centroid,k=knn_query_num)[1]

                # Db scan clustering
                pcd = convert_to_o3dpcd(xyz)
                dist = np.linalg.norm(pcd.get_center())
                ring_height = dist * np.tan(self.vres * np.pi/180)
                eps = np.clip(self.eps_scaling * ring_height, a_max=self.max_eps, a_min=self.min_eps)
                cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=3))
                y = np.bincount(cluster_labels[cluster_labels >= 0])
                if len(y) > 0:    
                    sel_cluster_id = np.argmax(y)
            
                    cluster_pt_indices = np.argwhere(cluster_labels == sel_cluster_id).squeeze()
                    clustered_pts = xyz[cluster_pt_indices]
                    if clustered_pts.shape[0] > min_cluster:
                        instances.append(clustered_pts)

        return instances

    def merge_multi_camera_detections(self, 
                                    isolated_inst,
                                    min_overlap=3,
                                    min_dist_to_check=3):
        """
        Merge instance clusters across cameras. If we don't do
        this, we may end up generating two cars for the same object
        which can be detrimental for the detector
        """    
        joined = []
        inst_d = [np.linalg.norm(inst.mean(axis=0)) for inst in isolated_inst]
        len_instances = len(isolated_inst)
        for i in range(len_instances):       
            for j in range(len_instances):
                
                if (abs(inst_d[i] - inst_d[j]) < min_dist_to_check) & (i != j) & (j not in joined): # if dist is less than 5m apart we check for same object
                    kd = cKDTree(isolated_inst[i])
                    knn = kd.query_ball_point(isolated_inst[j], r=0.1)
                    num_overlap = np.count_nonzero(knn)

                    # If clusters share more than 3 points, we combine them
                    if num_overlap > min_overlap:
                        isolated_inst.append(np.vstack([isolated_inst[i], isolated_inst[j]]))
                        joined.extend([i,j])
        
        merged_inst = [isolated_inst[i] for i in range(len(isolated_inst)) if i not in joined]
        return merged_inst

    def complete_det_pts(self, isolated_inst):
        """
        Takes as input the isolated clouds from the image instance mask
        """
        # Merge instances across cameras if multi-camera setup
        if len(self.data_obj.camera_channels) > 1:
            merged_inst = self.merge_multi_camera_detections(isolated_inst)
        else:
            merged_inst = isolated_inst

        # Filter min pts
        filtered_inst = [inst for inst in merged_inst if inst.shape[0] > self.min_lidar_pts]

        if not filtered_inst:
            return {'all_instances': None}
        else:
            if self.use_seev1:
                sc_model_ret = {}
                sc_model_ret['coarse'] = []
                isolated_o3dpcd = convert_to_o3dpcd(filtered_inst)    
                for pcd in isolated_o3dpcd:
                    bp_mesh = self.seev1_ball_pivoting(pcd)
                    sampled_pcd = self.seev1_sampling(bp_mesh, len(pcd.points))
                    if sampled_pcd is not None:
                        sc_model_ret['coarse'].append(np.asarray(sampled_pcd.points))

                sc_model_ret['all_instances'] = np.vstack(sc_model_ret['coarse'])
            else:
                sc_model_ret = self.partialsc.inference( filtered_inst,
                                                    batch_size_limit=self.partialsc.batch_size_limit,                                         
                                                    k=self.partialsc.surface_sel_k,
                                                    eps=self.partialsc.cluster_eps)

                sc_model_ret['all_instances'] = np.unique(np.vstack(sc_model_ret['clustered']), axis=0)
            return sc_model_ret
            
    def replace_with_completed_pts(self, original_pcd, sc_instances, point_dist_thresh=0.1):
        """
        Remove the cropped pointcloud from the original pointcloud and replace with the completed pts. 
        
        original_pcd: open3d.Pointcloud
        sc_instances: np.array (N,3) where N is the concatenation of all completed objects
        """
        if sc_instances is None:
            return np.asarray(original_pcd.points)

        # Remove the original points of the car
        dist = original_pcd.compute_point_cloud_distance(convert_to_o3dpcd(sc_instances))
        cropped_inds = np.where(np.asarray(dist) < point_dist_thresh)[0]
        pcd_without_object = np.asarray(original_pcd.select_by_index(cropped_inds, invert=True).points)

        # Replace with complete points
        pcd_with_all_complete = np.vstack((sc_instances, pcd_without_object))

        return pcd_with_all_complete
    
    def save_pcd(self, points, sample_idx):

        save_fname = self.data_obj.get_save_fname(sample_idx)

        # .pcd format from o3d only saves (N,3) shape
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(points)

        save_fname = save_fname + '.pcd'
        os.makedirs(os.path.dirname(save_fname), exist_ok=True)            
        try:
            o3d.io.write_point_cloud(save_fname, save_pcd, write_ascii=False)
        except Exception as e:
            print(f'sample idx: {sample_idx} - pcd pts: {save_pcd} - Error: {e}')

    def seev1_ball_pivoting(self, pcd, upper_radius=1.155, lower_radius=0.01):
        ball_radius = np.linspace(lower_radius, upper_radius, 20)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Invalidate existing normals
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
        bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(ball_radius))

        return bp_mesh

    def seev1_sampling(self, mesh, num_pcd_pts, gt_box=None, optimal_ring_height=0.05):
        if gt_box is not None:
            centroid_distance = np.linalg.norm(gt_box.get_center())
        else:
            centroid_distance = np.linalg.norm(np.asarray(mesh.get_center()))    

        ring_height = centroid_distance * np.tan(self.vres * np.pi/180)    
        upsampling_rate = ring_height/optimal_ring_height
        try:
            return mesh.sample_points_poisson_disk(int(upsampling_rate*num_pcd_pts))
        except:
            return None