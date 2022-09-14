import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask
import open3d as o3d
from pathlib import Path
import json
from detector2d.common_utils.mask_utils import mask2polygon, segm2json, xyxy2xywh
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from datasets.shared_utils import draw_lidar_on_image, convert_to_o3dpcd, db_scan

class SEE_VCN_Demo:
    def __init__(self, root_dir, camera, config_file, checkpoint_file, gpu_id):
        # Set configs here
        self.root_dir = Path(root_dir)
        self.camera = camera
        self.det2d_model = init_detector(config_file, checkpoint_file, device=f'cuda:{gpu_id}')
        self.file_ids = [pth.stem for pth in list((self.root_dir / 'pcd').glob('*'))]

        
    def get_calibration(self, idx):
        calib_file = self.root_dir / 'calib'/ f'{self.file_ids[idx]}.json'
        assert calib_file.exists(), f'No calib file found at {calib_file}'
        with open(calib_file) as f:
            return json.load(f) 
    
    def get_camera_instances(self, img, score_thresh=0.3, class_ids=[2]):
        """
        Load mmdetection instance segmentor to get mask
        """
        result = inference_detector(self.det2d_model, img)
        segm_json_result = segm2json(result, class_ids, score_thresh=score_thresh, mask=True)
        
        coco_output = {}
        coco_output['images'] = [{'id': 0, 'height': img.shape[0], 'width': img.shape[1]}]
        coco_output['annotations'] = []
        instance_id = 0
        for segm in segm_json_result:
            instance_id = instance_id + 1
            annotation_info = {
                "id": instance_id,
                "image_id": 0,
                "category_id": segm["category_id"],
                "iscrowd": 0,
                "bbox": segm['bbox']
            }
            annotation_info['segmentation'] = mask2polygon(segm['segmentation'])
            coco_output['annotations'].append(annotation_info)
            
        with open('demo.json', 'w+') as f:
            json.dump(coco_output, f)
            
        cocomask = COCO('demo.json')
        ann_ids = cocomask.getAnnIds(imgIds=0, catIds=[2]) # car is cat_id=2
        instances = cocomask.loadAnns(ann_ids)
        self.coco = cocomask
        return instances
    
    def get_pointcloud(self, idx):        
        pcd_path = self.root_dir / 'pcd' / f'{self.file_ids[idx]}.pcd'
        assert Path(pcd_path).exists(), f'No lidar file found at {pcd_path}'
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        return np.asarray(pcd.points)
    
    def get_image(self, idx):
        img_path = self.root_dir / 'image' / self.camera / f'{self.file_ids[idx]}.jpg'
        assert Path(img_path).exists(), f'No image file found at {img_path}'
        return mmcv.imread(img_path, channel_order='rgb')
    
    def map_pointcloud_to_image(self, calib, image, points, camera_model='pinhole', min_dist=1.0):
        """
        Filter lidar points, keep those in image FOV
        """
        IMG_H, IMG_W, _ = image.shape
        cameramat = np.array(calib['intrinsic']).reshape((3,3))
        lidar2cam = np.array(calib['extrinsic']).reshape((4,4))
        distcoeff = np.array(calib['distcoeff'])
        
        pts_3d_hom = np.hstack((points, np.ones((points.shape[0],1)))).T # (4,N)
        pts_imgframe = (lidar2cam[:3,:] @ pts_3d_hom).T # (3,4) * (4,N) = (3,N)        

        tmpxC = pts_imgframe[:,0] / pts_imgframe[:,2]
        tmpyC = pts_imgframe[:,1] / pts_imgframe[:,2]

        pre_distortion_mask = (pts_imgframe[:,2] > 0) & (abs(tmpxC) < np.arctan(IMG_W/IMG_H)) # Before distortion
        tmpxC = tmpxC[pre_distortion_mask]
        tmpyC = tmpyC[pre_distortion_mask]
        depth = pts_imgframe[:,2][pre_distortion_mask]

        r2 = tmpxC ** 2 + tmpyC ** 2
        if camera_model == "equidistant": # aka. fisheye                        
            r1 = np.sqrt(r2)
            a0 = np.arctan(r1)
            a1 = a0*(1 + distcoeff[0] * (a0**2) + distcoeff[1]* (a0**4) + distcoeff[2]* (a0**6) + distcoeff[3]* (a0**8))
            u =(a1/r1)*tmpxC
            v =(a1/r1)*tmpyC
        elif camera_model == "pinhole":
            tmpdist= 1 + distcoeff[0]*r2 + distcoeff[1]*(r2**2) + distcoeff[4]*(r2**3)
            u = tmpxC*tmpdist+2*distcoeff[2]*tmpxC*tmpyC+distcoeff[3]*(r2+2*tmpxC**2)
            v = tmpyC*tmpdist+distcoeff[2]*(r2+2*tmpyC**2)+2*distcoeff[3]*tmpxC*tmpyC
        else:
            raise NotImplementedError

        u = (cameramat[0,0]*u + cameramat[0,2])[...,np.newaxis]
        v = (cameramat[1,1]*v + cameramat[1,2])[...,np.newaxis]
        uv = np.hstack([u,v, depth[...,np.newaxis]])
        
        fov_inds =  (uv[:,0] > 0) & (uv[:,0] < IMG_W -1) & \
            (uv[:,1] > 0) & (uv[:,1] < IMG_H -1)

        combined_mask = np.zeros(pre_distortion_mask.shape, dtype=np.bool)
        combined_mask[pre_distortion_mask] = fov_inds
        imgfov = {"pc_lidar": points[combined_mask,:],
                  "pc_cam": uv[fov_inds,:], # same as pts_img, just here to keep it consistent across datasets
                  "pts_img": np.round(uv[fov_inds,:],0).astype(int),
                  "fov_inds": combined_mask }
        return imgfov
    
    def isolate(self, pts, eps_scaling=5, vres=1.0, min_lidar_pts=30):
        raw_pcds = convert_to_o3dpcd(pts)
        ring_heights = [np.linalg.norm(pcd.get_center()) * np.tan(vres * np.pi/180) for pcd in raw_pcds]
        eps = [eps_scaling * ring_height for ring_height in ring_heights]
        out_pcd = [db_scan(raw_pcds[i], eps=eps[i], return_largest_cluster=True) for i in range(len(raw_pcds)) if len(raw_pcds[i].points)>5]
        pcds = [np.asarray(pcd.points) for pcd in out_pcd if len(pcd.points) > min_lidar_pts]
        return pcds
    
    def replace_with_completed_pts(self, points, sc_instances, point_distance_thresh=0.2):
        # Remove the original points of the car
        original_pcd = convert_to_o3dpcd(points)
        dist = original_pcd.compute_point_cloud_distance(convert_to_o3dpcd(sc_instances))
        cropped_inds = np.where(np.asarray(dist) < 0.2)[0]
        pcd_without_object = np.asarray(original_pcd.select_by_index(cropped_inds, invert=True).points)

        # Replace with complete points
        pcd_with_all_complete = np.vstack((sc_instances, pcd_without_object))
        return pcd_with_all_complete