import time
import numpy as np
import torch
import transforms3d
from ..utils.misc import *
from ..utils.transform import rotate_points_along_z

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):

        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)

            if transform.__class__ in [NormalizeObjectPose, 
                                        RandomWorldFlip,
                                        GlobalRotation,
                                        RandomObjectScaling
                                        ]:
                data = transform(data)
            else:
                for k, v in data.items():                    
                    if k in objects and k in data:
                        data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, pts):
        return torch.from_numpy(pts).to(torch.float32)

class RandomPointDropout(object):
    """
    Randomly remove points.
    
    ring_dropout:   probability of dropping out a lidar ring
    point_dropout:  probability of dropping out a point
    vres_bounds:    [min,max] range of vres to determine how wide the ring gap is. 
                    Larger vres e.g. nusc, more spacing between rings
    """
    def __init__(self, parameters):
        self.point_dropout = parameters['point_dropout']

    def __call__(self, pts, min_pts=30):
        # Don't drop points if pointcloud too small
        if len(pts) < min_pts:        
            return pts
        
        sph_pts = cart2sph(pts)

        ## TODO: Try the random dropping from utils/misc.py instead of this

        # Randomly decide to drop points
        if np.random.choice([False, True], replace=False, p=[0.5, 0.5]):
            mask = np.random.choice(a=[False, True], size=len(sph_pts), p=[self.point_dropout,1-self.point_dropout])
        else:
            mask = np.random.choice(a=[True], size=len(sph_pts))    
        
        return sph2cart(sph_pts[mask])

class PeriodicSampling(object):
    """
    Periodic Sampling. By varying the threshold alpha, and period T, we can simulate
    a wide range of different types of irregular sampling.

    mask = |cos(2pi/T * ||xi-centre||) > cos(alpha*pi)

    alpha is a threshold where alpha=0 removes every point, alpha=1 keeps every point
    T is the period. 
    """
    def __init__(self, parameters):
        pass        


    def __call__(self, data):
        # Don't drop points if pointcloud too small
        if len(pts) < min_pts:        
            return pts
        
        pts = data['partial']
        center = data['label']['gtbox'][:3]        

        # Randomly decide to drop points
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            d_p = np.linalg.norm(pts - center, axis=1)
            
            alpha = np.random.uniform(0.5,1.0)
            period = np.random.uniform(0.1,0.5)
            pulse = 2 * np.pi / period

            mask = np.cos(pulse * d_p) > np.cos(np.pi * alpha)
        else:
            mask = np.random.choice(a=[True], size=len(pts))    
        
        return pts[mask]


class DownsampleRings(object):
    """
    Subsample every Nth ring to simulate different number of lidar beams.
    N is randomly chosen.
    
    """
    def __init__(self, parameters):
        pass

    def __call__(self, pts, min_in_pts=50, min_out_pts=30):
        # Don't drop points if pointcloud too small
        if len(pts) < min_in_pts:        
            return pts
        t0 = time.time()

        sph_pts = cart2sph(pts)
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable:
            hist = np.histogram(sph_pts[:,2], bins=50)        
            ring_indices = np.digitize(sph_pts[:,2], hist[1][np.argwhere(hist[0] > 0).squeeze(1)])
            num_rings = max(ring_indices)
            choose_rings = np.unique(ring_indices)[np.random.randint(0,3)::np.random.randint(1,num_rings)]
            mask = np.in1d(ring_indices, choose_rings)        

            if np.count_nonzero(mask) > min_out_pts:
                return sph2cart(sph_pts[mask])

        return pts

class LidarSimulation(object):
    """
    Same as DownsampleRings except that we also sample every Nth point within the
    ring to simulate different number of lidar beams. N is randomly chosen from a
    uniform distribution.

    The idea is to have a very dense sampling in the VC dataset, then here we 
    randomly generate different sorts of car point distributions to emulate all 
    kinds of lidars.
    
    """
    def __init__(self, parameters):
        pass

    def __call__(self, pts, min_in_pts=100, min_out_pts=30, max_sel_n_hpts_1_2_ring=30):
        # Don't drop points if pointcloud too small
        if len(pts) < min_in_pts:        
            return pts

        sph_pts = cart2sph(pts)
        hist = np.histogram(sph_pts[:,2], bins='sqrt')
        ring_indices = np.digitize(sph_pts[:,2], hist[1][np.argwhere(hist[0] > 0).squeeze(1)])
        num_rings = max(ring_indices)

        # Start from a random ring and select every N ring
        # Skip at most num_rings*0.3 since after that, most of the values
        # only lead to giving 3-5 rings. You'd have a predominantly 3-5 ring
        # object dataset...
        sel_n_ring = np.random.randint(1,max(np.ceil(num_rings*0.3),2))
        choose_rings = np.unique(ring_indices)[np.random.randint(0,max(np.ceil(num_rings*0.1),1))::sel_n_ring]
        mask = np.in1d(ring_indices, choose_rings)

        # 20% chance to do a 1-2 ring scenario since not all nuscenes is 1-2 rings
        onetwo_ring = np.random.choice([False, True], replace=False, p=[0.8, 0.2])
        if onetwo_ring and len(choose_rings) > 2:
            onetwo_ring_idxs = np.random.choice(choose_rings, size=np.random.randint(1,3))
            onetwo_mask = np.in1d(ring_indices, onetwo_ring_idxs)
            
        # Skip only at most 50% of the num pts in a ring
        _, counts = np.unique(ring_indices[mask], return_counts=True)
        sel_n_hpts = np.random.randint(1, max(np.ceil(min(counts)*0.5), 2))
        out = sph2cart(sph_pts[mask][np.random.randint(0,min(counts))::sel_n_hpts])

        # If 1-2 ring scenario doesn't have enough points, return the original sampled rings
        if onetwo_ring and len(choose_rings) > 2:
            sel_n_hpts = min(max_sel_n_hpts_1_2_ring, sel_n_hpts)    
            onetworing_pts = sph2cart(sph_pts[onetwo_mask][np.random.randint(0,min(counts))::sel_n_hpts])
            if len(onetworing_pts) < min_out_pts:
                # print(f'in: {len(pts)}, onetwo: {len(onetworing_pts)}, out: {len(out)} [rings: {len(choose_rings)}]')
                return out
            else:
                # print(f'in: {len(pts)}, onetwo: {len(onetworing_pts)}, out: {len(out)}')
                return onetworing_pts
        else:
            if len(out) > min_out_pts:
                # print(f'in: {len(pts)}, out: {len(out)} [rings: {len(choose_rings)}]')
                return out

        # print(f'in: {len(pts)}, out: {len(out)} [rings: {len(choose_rings)}], not enough pts in out...')
        return pts

class Jitter(object):
    """
    Add gaussian noise to the pts. This makes the points not look like a lidar point 
    distribution!! The points no longer look ring-like.
    
    stdev_bounds: [min,max] range of stdev to sample from for gaussian noise generation
    """
    def __init__(self, parameters):
        self.clip = parameters['clip'] if parameters['clip'] is not None else 0.05
        self.sigma = parameters['sigma'] if parameters['sigma'] is not None else 0.01

    def __call__(self, pts):
        N, C = pts.shape
        noise = np.clip(self.sigma * np.random.randn(N,C), -1*self.clip, self.clip)
        return pts + noise

class AddGNSpherical(object):
    """
    This retains the ring-like appearance of the point cloud, yet introducing some noise.

    Add gaussian noise to the pts only for the distance element of spherical coordinates; 
    [0.005,0.01] is not bad.
    
    stdev_bounds: [min,max] range of stdev to sample from for gaussian noise generation
    """
    def __init__(self, parameters):
        
        self.stdev_bounds = [0.005, 0.03]

    def __call__(self, pts):
        enable = np.random.choice([False, True], replace=False, p=[0.2, 0.8])
        if enable:
            noise_stdev = np.random.uniform(self.stdev_bounds[0], self.stdev_bounds[1])
            noise = np.random.normal(0, noise_stdev, len(pts)) 
            mask = np.random.choice([True, False], size=len(pts), p=[0.5,0.5])
            noise[mask] = 0.0

            sph_pts = cart2sph(pts)
            sph_pts[:,0] += noise

            return sph2cart(sph_pts)
        else:
            return pts

class ResamplePoints(object):
    """
    Drop or duplicate points so that pcd has exactly n points.
    """
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, pts):        
        # Duplicate points
        tiled_pts = np.tile(pts, (int(np.ceil(self.n_points/len(pts))),1))

        # Randomly select from duplicated points
        choice = np.random.permutation(tiled_pts.shape[0])
        resampled_pts = tiled_pts[choice[:self.n_points]]

        return resampled_pts

class RandomWorldFlip(object):
    def __init__(self, parameters):
        self.along_axis = parameters['along_axis']        

    def __call__(self, data):
        if 'X' in self.along_axis:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            if enable:
                data['label']['gtbox'][1] = -data['label']['gtbox'][1]
                data['label']['gtbox'][6] = -data['label']['gtbox'][6]
                data['partial'][:, 1] = -data['partial'][:, 1]
                data['complete'][:, 1] = -data['complete'][:, 1]

        elif 'Y' in self.along_axis:
            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
            if enable:
                data['label']['gtbox'][0] = -data['label']['gtbox'][0]
                data['label']['gtbox'][6] = -(data['label']['gtbox'][6] + np.pi)
                data['partial'][:, 0] = -data['partial'][:, 0]
                data['complete'][:, 0] = -data['complete'][:, 0]

        return data

class RandomObjectScaling(object):    
    def __init__(self, parameters):
        self.scale_range = parameters['scale_range']        

    def __call__(self, data):
        """
        Randomly scales the object in each dimension to make the network
        more robust to different variations of object sizes

        Pointcloud input (N,3)
        """

        if self.scale_range[1] - self.scale_range[0] < 1e-3:
            return data        

        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
        if enable: 
            # Transform to canonical frame
            gtbox = data['label']['gtbox']
            partial_cn = rotate_points_along_z((data['partial'] - gtbox[:3])[np.newaxis, :, :], np.array([-gtbox[-1]]))[0]
            complete_cn = rotate_points_along_z((data['complete'] - gtbox[:3])[np.newaxis, :, :], np.array([-gtbox[-1]]))[0]

            noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1], 3)    
            partial_cn[:,:3] *= noise_scale
            complete_cn[:,:3] *= noise_scale
            data['label']['gtbox'][3:6] *= noise_scale

            # Transform back to view-centric frame
            data['partial'] = rotate_points_along_z(partial_cn[np.newaxis, :, :], np.array([gtbox[-1]]))[0] + gtbox[:3]
            data['complete'] = rotate_points_along_z(complete_cn[np.newaxis, :, :], np.array([gtbox[-1]]))[0] + gtbox[:3]

        return data

class GlobalScaling(object):
    def __init__(self, parameters):
        self.scale_range = parameters['scale_range']        

    def __call__(self, data):
        """
        Args:
            gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
            points: (M, 3 + C),
            scale_range: [min, max]
        Returns:
        """
        if self.scale_range[1] - self.scale_range[0] < 1e-3:
            return data
        noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        data['partial'][:,:3] *= noise_scale
        data['complete'][:,:3] *= noise_scale
        data['label']['gtbox'][:6] *= noise_scale

        return data

class GlobalRotation(object):
    def __init__(self, parameters):
        self.rot_range = parameters['rot_range']        

    def __call__(self, data):
        """
        Args:
            gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
            points: (M, 3 + C),
            scale_range: [min, max]
        Returns:
        """
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        data['partial'] = rotate_points_along_z(data['partial'][np.newaxis, :, :], np.array([noise_rotation]))[0]
        data['complete'] = rotate_points_along_z(data['complete'][np.newaxis, :, :], np.array([noise_rotation]))[0]
        data['label']['gtbox'][0:3] = rotate_points_along_z(data['label']['gtbox'][np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0]
        data['label']['gtbox'][6] += noise_rotation

        return data

### ----- Normalized Pose/ Fixed Input Transforms from original PoinTr -----

# (Comments for VC dataset) Doesn't work for non-normalized as it places additional points at 0
# Also not necessary if network can take non-fixed number of points
class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud

# Does the same thing as RandomFlipAlongX/Y but it's only for points, not gtbox
# For opd gtbox format, we'll have to convert it to boxpts first.
class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
