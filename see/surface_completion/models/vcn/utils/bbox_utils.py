import numpy as np
import torch
import torch.nn as nn
from ..utils.vis_utils import *
from ..utils.misc import check_numpy_to_torch
from ..utils.transform import rot_from_heading

def get_dims(pts):
    """
    Returns a bbox in the format 
    (N,3): [dx,dy,dz]

    Input is of shape B N 3

    This method assumes that heading = 0 rad. 
    """
    maxpts, _ = torch.max(pts, dim=1, keepdim=True)
    minpts, _ = torch.min(pts,dim=1, keepdim=True)
    
    # print("maxpts.shape = ", maxpts.shape)
    # print("minpts.shape = ", minpts.shape)
    l1 = maxpts[:,:,0] - minpts[:,:,0]
    l2 = maxpts[:,:,1] - minpts[:,:,1]
    l3 = maxpts[:,:,2] - minpts[:,:,2]
    dims = torch.stack([l1, l2, l3],dim=2)

    return dims.squeeze(1)

def get_bbox_from_keypoints(pts, gt_box):
    """
    Returns a bbox in the format 
    (N,3): [x,y,z,dx,dy,dz,heading]

    Input is of shape B N 3, B 7

    The only thing it uses from the gt_box label is heading. 
    Centre/dims are estimated from the points directly.
    """
    gt_rmat = rot_from_heading(gt_box[:,-1]).cuda()
    maxpts, _ = torch.max(pts, dim=1, keepdim=True)
    minpts, _ = torch.min(pts, dim=1, keepdim=True)
    meanbounds_centre = (maxpts + minpts)/2

    # V -> C we use rmat.T, C->V we use rmat
    norm_pts = torch.bmm(pts - meanbounds_centre, gt_rmat.permute(0,2,1))
    dims = get_dims(norm_pts).cuda()

    return torch.cat([meanbounds_centre.squeeze(1), dims, gt_box[:,-1].unsqueeze(1).cuda()],dim=1)

def get_oob_points(pc, gt_box, istorch=False):
    """
    Return the points that are out of the bounding box.

    pc (N 3)
    gt_box (7)

    """
    if isinstance(gt_box, torch.Tensor):
        gt_box = gt_box.cpu().numpy()        

    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
        istorch = True

    lboxpts = opd_to_boxpts(gt_box)
    labelbox = boxpts_to_o3dbox(lboxpts)

    o3dpc = convert_to_o3dpcd(pc)
    cropped = o3dpc.crop(labelbox)
    cropped_np = np.asarray(cropped.points)

    pc_set = set(map(tuple, pc))
    crop_set = set(map(tuple, cropped_np))
    oob = np.array(list(pc_set - crop_set))
    if istorch:
        return torch.from_numpy(oob).cuda().float()
    else:
        return oob
    
