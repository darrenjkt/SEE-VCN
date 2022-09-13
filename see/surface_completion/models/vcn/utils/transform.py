import numpy as np
import torch
import torch.nn as nn
from ..utils.misc import check_numpy_to_torch

def rot_from_heading(heading):
    """
    Input (B) heading array.
    Output (B,3,3) rotation matrix array
    """
    yaw, _ = check_numpy_to_torch(heading)
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)
    zeros = yaw.new_zeros(len(yaw))
    ones = yaw.new_ones(len(yaw))

    # Clockwise rotation is
    # [[0  -1]
    #  [1   0]]
    # Anti-clockwise rotation is
    # [[0   1]
    #  [-1  0]]
    # Here we take anti-clockwise as convention and so it's
    # applied in the conversion

    rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
    return rot_matrix

def rotate_points_along_z(points, angle):
    """
    For view-centric to canonical, use -angle
    For canonical to view-centric, use +angle
    The above is the same as rot_mat.T and rot_mat respectively

    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def vc_to_cn_rt(points, rot, trans):
    """
    Transform the given points from view-centric to canonical frame using B 3 3 rot mat and B 1 3 trans

    This function assumes the convention where given rotation matrix specifies the rotation of the object 
    from 0 rads. Hence why we invert the rotation matrix with permute to reset it to canonical.

    Args:
        points: (B, N, 3)
        rot: (B, 3, 3)
        trans: (B, 1, 3) or (B, 3)
    """
    if len(trans.shape) == 2:
        trans = trans.unsqueeze(1)

    return torch.matmul(points - trans, rot.permute(0,2,1))

def cn_to_vc_rt(points, rot, trans):
    """
    Transform the given points from view-centric to canonical frame using B 3 3 rot mat and B 1 3 trans
    Args:
        points: (B, N, 3)
        rot: (B, 3, 3)
        trans: (B, 1, 3) or (B, 3)
    """
    if len(trans.shape) == 2:
        trans = trans.unsqueeze(1)

    return torch.matmul(points, rot) + trans


def vc_to_cn(points, gt_label):
    """
    Transform the given points from view-centric to canonical frame using ground truth box format

    For view-centric to canonical, use -angle
    For canonical to view-centric, use +angle

    Args:
        points: (B, N, 3)
        gt_label: (B, 7), angle along z-axis, angle increases x ==> y
    """
    assert gt_label.shape[1] == 7, f'gt_label wrong shape, should be (B 7) but given shape is {gt_label.shape}'
    assert points.shape[2] == 3, f'points wrong shape, should be (B N 3) but given shape is {points.shape}.'

    points, is_numpy = check_numpy_to_torch(points)
    gt_label, _ = check_numpy_to_torch(gt_label)

    centre = gt_label[:,:3].unsqueeze(1)
    centre_points = points - centre
    points_cn = rotate_points_along_z(centre_points, -gt_label[:,-1])

    return points_cn.numpy() if is_numpy else points_cn

def cn_to_vc(points, gt_label):
    """
    Transform the given points from canonical frame to view-centric

    For view-centric to canonical, use -angle
    For canonical to view-centric, use +angle

    Args:
        points: (B, N, 3)
        gt_label: (B, 7), angle along z-axis, angle increases x ==> y
    """
    assert gt_label.shape[1] == 7, f'gt_label wrong shape, should be (B 7) but given shape is {gt_label.shape}'
    assert points.shape[2] == 3, f'points wrong shape, should be (B N 3) but given shape is {points.shape}.'

    points, is_numpy = check_numpy_to_torch(points)
    gt_label, _ = check_numpy_to_torch(gt_label)

    centre = gt_label[:,:3].unsqueeze(1)    
    points_rot = rotate_points_along_z(points, gt_label[:,-1])
    points_vc = points_rot + centre

    return points_vc.numpy() if is_numpy else points_vc

def normalize_scale(points, gt_label):
    """
    Normalize the scale of the object with the length of the gt box

    Args:
        points: (B, N, 3)
        gt_label: (B, 7), angle along z-axis, angle increases x ==> y
    """
    assert gt_label.shape[1] == 7, f'gt_label wrong shape, should be (B 7) but given shape is {gt_label.shape}'
    assert points.shape[2] == 3, f'points wrong shape, should be (B N 3) but given shape is {points.shape}.'

    return points/gt_label[:,3].view(-1,1,1)

def restore_scale(points, gt_label):
    """
    Restore the scale of the object with the length of the gt box

    Args:
        points: (B, N, 3)
        gt_label: (B, 7), angle along z-axis, angle increases x ==> y
    """
    assert gt_label.shape[1] == 7, f'gt_label wrong shape, should be (B 7) but given shape is {gt_label.shape}'
    assert points.shape[2] == 3, f'points wrong shape, should be (B N 3) but given shape is {points.shape}.'

    return points * gt_label[:,3].view(-1,1,1)

def rotm_to_heading(R):
    """
    Returns heading (euler rotation around z-axis)
    
    This gets the rotation by applying the rotation matrix to a vector that 
    is perpendicular to the axis of rotation. 
    """
    assert len(R.shape) == 3, f'Shape (B,3,3) expected. Instead received {R.shape}.'
    v1 = torch.tensor([1,0,0]).cuda().float()
    v2 = torch.matmul(v1, R)
    
    # Cross product to get sin value
    v1 = v1.tile(R.shape[0], 1)
    num = torch.cross(v1,v2, dim=1)
    den = (torch.linalg.norm(v1, axis=1) * torch.linalg.norm(v2, axis=1)).unsqueeze(1)
    sinth = num/den
    
    # Dot product to get cos value
    p1 = torch.matmul(v1, v2.permute(1,0))
    p2 = torch.linalg.norm(v1,axis=1)
    p3 = torch.linalg.norm(v2,axis=1)
    costh = (p1 / (p2*p3)).diag()
    
    # atan2 to get result between [-pi, pi]
    return torch.atan2(sinth[:,-1], costh)