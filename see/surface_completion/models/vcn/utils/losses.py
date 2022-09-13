import numpy as np
import torch
import torch.nn as nn
from ..utils.misc import in_hull
from ..utils.bbox_utils import *

def geodesic_distance(m1, m2):
    """
    Computes geodesic distance for two matrices m1,m2 shape B 3 3
    https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/sanity_test/code/tools.py#L282
    
    Max geo dist for rot is geo_loss = pi
    """
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
        
    theta = torch.acos(cos)
    return theta

def sin_loss(heading1, heading2, loss_func=None):
    """
    Sin loss: sin(a - b) = sinacosb-cosasinb
    
    This loss treats pi and 0 as the same. They just use an
    additional direction classifier to distinguish the two. The
    reason given by SECOND paper is that 0 and pi gives the same
    box, but one is heavily penalised when it is misidentified.
    By using sin loss, they directly model IoU against the angle
    offset function. 
    
    Direction classifier: if heading > 0, then +ve, else negative.
    i.e. [-pi,0) is -ve, and [0, pi] is +ve
    
    Loss func: e.g. nn.SmoothL1Loss()
    """
    term1 = torch.sin(heading1) * torch.cos(heading2)
    term2 = torch.sin(heading2) * torch.cos(heading1)
    
    if loss_func is None:
        return term1 - term2
    else:
        return loss_func(term1, term2)

def angle_between_vectors(v1, v2):
    """
    This function uses the dot product to calculate the non-directional 
    angle between two vectors.

    With dot product, the angle computed is the included angle between 
    two vectors - and thus always a value between 0 and pi. 
    """

    # Between two single vectors. Shape (2)
    if len(v1.shape) == 1:
        num = torch.dot(v1, v2)
        den = torch.linalg.norm(v1) * torch.linalg.norm(v2)
        
        return torch.acos(num/den)

    # Between a batch of vectors. Shape (B 2)
    elif len(v1.shape) == 2:
        p1 = torch.matmul(v1, v2.permute(1,0))
        p2 = torch.linalg.norm(v1,axis=1)
        p3 = torch.linalg.norm(v2,axis=1)
        p4 = p1 / (p2*p3)
        return torch.acos(torch.clip(p4,-1.0,1.0)).diag()

    
def points_outside_hull(pred, gt):
    """
    Returns the percentage of points that are outside the hull.
    Warning: This is very very slow. Not used for now.

    pred (B N 3)
    gt (B N 3)
    """
    hull_err = []
    for i in range(len(pred)):
        mask = in_hull(pred[i], gt[i])
        num_out_of_hull = len(mask) - np.count_nonzero(mask)            
        hull_err.append( num_out_of_hull/len(mask) )
        
    return np.array(hull_err).mean()


def get_oob_error(pc, gt_box):
    """
    pc (B N 3)
    gt_box (B 7)

    """
    oob_error = []
    for i in range(len(pc)):
        oob_pts = get_oob_points(pc[i], gt_box[i])
        num_oob = len(oob_pts)
        num_pc = len(torch.unique(pc[i]))
        oob_error.append( num_oob/num_pc )

    return torch.from_numpy(np.array(oob_error)).cuda().float()
