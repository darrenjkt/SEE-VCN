# This computes coarse loss with fps sampling, no fine loss or pred
import torch
import torch.nn as nn
from .build import MODELS
from ..utils import misc
from ..extensions.chamfer_dist import ChamferDistanceL2
from ..utils.transform import rot_from_heading, rotate_points_along_z
from ..utils.losses import geodesic_distance
from ..utils.bbox_utils import get_dims, get_bbox_from_keypoints
from ..utils.sampling import get_partial_mesh_batch

def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def conv_layers(layer_dims, last_as_conv=False):
    
    in_channels = layer_dims[0]
    conv_layers = []
    for out_channel in layer_dims[1:]:
        if out_channel == layer_dims[-1] and last_as_conv:
            conv_layers.append(nn.Conv1d(in_channels, out_channel, kernel_size=1))
            break
            
        conv_layers += [nn.Conv1d(in_channels, out_channel, kernel_size=1),
                        nn.BatchNorm1d(out_channel),
                        nn.ReLU()]
        in_channels = out_channel
    return nn.Sequential(*conv_layers)

def fc_layers(layer_dims, last_as_linear=True):
    
    in_channels = layer_dims[0]
    layers = []
    for out_channel in layer_dims[1:]:
        if out_channel == layer_dims[-1] and last_as_linear:
            layers.append(nn.Linear(in_channels,out_channel))
            break
            
        layers += [nn.Linear(in_channels,out_channel),
                    nn.ReLU(inplace=True)]
        in_channels = out_channel            
        
    return nn.Sequential(*layers)

class FeatureEncoder(nn.Module):
    def __init__(self, dims):
        super(FeatureEncoder, self).__init__()
        # 3, 64, 128, 256, 256, 512
        self.mlp_conv1 = nn.Sequential(
            nn.Conv1d(dims[0],dims[1],1),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims[1],dims[2],1)
        )
        self.mlp_conv2 = nn.Sequential(
            nn.Conv1d(dims[3],dims[4],1),
            nn.BatchNorm1d(dims[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(dims[4],dims[5],1)
        )
    def forward(self, x, n, keepdims=False):
        # Pytorch is (B,C,N) format

        feature = self.mlp_conv1(x)  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.mlp_conv2(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=keepdims)[0]
        
        return feature_global

    
@MODELS.register_module()
class VCN_VC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sel_k = 30 # select nearest 30 points to each input point
        
        self.number_coarse = 1024
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(3,64,1),            
            nn.LeakyReLU(),
            nn.Conv1d(64,128,1),
            nn.LeakyReLU(),
            nn.Conv1d(128,1024,1),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self.pose_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 9)            
        )

        self.encoder = FeatureEncoder([3, 128, 256, 512, 512, self.number_coarse])
        self.shape_fc = fc_layers([1024, 1024, 1024, 3*self.number_coarse], last_as_linear=True) # canonical shape
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_coarse = ChamferDistanceL2()
        self.loss_partial = ChamferDistanceL2()
        self.loss_translation = nn.SmoothL1Loss(reduction='none')
        self.loss_dims = nn.SmoothL1Loss(reduction='none')    

    def get_loss(self, ret_dict, in_dict):     
        gt_boxes = in_dict['gt_boxes']

        loss_dict = {}        
        pred_box = get_bbox_from_keypoints(ret_dict['coarse'], gt_boxes) # B 7        
        loss_dict['dims'] = self.loss_dims(gt_boxes[:,3:6].cuda(), pred_box[:,3:6]).mean()

        # Translation loss
        gt_centres = gt_boxes[:,:3]
        loss_dict['translation'] = self.loss_translation(gt_centres.cuda(), ret_dict['reg_centre']).mean()
        
        # Rotation loss
        gt_headings = gt_boxes[:,-1]
        gt_rmats = rot_from_heading(gt_headings)
        theta = geodesic_distance(ret_dict['reg_rot'], gt_rmats)
        loss_dict['rotation'] = theta.mean()

        # Coarse loss - downsample complete with fps
        if in_dict['training']:
            ds_complete = misc.fps(in_dict['complete'], ret_dict['coarse'].shape[1])
            loss_dict['coarse'] = self.loss_coarse(ret_dict['coarse'], ds_complete)            

            pred_surface = get_partial_mesh_batch( in_dict['input'], ret_dict['coarse'], k=self.sel_k)
            gt_surface = get_partial_mesh_batch( in_dict['input'], ds_complete, k=self.sel_k)
            loss_dict['partial'] = self.loss_partial(pred_surface, gt_surface)                        

        return loss_dict

    def forward(self, in_dict):
        ret = {}

        bs , n , _ = in_dict['input'].shape
        pc = in_dict['input']
    
        # Frustum angle is anti-clockwise about x axis (since +y is to the left, following RHS convention)
        frustum_angle = torch.atan2(in_dict['input'][:,:,1].mean(dim=1), in_dict['input'][:,:,0].mean(dim=1))
        pc_fview = rotate_points_along_z(in_dict['input'], -frustum_angle)
    
        # Centre pointcloud on mean of the points
        pts_mean = pc_fview.mean(dim=1).unsqueeze(1)
        pts_meancentered = pc_fview - pts_mean
    
        # Regress the relative pose
        pose_feat = self.pose_encoder(pts_meancentered.permute(0,2,1)).view(bs, -1)  # B 256 n
        rel_pose = self.pose_fc(pose_feat)
        trans = rel_pose[:,:3].unsqueeze(1) # B 1 3
        centre = pts_mean + trans
        rot6d = rel_pose[:,3:9]
        rot_mat = compute_rotation_matrix_from_ortho6d(rot6d)     
        
        pc_cn = torch.matmul(pc_fview - centre, rot_mat.permute(0,2,1))     
    
        # encoder
        feature_global = self.encoder(pc_cn.permute(0,2,1), n)  # B 1024
        coarse = self.shape_fc(feature_global).reshape(-1,self.number_coarse,3) # B coarse_pts 3            
        coarse_vc = torch.matmul(coarse, rot_mat) + centre
        
        # Bring points back to sensor view
        ret['coarse'] = rotate_points_along_z(coarse_vc.contiguous(), frustum_angle)
        
        # Bring regressed frustum view rotation/centroid to sensor view rotation/centroid
        ret['reg_rot'] = torch.matmul(rot_mat, rot_from_heading(frustum_angle))
        ret['reg_centre'] = rotate_points_along_z(centre, frustum_angle).squeeze(1)
                
        return ret