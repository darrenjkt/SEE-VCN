import os
import sys
sys.path.append('/VCN')
from pathos.pools import ProcessPool
import glob
import open3d as o3d
import numpy as np
from utils.vis_utils import *
from utils import misc
from utils.transform import *
import pickle
from pathlib import Path
from tqdm import tqdm
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def generate_views(scene, eye):
    # [y,z,x]
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=45,
        center=[0, 0, 0],
        eye=eye,
        up=[0, 1, 0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points).to_legacy()
    np_pc = np.asarray(pcd.points)
    return np_pc

def get_object_surface(scene, complete_pts=16384):
    pcs = []
    pcs.append(generate_views(scene, eye=[1,1,1]))
    pcs.append(generate_views(scene, eye=[-1,1,1]))
    pcs.append(generate_views(scene, eye=[1,1,-1]))
    pcs.append(generate_views(scene, eye=[-1,1,-1]))
    pcs.append(generate_views(scene, eye=[1,-1,1]))
    pcs.append(generate_views(scene, eye=[-1,-1,1]))
    pcs.append(generate_views(scene, eye=[1,-1,-1]))
    pcs.append(generate_views(scene, eye=[-1,-1,-1]))
    pcs.append(generate_views(scene, eye=[0,0,-1]))
    pcs.append(generate_views(scene, eye=[0,0,1]))
    pcs.append(generate_views(scene, eye=[-1,0,0]))
    pcs.append(generate_views(scene, eye=[1,0,0]))
    pts = np.concatenate(pcs, axis=0)
    
    pc = torch.from_numpy(np.expand_dims(pts,axis=0)).cuda().float()
    surface = misc.fps(pc, complete_pts).squeeze(0).cpu().numpy()
    return surface

def main(m_id):
    obj_path = '/VCN/data/shapenet/ShapeNetCore.v2/02958343/%s/models/model_normalized.obj'
    path = obj_path % m_id
    if Path(path).exists():
        obj = o3d.io.read_triangle_mesh(path)
        obj.remove_unreferenced_vertices() 
        obj.remove_degenerate_triangles()
        obj.remove_duplicated_vertices()

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(obj)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)
        obj_surface = get_object_surface(scene, complete_pts=16384) 
        if np.isnan(obj_surface).mean(dtype=bool):
            print(f'm_id: {m_id} is nan')
            return False

        save_dir = Path(path).parent
        o3d.io.write_point_cloud(str(save_dir / f'model_surface.pcd'), convert_to_o3dpcd(obj_surface))

    return True

if __name__ == "__main__":
    
    data_dir = '/VCN/data/shapenet/ShapeNetCore.v2/02958343'
    model_glob = glob.glob(f'{data_dir}/*/')
    models = set([model.split('/')[-2] for model in model_glob])

    for m_id in tqdm(models, total=len(models), colour='green', desc='Surfaces'):
        main(m_id)
        
    
