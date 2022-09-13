import numpy as np
import os
import glob
import time
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datasets.shared_utils import convert_to_o3dpcd, cfg_from_yaml_file
import setproctitle
import open3d as o3d
from SEEv2 import SEEv2
import shutil

from torch.multiprocessing import Process, Pool

def process_gt(sample_idx):    
    t0 = time.time()

    # Continue where we left off
    saved_pcd_path = Path(seev2.data_obj.get_save_fname(sample_idx) + '.pcd')
    if saved_pcd_path.exists():
        pcd = o3d.io.read_point_cloud(str(saved_pcd_path))
        if len(pcd.points) != 0:
            return 0.0, None        

    pcd_gtboxes = seev2.get_pcd_gtboxes(sample_idx)    
    if pcd_gtboxes is None:
        return 0.0, None 

    isolated_pts, gt_labels = seev2.isolate_gt_pts(pcd_gtboxes)
    inst_dict = seev2.complete_gt_pts(isolated_pts, gt_labels)
    final_pcd = seev2.replace_with_completed_pts(pcd_gtboxes['pcd'], 
                                                inst_dict['all_instances'], 
                                                point_dist_thresh=seev2.replace_distance_thresh)    
    
    time_taken_frame = time.time() - t0    
    if inst_dict['all_instances'] is not None:
        time_taken_car = time_taken_frame/len(inst_dict['coarse'])
    else:
        time_taken_car = None

    seev2.save_pcd(final_pcd, sample_idx)
    return time_taken_frame, time_taken_car

def process_det(sample_idx): 
    t0 = time.time()

    proj_dict = seev2.get_det_instances(sample_idx)
    isolated_pts = seev2.isolate_det_pts(proj_dict)
    inst_dict = seev2.complete_det_pts(isolated_pts)
    final_pcd = seev2.replace_with_completed_pts(convert_to_o3dpcd(seev2.data_obj.get_pointcloud(sample_idx)), 
                                                inst_dict['all_instances'], 
                                                point_dist_thresh=seev2.replace_distance_thresh)  

    time_taken_frame = time.time() - t0
    if inst_dict['all_instances'] is not None:
        time_taken_car = time_taken_frame/len(inst_dict['coarse'])
    else:
        time_taken_car = None

    seev2.save_pcd(final_pcd, sample_idx)
    return time_taken_frame, time_taken_car

def run(num_proc):
    """
    Runs the processing of the dataset in parallel with the specified
    isolation method
    """    
    if cfg.get('PC_ISOLATION', False):
        process = process_det
    else:
        print('Using gt boxes to isolate points')
        process = process_gt
    
    t1 = time.time()
    sample_indices = range(0, seev2.data_obj.__len__())

    avg_time = []
    print(f'\nGenerating {num_proc} processes')
    mp = Pool(processes=num_proc)
    time_taken = list(tqdm(mp.imap(process,sample_indices), total=seev2.data_obj.__len__()))
    avg_time.extend(time_taken)
    mp.close()
    mp.join()

    # Update infos with the meshed paths
    seev2.data_obj.update_infos()

    avg_frame = [i[0] for i in avg_time]
    avg_car = [i[1] for i in avg_time if i[1] is not None]
    print(f'Average time per frame = {sum(avg_frame)/len(avg_frame):0.5f}s')
    print(f'Average time per car = {sum(avg_car)/len(avg_car):0.5f}s')
    print(f'Time taken for {seev2.data_obj.__len__()} files: {time.time()-t1}')
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extrapolate the surface of all instances in the point cloud near their input points')
    parser.add_argument('--cfg_file', required=True, help='Specify cfg_file')
    parser.add_argument('--num_proc', default=3, type=int, help='Specify cfg_file')
    args = parser.parse_args()
    cfg = cfg_from_yaml_file(args.cfg_file)

    # Copy cfg file to keep on record
    cfg_filepath = Path(args.cfg_file).resolve()
    config_tag = f'{cfg_filepath.stem}_{cfg.EXTRA_TAG}' if cfg.EXTRA_TAG != '' else cfg_filepath.resolve().stem        
    save_cfg_filepath = Path(cfg.DATASET.DATA_DIR) / f'infos_{config_tag}' / cfg_filepath.name
    save_cfg_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_filepath, save_cfg_filepath)

    return args, cfg   

args, cfg = parse_args()    
seev2 = SEEv2(cfg=cfg, cfg_path=args.cfg_file)

if __name__ == "__main__":   
    
    torch.multiprocessing.set_start_method('spawn') 
    run(num_proc=args.num_proc)
