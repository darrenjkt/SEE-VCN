import os
import glob
import random
from pathlib import Path
import shutil
import argparse
import yaml
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--gpus', type=str, help='gpu id(s)')
    parser.add_argument('--dist', action='store_true', help='use multi-gpu distributed training')
    parser.add_argument('--dir_name', type=str, help='queue folder for specific gpu')
    args = parser.parse_args()
    
    if args.gpus is None:
        args.gpus = '0'
    num_gpus = len(args.gpus.split(','))
    print(f'Using {args.gpus}')

    if not args.dist:
        queue_folder = f'gpu{args.gpus}'
    else:
        queue_folder = 'dist'

    queue_dir = f'models/queue/{queue_folder}'    
    queued = sorted(glob.glob(f'{queue_dir}/*'))
    completed = []

    for folder in queued:     
        print(f'current queue: {queued}')           
        cur_folder = Path(folder)
        exp_name = str(cur_folder.stem)
        if exp_name in completed:
            continue

        model_cfg = cur_folder / 'config.yaml'
        model_py = cur_folder / 'model.py'
        kitti_dataset = cur_folder / 'KITTI.yaml'        
        vccars_dataset = cur_folder / 'VC_cars.yaml'        
        resume_flag = (cur_folder / '.resume').exists()
        
        if not model_cfg.exists():
            continue

        with open(model_cfg) as f:
            cfg = yaml.full_load(f)
            
        model_name = cfg['model']['NAME']
        
        dst_modelcfg = f'cfgs/VC_models/{model_name}.yaml'
        dst_modelpy = f'models/{model_name}.py'
        dst_kitti = 'cfgs/dataset_configs/KITTI.yaml'
        dst_vccars = 'cfgs/dataset_configs/VC_cars.yaml'

        shutil.copyfile(model_cfg, dst_modelcfg)
        shutil.copyfile(model_py, dst_modelpy)
        shutil.copyfile(kitti_dataset, dst_kitti)
        shutil.copyfile(vccars_dataset, dst_vccars)

        if args.dist:
            assert num_gpus > 1, "Please specify more than 1 GPU when using distributed training"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            port = str(random.randint(0,15000))

            call = ["bash","./scripts/dist_train.sh",str(num_gpus),port,"--config",f"./cfgs/VC_models/{model_name}.yaml","--exp_name",exp_name]
            if resume_flag:
                call.append("--resume")
            ret = subprocess.call(call)

        else:
            assert num_gpus == 1, "Please specify just 1 GPU if not using distributed training. Else specify --dist."

            call = ["bash","./scripts/train.sh",args.gpus,"--config",f"./cfgs/VC_models/{model_name}.yaml","--exp_name",exp_name]
            if resume_flag:
                call.append("--resume")
            ret = subprocess.call(call)

        if ret == 0:
            completed.append(exp_name)

            # Check for new additions to the queue
            updated_queue = glob.glob(f'{queue_dir}/*')
            diff = set(updated_queue) ^ set(queued)
            queued.extend(diff)
            continue
        else:
            print(f'Error when running: {exp_name}')
            print(f'Error code: {ret}')
            exit()