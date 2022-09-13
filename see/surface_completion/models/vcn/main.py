from tools.runner import run_vc, test_net_vc
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import shutil
import torch
from tensorboardX import SummaryWriter

def main():
    # args

    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
            test_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
            test_writer = None
    # config
    config = get_config(args, logger = logger)
    if config.dataset.test._base_.get('EXT_MESH', False) and args.ext_mesh_tag is None:
        raise ValueError('Must define --ext_mesh_tag if using --ext_mesh')

    # save model python file if training or copy from experiment to model folder if testing
    if args.test:
        shutil.copy(os.path.join(args.experiment_path, f'model.py'), f'models/{config.model.NAME}.py')
    else:
        shutil.copy(f'models/{config.model.NAME}.py', os.path.join(args.experiment_path, f'model.py'))

    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)

    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    # run
    if args.test:
        test_net_vc(args, config)
    else:
        run_vc(args, config, train_writer, val_writer, test_writer)


if __name__ == '__main__':
    main()
