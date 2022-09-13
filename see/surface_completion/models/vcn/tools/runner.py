import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
from utils.bbox_utils import get_bbox_from_keypoints
import time
import numpy as np
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import setproctitle
import pickle
from utils.transform import vc_to_cn, cn_to_vc, normalize_scale, restore_scale
from utils.sampling import *

############## Experimental: VC functions ############## 

def run_vc(args, config, train_writer=None, test_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, val_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                                                    builder.dataset_builder(args, config.dataset.val), \
                                                                                    builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)

    # print(base_model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config, steps_per_epoch=len(train_dataloader), epochs=config.max_epoch)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in tqdm(range(start_epoch, config.max_epoch), total=(config.max_epoch-start_epoch), colour='green', desc='Epoch'):
        setproctitle.setproctitle(f'{args.exp_name}: {epoch}/{config.max_epoch}')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(config.model.losses)

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Iter', colour='yellow'):            

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS            
            num_iter += 1
            label = data[2]

            in_dict = {}
            if isinstance(data[0], list):
                num_inputs = len(data[0])
                for i in range(num_inputs):
                    in_dict[f'input_{i}'] = data[0][i].cuda()
                    in_dict[f'complete_{i}'] = data[1][i].cuda()
                    in_dict[f'gt_boxes_{i}'] = label[i]['gt_boxes'].cuda()
                    in_dict[f'num_pts_{i}'] = label[i]['num_pts']
                    # in_dict[f'areas_{i}'] = torch.from_numpy(label[i]['areas']).cuda()
            else:
                in_dict['input'] = data[0].cuda()
                in_dict['gt_boxes'] = label['gt_boxes'].cuda() 
                in_dict['num_pts'] = label['num_pts']
                # in_dict['areas'] = torch.from_numpy(label['areas']).cuda()
                in_dict['complete'] = data[1].cuda()
                # in_dict['cn_areas'] = torch.from_numpy(label['cn_areas']).cuda()

            in_dict['training'] = True
            if not config.dataset.val.others.fixed_input:
                in_dict['npts'] = label['num_pts']                

            ret_dict = base_model(in_dict)                        
            loss_dict = base_model.module.get_loss(ret_dict, in_dict)            

            loss = torch.cat([loss_dict[key].unsqueeze(0)*config.model.loss_weights[i] for i, key in enumerate(config.model.losses)]).sum()
            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()                
                base_model.zero_grad()

            if config.scheduler.type == 'OneCycle':
                # One cycle LR scheduler requires step after each iteration
                scheduler.step()

            losses.update([loss_dict[key]*config.model.loss_weights[i] for i, key in enumerate(config.model.losses)])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                for k,v in loss_dict.items():
                    train_writer.add_scalar(f'Train/Loss/Batch/{k}', v.item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 200 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) \nLosses  = %s Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(), ['%s' % l for l in config.model.losses],
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        
        if not config.scheduler.type == 'OneCycle':    
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)

        epoch_end_time = time.time()

        if train_writer is not None:
            last_lr = scheduler.get_last_lr()
            train_writer.add_scalar(f'Train/Epoch/lr', last_lr, epoch)
            for i, (k,v) in enumerate(loss_dict.items()):
                train_writer.add_scalar(f'Train/Loss/Epoch/{k}', losses.avg(i), epoch)
            
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) \nLosses  = %s Loss Values = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%s' % l for l in config.model.losses], ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate_vc(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save checkpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        if epoch % args.test_freq == 0 and epoch != 0:
            # Test on actual lidar data
            config.TRAINING_TEST = True
            test_vc(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger, test_writer=test_writer, epoch=epoch)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     

    if train_writer is not None:
        train_writer.close()    
    if val_writer is not None:
        val_writer.close()
    if test_writer is not None:
        test_writer.close()

def validate_vc(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(config.model.losses)
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(val_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(val_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = config.dataset.val._base_.NAME
            label = data[2]
            
            in_dict = {}
            if isinstance(data[0], list):
                num_inputs = len(data[0])
                for i in range(num_inputs):
                    in_dict[f'input_{i}'] = data[0][i].cuda()
                    in_dict[f'complete_{i}'] = data[1][i].cuda()
                    in_dict[f'gt_boxes_{i}'] = label[i]['gt_boxes'].cuda()
                    in_dict[f'num_pts_{i}'] = label[i]['num_pts']
            else:
                in_dict['input'] = data[0].cuda()
                in_dict['gt_boxes'] = label['gt_boxes'].cuda() 
                in_dict['num_pts'] = label['num_pts']
                in_dict['complete'] = data[1].cuda()

            in_dict['training'] = False

            if not config.dataset.val.others.fixed_input:
                in_dict['npts'] = label['num_pts']                

            ret_dict = base_model(in_dict)        

            # For atlas-dsr we can't do loss on validation cause we have no graph    
            # loss_dict = base_model.module.get_loss(ret_dict, in_dict)
            # test_losses.update([v.item()*config.model.loss_weights[i] for i, (k,v) in enumerate(loss_dict.items())])

            in_dict['num_pts'] = np.array(in_dict['num_pts'])            
            
            if 'coarse' not in ret_dict.keys():
                ret_dict['coarse'] = cn_to_vc(ret_dict['coarse_cn'], in_dict['gt_boxes'])                

            ret_dict['pred_box'] = get_bbox_from_keypoints(ret_dict['coarse'], in_dict['gt_boxes'])

            _metrics = Metrics.get(ret_dict, in_dict)       
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 200 == 0:                
                if dataset_name == 'VC':
                    img_id = label['ids'][0]
                else:
                    img_id = idx

                input_pc = in_dict['input'][0].detach().cpu().numpy()
                input_img = misc.get_ptcloud_img(input_pc, centered=True)
                val_writer.add_image(f'{img_id}/Input-Centered', input_img, epoch, dataformats='HWC')

                sparse = ret_dict['coarse'][0].cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse, centered=True)
                val_writer.add_image(f'{img_id}/Keypoints-Centered', sparse_img, epoch, dataformats='HWC')

                # gt_ptcloud = in_dict['complete'].squeeze().cpu().numpy()
                # gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud, centered=True)
                # val_writer.add_image(f'{img_id}/GT-Centered', gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % 20 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s\nLosses = %s Losses Values = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%s' % l for l in config.model.losses], ['%.4f' % l for l in test_losses.val()]), logger=logger)
        
        for k,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f\t' % value
    print_log(msg, logger=logger)

    metric_scores = zip(test_metrics.items, category_metrics[taxonomy_id].avg())
    metric_dict = dict(metric_scores)
    print('-- Summary --')
    for k,v in metric_dict.items():
        print(f'{k}: {v:0.3f}')

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Val/Loss/Epoch/Coarse', test_losses.avg(0), epoch)        
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Val/Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


def test_net_vc(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test_vc(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test_vc(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None, test_writer=None, epoch=None):

    
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(config.model.losses)
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    time_total = []

    # Take the median mainly for rotation errors (TODO: incorporate this properly)
    MedianMeter = {}
    for name in Metrics.names():
        if 'Rotation' in name:
            MedianMeter[name] = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test', colour='yellow'):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = config.dataset.test._base_.NAME
            label = data[2]
            t0 = time.time()

            in_dict = {}
            if isinstance(data[0], list):
                num_inputs = len(data[0])
                for i in range(num_inputs):
                    in_dict[f'input_{i}'] = data[0][i].cuda()
                    in_dict[f'complete_{i}'] = data[1][i].cuda()
                    in_dict[f'gt_boxes_{i}'] = label[i]['gt_boxes'].cuda()
                    in_dict[f'num_pts_{i}'] = label[i]['num_pts']
            else:
                in_dict['input'] = data[0].cuda()
                in_dict['gt_boxes'] = label['gt_boxes'].cuda() 
                in_dict['num_pts'] = label['num_pts']
                in_dict['complete'] = data[1].cuda()

            in_dict['training'] = False

            if not config.dataset.val.others.fixed_input:
                in_dict['npts'] = label['num_pts']                

            ret_dict = base_model(in_dict)
            time_taken = time.time() - t0            
            time_total.append(time_taken)            

            # if config.get('TRAINING_TEST', False):
            #     loss_dict = base_model.module.get_loss(ret_dict, in_dict)
            # else:
            #     loss_dict = base_model.get_loss(ret_dict, in_dict)

            # test_losses.update([v.item()*config.model.loss_weights[i] for i, (k,v) in enumerate(loss_dict.items())])

            in_dict['num_pts'] = np.array(in_dict['num_pts'])
            
            if 'coarse' not in ret_dict.keys():
                ret_dict['coarse'] = cn_to_vc(ret_dict['coarse_cn'], in_dict['gt_boxes'])                     
            coarse_points = ret_dict['coarse']   
            ret_dict['pred_box'] = get_bbox_from_keypoints(coarse_points, in_dict['gt_boxes'])

            if 'dataset' in label.keys():
                if label['dataset'][0] in ['waymo', 'nuscenes']:
                    ret_dict['pred_box'][:,4] += 0.25

            _metrics = Metrics.get(ret_dict, in_dict)
            test_metrics.update(_metrics)
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # Median error
            for idx, name in enumerate(Metrics.names()):
                if 'Rotation' in name:
                    MedianMeter[name].append(_metrics[idx])

            if not config.get('TRAINING_TEST', False):
                if config.dataset.test._base_.get('EXT_MESH', False):
                    target_path = Path(config.dataset.test._base_.DATA_PATH) / 'test' / f'completed-{args.ext_mesh_tag}'
                    if not os.path.exists(target_path):
                        os.mkdir(target_path)

                    meta_dir = Path(config.dataset.test._base_.DATA_PATH) / 'test' / f'metadata-{args.ext_mesh_tag}'
                    if not os.path.exists(meta_dir):
                        os.mkdir(meta_dir)
                    
                    # Experiment with different k values
                    pred_surface = get_partial_mesh_batch( in_dict['input'], ret_dict['coarse'], k=30)
                    pred_clusters = get_largest_cluster_batch(pred_surface, eps=0.4, min_points=1)
                    for idx, cluster in enumerate(pred_clusters):
                        pred_clustered_surface = torch.unique(cluster, dim=0).cpu().numpy().squeeze()
                        object_pcd = misc.convert_to_o3dpcd(pred_clustered_surface)
                        o3d.io.write_point_cloud(str(target_path / f'{label["ids"][idx]}.pcd'), object_pcd)

                        with open(str(meta_dir / f'{label["ids"][idx]}.pkl'), 'wb') as f:
                            # For the SEE-IDEAL case, we don't use shape completions below 0.6 IOU_3D
                            metadata = {
                                        'num_pts': in_dict['num_pts'][idx].item()
                                        }  
                            pickle.dump(metadata, f)

                else:
                    vis_dir = os.path.join(args.experiment_path, 'vis_result')
                    if not os.path.exists(vis_dir):
                        os.mkdir(vis_dir)

                    for idx in range(len(label['ids'])):
                        target_path = os.path.join(vis_dir, f'{label["ids"][idx]}')
                        if not os.path.exists(target_path):
                            os.mkdir(target_path)

                        # print(f'idx: {idx}, folder: {target_path}')
                        np.save(os.path.join(target_path, 'num_pts.npy'), np.array(label['num_pts'][idx]))
                        np.save(os.path.join(target_path, 'input.npy'), in_dict['input'][idx].cpu().numpy())
                        np.save(os.path.join(target_path, 'gt.npy'), in_dict['complete'][idx].cpu().numpy())
                        np.save(os.path.join(target_path, 'gt_label.npy'), in_dict['gt_boxes'][idx].cpu().numpy())
                        np.save(os.path.join(target_path, 'coarse.npy'), ret_dict['coarse'][idx].cpu().numpy())            
                        np.save(os.path.join(target_path, 'pred_box.npy'), ret_dict['pred_box'][idx].cpu().numpy())

                        if 'reg_rot' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'pred_rot.npy'), ret_dict['reg_rot'][idx].cpu().numpy())                        
                        if 'reg_centre' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'pred_centre.npy'), ret_dict['reg_centre'][idx].cpu().numpy())
                        if 'fine' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'fine.npy'), ret_dict['fine'][idx].cpu().numpy())            
                        if 'dir_cls' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'dir_cls.npy'), ret_dict['dir_cls'][idx].cpu().numpy())
                        if 'reg_iou_3d' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'reg_iou_3d.npy'), ret_dict['reg_iou_3d'][idx].cpu().numpy())
                        if 'raw_rot_mat' in ret_dict.keys():
                            np.save(os.path.join(target_path, 'raw_rot_mat.npy'), ret_dict['raw_rot_mat'][idx].cpu().numpy())                        

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s \nLosses %s Loss Values = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%s' % l for l in config.model.losses], ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        for k,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
        print(f'Average time taken: {sum(time_total)/len(time_total):0.4f}')
     

    # Median error
    print("MEDIAN ERROR")
    final_metrics = test_metrics.avg()    
    for idx, name in enumerate(Metrics.names()):        
        if 'Rotation' in name:
            vals = np.array(MedianMeter[name])
            vals = vals[vals > 0.0]
            final_metrics[idx] = np.median(vals)

    final_category_metrics = dict()
    for taxonomy_id in category_metrics:
        cat_metrics = category_metrics[taxonomy_id].avg()
        for idx, name in enumerate(Metrics.names()):        
            if 'Rotation' in name:
                vals = np.array(MedianMeter[name])
                vals = vals[vals > 0.0]
                cat_metrics[idx] = np.median(vals)
        final_category_metrics[taxonomy_id] = cat_metrics


    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)
    
    for taxonomy_id in final_category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        print('final_category_metrics[taxonomy_id] = ', final_category_metrics[taxonomy_id])
        for value in final_category_metrics[taxonomy_id]:
            msg += '%.3f\t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in final_metrics:
        msg += '%.3f\t' % value
    print_log(msg, logger=logger)



    metric_scores = zip(test_metrics.items, final_metrics)
    metric_dict = dict(metric_scores)
    print('-- Summary --')
    for key, item in metric_dict.items():
        print(f'{key}: {item:0.3f}')
    
    if config.get('TRAINING_TEST', False):
        # Add testing results to TensorBoard
        if test_writer is not None:
            test_writer.add_scalar('Test/Loss/Epoch/Coarse', test_losses.avg(0), epoch)
            for i, metric in enumerate(test_metrics.items):
                test_writer.add_scalar('Test/Metric/%s' % metric, final_metrics[i], epoch)
    return 