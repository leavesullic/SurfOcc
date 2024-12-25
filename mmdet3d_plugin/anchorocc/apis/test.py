# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import time
import os

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger

import mmcv
import numpy as np
from fvcore.nn import parameter_count_table
from projects.mmdet3d_plugin.utils import cm_to_ious, format_results, SSCMetrics

# utils for saving predictions 
from .utils import *

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3, pred_save=None, test_save=None):
    model.eval()
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    # evaluate ssc
    is_semkitti = hasattr(dataset, 'camera_used')
    ssc_metric = SSCMetrics().cuda()
    logger.info(parameter_count_table(model, max_depth=4))
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
            
            # for one-gpu test, print results for each batch
            ious = cm_to_ious(evaluation_semantic)
            res_table, _ = format_results(ious, return_dic=True)
            print(res_table)
        
        img_metas = data['img_metas'].data[0][0]
        # save for test submission
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            target_voxels = result['target_voxels'].clone()
            ssc_metric.update(y_pred=output_voxels,  y_true=target_voxels)
            
            # compute metrics
            scores = ssc_metric.compute()
            if is_semkitti:
                print('\n Evaluating semanticKITTI occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            else:
                print('\n Evaluating nuScenes occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            
            # save for val predictions, mostly for visualization
            if pred_save is not None:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'], 
                        target_points=result['target_points'], 
                        save_path=pred_save, 
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        for _ in range(batch_size):
            prog_bar.update()
    
    res = {
        'ssc_scores': ssc_metric.compute(),
    }
    
    if type(evaluation_semantic) is np.ndarray:
        res['evaluation_semantic'] = evaluation_semantic
    
    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, pred_save=None, test_save=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    ssc_metric = SSCMetrics().cuda()
    is_semkitti = hasattr(dataset, 'camera_used')
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    is_val_save_predictins = pred_save is not None
    if is_val_save_predictins:
        os.makedirs(pred_save, exist_ok=True)
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
        
        img_metas = data['img_metas'].data[0][0]
        # occupancy prediction
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
            if result['target_voxels'] is not None:
                target_voxels = result['target_voxels'].clone()
                ssc_results_i = ssc_metric.compute_single(
                    y_pred=output_voxels, y_true=target_voxels)
                ssc_results.append(ssc_results_i)
            
            if is_val_save_predictins:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'],
                        target_points=result['target_points'], 
                        save_path=pred_save,
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    
    if is_test_submission:
        return None
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    if type(evaluation_semantic) is np.ndarray:
        # convert to tensor for reduce_sum
        evaluation_semantic = torch.from_numpy(evaluation_semantic).cuda()
        dist.all_reduce(evaluation_semantic, op=dist.ReduceOp.SUM)
        res['evaluation_semantic'] = evaluation_semantic.cpu().numpy()
    
    return res

def nus_custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    occ_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            bs=result.shape[0]
            assert bs==1, \
                'Evaluation only supports batch_size=1 in this version'
            # encode mask results
            if isinstance(result, dict):
                print("result format error for occ3d-nus, please check")
                assert False
            else:
                batch_size = 1
                occ_results.extend([result.squeeze(dim=0).cpu().numpy().astype(np.uint8)])
                # batch_size = len(result)
                # bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        tmpdir = tmpdir + '_occ' if tmpdir is not None else None
        occ_results = collect_results_cpu(occ_results, len(dataset), tmpdir)
    else:
        # bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        # tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        # if have_mask:
        #     mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        # else:
        #     mask_results = None
        tmpdir = tmpdir + '_occ' if tmpdir is not None else None
        occ_results = collect_results_cpu(occ_results, len(dataset), tmpdir)

    return occ_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

