"""
This file contains a Trainer class which handles the training and evaluation of SOC.
"""
import math
import sys
import os
from os import path
import shutil
import random
import numpy as np
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.cuda.amp as amp
from PIL import Image
from tqdm import tqdm
import gc
from metrics import calculate_precision_at_k_and_iou_metrics, calculate_bbox_precision_at_k_and_iou_metrics
from utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR
import misc as utils
from models import build_model
from models.soc import build_postprocessors
from models.video_swin_transformer import compute_mask
import json
from collections import namedtuple
from datasets.coco.coco_eval import CocoEvaluator
from datasets.coco.refexp_eval import RefExpEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Trainer:
    def __init__(self, config, process_id, device_id, num_processes):
        self.config = config

        self.world_size = num_processes
        self.distributed = num_processes > 1
        self.process_id = process_id
        self.is_main_process = process_id == 0
        self.device = init_process_group_and_set_device(num_processes, process_id, device_id, config)

        # fix the seed for reproducibility
        seed = config.seed + config.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion, postprocessor = build_model(config)
        model.to(self.device)
        model_without_ddp = model
        if config.distributed:
            model = DDP(model, device_ids=[device_id])
            model_without_ddp = model.module
        self.model = model
        self.backbone_name = config.backbone
        self.criterion = criterion

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        self.dataset_name = config.dataset_name
        
        ###dataset setting
        if self.dataset_name == 'coco':
            dataset_names = ['refcoco', 'refcoco+', 'refcocog']
            dataset_train = []
            for idx, name in enumerate(dataset_names):
                dataset_train.append(build_dataset(image_set="train", dataset_file=name, **vars(config)))
            collator = dataset_train[0].collator
            dataset_train = torch.utils.data.ConcatDataset(dataset_train)
        else:
            dataset_train = build_dataset(image_set ='train', **vars(config))
            collator = dataset_train.collator
        if self.distributed:
            self.sampler_train = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.rank,
                                                    shuffle=True, seed=config.seed, drop_last=False)
        else:
            self.sampler_train = None
        self.data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=self.sampler_train,
                                            collate_fn=collator, num_workers=config.num_workers,
                                            pin_memory=True, shuffle=self.sampler_train is None)
        
        ##build val
        Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])
        if self.dataset_name != 'coco':
            self.dataset_names = [self.dataset_name]
        else:
            self.dataset_names = ['refcoco', 'refcoco+', 'refcocog']
        
        self.val_tuples = []
        for name in self.dataset_names:
            dataset_val = build_dataset(image_set="val", dataset_file=name, **vars(config))
            sampler_val = (
                DistributedSampler(dataset_val, shuffle=False) if self.distributed else SequentialSampler(dataset_val)
            )
            data_loader_val = DataLoader(
                dataset_val,
                config.eval_batch_size,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=dataset_val.collator,
                num_workers=config.num_workers,
            )
            base_ds = get_coco_api_from_dataset(dataset_val)
            self.val_tuples.append(Val_all(dataset_name=name, dataloader=data_loader_val, base_ds=base_ds, evaluator_list=None))


        # Optimizer, LR-Scheduler, AMP Grad Scaler:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n and "text_encoder" not in n and p.requires_grad]},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": config.lr_backbone},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
             "lr": config.text_encoder_lr},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)   
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.1, verbose=True)
        self.grad_scaler = amp.GradScaler(enabled=config.enable_amp)
        self.max_norm = config.clip_max_norm

        if self.is_main_process:
            self.output_dir_path = create_output_dir(config)
            self.checkpoint_dir_path = create_checkpoint_dir(self.output_dir_path)
            if config.wandb_mode == 'online':
                wandb.init(project='RefVOS', config=config, mode=config.wandb_mode, name='coco')
            print(config)
        else:
            self.output_dir_path = ''
        if self.distributed:
            # sync the newly created output dir among all processes:
            output_dir_sync_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(output_dir_sync_list, self.output_dir_path)
            self.output_dir_path = output_dir_sync_list[0]

        self.total_epochs = config.epochs
        self.epoch = 0
        self.iteration = 0
        self.best_mAP = 0
        self.best_loss = math.inf

    def train(self):
        print("Training started...")
        for self.epoch in tqdm(range(self.epoch, self.total_epochs), disable=not self.is_main_process):
            self.model.train()
            self.criterion.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(self.epoch)
            print_freq = 10
            if self.distributed:
                self.sampler_train.set_epoch(self.epoch)
            total_epoch_loss = 0
            loss_sums_dict = {k: 0 for k in self.criterion.weight_dict.keys()}
            for batch_dict in tqdm(self.data_loader_train, disable=not utils.is_main_process()):
                samples = batch_dict['samples'].to(self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_queries']
                # keep only the valid targets (targets of frames which are annotated). for example, in a2d-sentences
                # only the center frame in each window is annotated.
                if self.config.dataset_name == 'a2d_sentences':
                    valid_indices = []
                    new_targets = []
                    frames = len(targets)
                    batch = len(targets[0])
                    for b in range(batch):
                        for i, t in enumerate(targets):
                            if targets[i][b] is not None:
                                valid_indices.append(i + b * frames)
                                new_targets.append(targets[i][b])
                    valid_indices = torch.tensor(valid_indices).to(self.device)
                    targets = [tuple(new_targets)]
                else:
                    valid_indices = None
                with amp.autocast(enabled=self.config.enable_amp):
                    outputs = self.model(samples, valid_indices, text_queries, targets)
                    loss_dict = self.criterion(outputs, targets)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                            k in weight_dict}
                total_loss_reduced = sum(loss_dict_reduced_scaled.values()).item()
                if not math.isfinite(total_loss_reduced):
                    print("Loss is {}, stopping training".format(total_loss_reduced))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                self.grad_scaler.scale(losses).backward()
                if self.max_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)  # gradients must be unscaled before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                metric_logger.update(loss=total_loss_reduced, **loss_dict_reduced_scaled,)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                # if self.is_main_process:
                #     wandb.log({'total_iteration_loss': total_loss_reduced})
                self.iteration += 1
                total_epoch_loss += total_loss_reduced
                for k in loss_sums_dict.keys():
                    loss_sums_dict[k] += loss_dict_reduced_scaled.get(k, torch.zeros(1)).item()
            
            metric_logger.synchronize_between_processes()
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #          'epoch': self.epoch}

            # if self.dataset_name == 'a2d_sentences':
            #     self.lr_scheduler.step()
            # else:  # refer-youtube-vos
            #     self.lr_scheduler.step(total_epoch_loss)  # note that this loss is synced across all processes
            self.lr_scheduler.step()
            
            # evaluation:
            # run gc collection before starting evaluation to avoid possible OOM errors due to swin-T caching:
            self.clear_memory()
            if self.epoch >= 0:
                test_stats = self.evaluate()
                # for key, value in eval_metrics.items():
                #     log_stats[key] = value
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': self.epoch,
                }
            if self.is_main_process:
                mAP_scores = []
                for name in self.dataset_names:
                    mAP_scores.append(test_stats.get(name + "_" + 'coco_eval_masks')[0]) 
                self.save_checkpoint(sum(mAP_scores) / len(mAP_scores))
                # eval_metrics.update({'epoch': self.epoch, 'epoch_loss': total_epoch_loss})
                # eval_metrics.update(loss_sums_dict)
                if self.config.wandb_mode == 'online':
                    wandb.log(log_stats)
                with open(os.path.join(self.output_dir_path,'log.txt'), 'a')as f:
                    f.write(json.dumps(log_stats) + "\n")
                # wandb.log({'main_model_learning_rate': self.optimizer.param_groups[0]['lr']})

            # run gc collection before starting a new epoch to avoid possible OOM errors due to swinT caching :
            self.clear_memory()
            if self.distributed:
                dist.barrier()
    
    def build_evaluator_list(self, base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = ["bbox"]
        iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        # TODO: currently ont support RefExpEvaluator (memory error)
        return evaluator_list

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        test_stats = {}
        for i, item in enumerate(self.val_tuples):
            evaluator_list = self.build_evaluator_list(item.base_ds, item.dataset_name)
            postprocessor =  build_postprocessors(dataset_name=self.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            curr_test_stats = evaluate_coco(model=self.model, postprocessor=postprocessor, item = item, evaluator_list = evaluator_list, device=self.device, is_main_process=self.is_main_process, coco_path = self.config.ann_file, distributed=self.distributed)
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        # if self.is_main_process:
        #     # mAP_scores = []
        #     # for name in self.dataset_names:
        #     #     mAP_scores.append(test_stats.get(name + "_" + 'coco_eval_masks')[0]) #MAP
        #     # print(sum(mAP_scores) / len(mAP_scores))
        #     log_stats = {
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': self.epoch,
        #         }
        #     with open(os.path.join(self.output_dir_path,'log.txt'), 'a')as f:
        #         f.write(json.dumps(log_stats) + "\n")
        #     print(log_stats)
        return test_stats    
                
    def to_device(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [self.to_device(s) for s in sample]
        return sample

    def load_checkpoint(self, checkpoint_path, total_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1  # the epoch after the one saved is about to begin
        if total_epoch == None:
            self.total_epochs = checkpoint['total_epochs']
        else:
            self.total_epochs = total_epoch
        self.best_mAP = checkpoint['best_mAP']
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    def save_checkpoint(self, epoch_score):
        if not self.is_main_process:
            return
        is_best = False
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_dict = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        is_best_mAP = epoch_score > self.best_mAP
        if is_best_mAP:
            self.best_mAP = epoch_score
            is_best = True
        checkpoint_dict['best_mAP'] = self.best_mAP
        checkpoint_dict['best_loss'] = self.best_loss
        filename = self.get_checkpoint_filename()
        torch.save(checkpoint_dict, filename)
        print(f'saved checkpoint: {filename}')
        if is_best:
            best_filename = self.get_checkpoint_filename(is_best=True)
            shutil.copyfile(filename, best_filename)
        self.remove_extra_checkpoints()

    def get_checkpoint_filename(self, is_best=False):
        basename = 'best' if is_best else f'{self.epoch:02d}'
        return os.path.join(self.checkpoint_dir_path, f'{basename}.pth.tar')

    def remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self.checkpoint_dir_path))
        max_num_checkpoints = 10
        num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self.checkpoint_dir_path, filename))

    def clear_memory(self):
        if self.backbone_name == 'video-swin-t' or self.backbone_name == 'video-swin-s' or self.backbone_name == 'video-swin-b':
            compute_mask.cache_clear()  # empty cache of SwinT
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_coco(model, postprocessor, item, evaluator_list, device, is_main_process, coco_path, distributed):
    predictions = []
    for batch_dict in tqdm(item.dataloader, disable=not is_main_process):
        samples = batch_dict['samples'].to(device)
        targets = to_device(batch_dict['targets'], device)
        text_queries = batch_dict['text_queries']
        valid_indices = None
        outputs = model(samples, valid_indices, text_queries, targets)
        # outputs, targets = flatten_temporal_batch_dims(outputs, targets)
        targets = [frame_t_target for step_t in targets for frame_t_target in step_t]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        image_ids = [t['image_id'] for t in targets]
    # processed_outputs = postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
    #                                     resized_sample_sizes=[t['size'] for t in targets],
    #                                     orig_sample_sizes=[t['orig_size'] for t in targets])
        results = postprocessor['bbox'](outputs, orig_target_sizes)
        results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {image_id.item(): output for image_id, output in zip(image_ids, results)}

        for evaluator in evaluator_list:
            evaluator.update(res)
        # REC & RES predictions
        for p, image_id in zip(results, image_ids):
            for s, b, m in zip(p['scores'], p['boxes'], p['rle_masks']):
                predictions.append({'image_id': image_id.item(),
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'bbox': b.tolist(),
                                    'segmentation': m,
                                    'score': s.item()})

    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
    stats = {}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessor.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessor.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
    
    if distributed:
        gathered_pred_lists = utils.all_gather(predictions)
        predictions = [p for p_list in gathered_pred_lists for p in p_list]

    eval_metrics = {}
    if is_main_process:
        if item.dataset_name == 'refcoco':
            coco_gt = COCO(os.path.join(coco_path, 'refcoco/instances_refcoco_val.json'))
        elif item.dataset_name == 'refcoco+':
            coco_gt = COCO(os.path.join(coco_path, 'refcoco+/instances_refcoco+_val.json'))
        elif item.dataset_name == 'refcocog':
            coco_gt = COCO(os.path.join(coco_path, 'refcocog/instances_refcocog_val.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        # ap_metrics = coco_eval.stats[:6]
        # eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        # bbox 
        precision_at_k, overall_iou, mean_iou = calculate_bbox_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'bbox P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'bbox overall_iou': overall_iou, 'bbox mean_iou': mean_iou})
        # mask
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'segm P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'segm overall_iou': overall_iou, 'segm mean_iou': mean_iou})
        print(eval_metrics)
        stats.update(eval_metrics)

    return stats

def init_process_group_and_set_device(world_size, process_id, device_id, config):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    config.world_size = world_size
    config.rank = process_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device


def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample
