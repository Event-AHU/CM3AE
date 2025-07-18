
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from OurDataset import csv_Dataset
from pathlib import Path
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from tensorboardX import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('our', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='mae_vit_base_patch16', type=str,
        choices=['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")    #True
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', type=int, default=128,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0002, type=float, help="""Learning rate at the end of  #0.0005
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the      #1e-6
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")       #训练中断后加载之前训好的模型
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")


    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # load csv
    parser.add_argument("--csv-separator", type=str, default="\t", help="For csv-like datasets, which separator to use.")
    parser.add_argument("--csv-img-key", type=str, default="aps", help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument("--csv-event-key", type=str, default="dvs", help="For csv-like datasets, the name of the key for the event paths.")
    parser.add_argument("--csv-voxel_key", type=str, default="voxel", help="For csv-like datasets, the name of the key for the voxel paths.")
    parser.add_argument("--is_train", type=bool, default=True, help="trian true  or false ")

    # Misc
    parser.add_argument('--output_dir', default="./CM3AE/all_0425", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--pretrained_weights', default="./CM3AE/pre_model_mae_0312.pth", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=7, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--local-rank', dest='local_rank', type=int, help='node rank for distributed training')
    return parser

def train(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    device = torch.device(args.device)

    # ============ preparing data ... ============
    
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = csv_Dataset(args, "./CM3AE/predata_csv/HARDVS_train_our_all.csv", transform_train)
    print(dataset_train)

    sampler_train= torch.utils.data.DistributedSampler(dataset_train, shuffle=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(dataset_train)} images.")

    model = models.__dict__[args.arch](norm_pix_loss=args.norm_pix_loss)

    model = utils.MultiCropWrapper(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_without_ddp = model.module

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False,find_unused_parameters=True) 

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)

    loss_all = CM3AELoss().cuda()
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader_train),
    )
                  
    print(f"Loss, optimizer and schedulers ready.")
    #utils.load_pretrained_weights(model, args.pretrained_weights, checkpoint_key='model')
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            loss_all=loss_all
        )

    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting our training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(model, data_loader_train, loss_all, optimizer, device,lr_schedule, wd_schedule, epoch, fp16_scaler, args)

        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'loss_all': loss_all.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.output_dir and (epoch % args.saveckp_freq == 0 or epoch == args.epochs -1):
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, loss_scaler = loss_all)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, data_loader, loss_all, optimizer, device: torch.device, lr_schedule, wd_schedule, epoch, fp16_scaler, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    for it, (images, events, voxels) in enumerate(metric_logger.log_every(data_loader, 20, header)):

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it 
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu      
        images = images.to(device=device, non_blocking=True)   
        events = events.to(device=device, non_blocking=True)
        voxels = voxels.to(device=device, non_blocking=True)

        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            loss_rgb_mae, loss_event_mae, loss_fuse_two_rgb_mae, loss_e_r, loss_fuse_three_rgb_mae , loss_e_v = model(images, events, voxels, mask_ratio=args.mask_ratio)

            all_loss = loss_all(loss_rgb_mae, loss_event_mae, loss_fuse_two_rgb_mae, loss_e_r, loss_fuse_three_rgb_mae , loss_e_v)
            
            loss = all_loss.pop('loss')
            

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()

        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict

class CM3AELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss_rgb_mae, loss_event_mae, loss_fuse_two_rgb_mae, loss_e_r, loss_fuse_three_rgb_mae , loss_e_v ):

        loss_rgb_mae = loss_rgb_mae * 4
        loss_event_mae = loss_event_mae * 0.5
        loss_fuse_two_rgb_mae = loss_fuse_two_rgb_mae * 0.5
        loss_fuse_three_rgb_mae = loss_fuse_three_rgb_mae * 0.5
        loss_e_r = loss_e_r * 0.1
        loss_e_v = loss_e_v * 0.1

        total_loss = dict(rgb_mae = loss_rgb_mae, event_mae = loss_event_mae, re_mae = loss_fuse_two_rgb_mae, rev_mae = loss_fuse_three_rgb_mae, CL_er = loss_e_r, CL_ev = loss_e_v, loss = loss_rgb_mae + loss_event_mae + loss_fuse_two_rgb_mae  + loss_e_r + loss_fuse_three_rgb_mae + loss_e_v)
       
        return total_loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CM3AE', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
