from __future__ import print_function

import os
import argparse
import numpy as np
import time
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import RandomSampler 
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.torch_utils import torch_distributed_zero_first
import yaml

from models.model import RetinaNet
from eval import evaluate
from datasets import *
from utils.utils import *
# from torch_warmup_lr import WarmupLR

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

mixed_precision = True
try:  
    # from apex import amp
    from torch.cuda import amp
except:
    print('fail to speed up training via apex \n')
    mixed_precision = False  # not installed

DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }


def train_model(args, hyps):
    #  parse configs
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    results_file = 'result.txt'
    model_dir = args.model_dir
    if not model_dir.endswith('/'):
        model_dir += '/'
    weight =  model_dir + 'last.pt' if args.resume or args.load else args.weight
    last = model_dir + 'last.pt'
    best = model_dir + 'best.pt'
    start_epoch = 0
    best_fitness = 0 #   max f1

    """ [DIST] 初始化设置 """
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=args.rank, world_size=args.world_size)
        cuda = device.type != 'cpu'

    # creat folder
    if args.rank in [-1, 0]:
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    for f in glob.glob(results_file):  # [SAI-KEY] 此处有时会在第一次启动时报error
        os.remove(f)

    # multi-scale
    if args.multi_scale:
        scales = args.training_size + 32 * np.array([x for x in range(-1, 5)])
        # set manually
        # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
        print('Using multi-scale %g - %g' % (scales[0], scales[-1]))   
    else :
        scales = args.training_size 
############

    # dataloader
    """ [DIST] Dataset和DataLoader """
    assert args.dataset in DATASETS.keys(), 'Not supported dataset!'
    ds = DATASETS[args.dataset](dataset=args.train_data, augment=args.augment, hyps=hyps)  # DOTADataset(dataset=args.train_data, augment=args.augment)
    train_sampler = DistributedSampler(ds)  # [SAI-KEY] DDP使用DistributedSampler

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers
    collater = Collater(scales=scales, keep_ratio=True, multiple=32)
    loader = data.DataLoader(
        dataset=ds,
        sampler=train_sampler,
        batch_size= int(batch_size/args.world_size), # 每个rank中的batch, 总的batch_size为 (world_size * batch_size)
        num_workers=0,
        collate_fn=collater,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    init_seeds()
    model = RetinaNet(backbone=args.backbone, hyps=hyps)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyps['lr0'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    # scheduler = WarmupLR(scheduler, init_lr=hyps['warmup_lr'], num_warmup=hyps['warm_epoch'], warmup_strategy='cos')
    scheduler.last_epoch = start_epoch - 1


    num_gpus = torch.cuda.device_count()

    # load chkpt
    if weight.endswith('.pt'):
        chkpt = torch.load(weight)
        # load model
        if 'model' in chkpt.keys() :
            if num_gpus > 1:
                model.load_state_dict(chkpt['model'].state_dict())
            else:
                model.load_state_dict(chkpt['model'].state_dict())  # [SAI-KEY] 如果是单GPU的话，直接存储的model
        else:
            model.load_state_dict(chkpt)
        # load optimizer
        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume :
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        # load results
        if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if args.resume and 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1   

        del chkpt

    """ [DIST] DistributedDataParallel """
    if num_gpus > 1:
        logger.info('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
 

    if mixed_precision:
        scaler = amp.GradScaler(enabled=cuda)

    model_info(model, report='summary')  # 'full' or 'summary'
    results = (0, 0, 0, 0)

    for epoch in range(start_epoch, epochs):
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',  'cls', 'reg', 'total', 'targets', 'img_size'))
        """ [DIST] shuffle """
        if args.rank != -1:
            train_sampler.set_epoch(epoch)
        
        pbar = enumerate(loader)
        if args.rank in [-1, 0]:
            pbar = tqdm(pbar, total=len(loader))  # progress bar
        mloss = torch.zeros(2, device=device)
        optimizer.zero_grad()
        for i, batch in pbar:
            model.train()

            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()
            ims, gt_boxes = batch['image'], batch['boxes']
            ims = ims.float()
            if torch.cuda.is_available():
                ims, gt_boxes = ims.to(device), gt_boxes.to(device)
            if mixed_precision:
                with amp.autocast(enabled=cuda): 
                    losses = model(ims, gt_boxes, process =epoch/epochs )
                    loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
                    loss = loss_cls + loss_reg
            else:
                losses = model(ims, gt_boxes, process =epoch/epochs )
                loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
                loss = loss_cls + loss_reg   
            if not torch.isfinite(loss):
                import ipdb; ipdb.set_trace()
                print('WARNING: non-finite loss, ending training ')
                break
            if bool(loss == 0):
                continue

            # calculate gradient
            if mixed_precision:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # Limit gradient norm to a certain range, here less than 0.1 with a L2-Norm
                scaler.step(optimizer)  # Call optimizer.step() inside 
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            if args.rank in [-1, 0]:
                loss_items = torch.stack([loss_cls, loss_reg], 0).detach()
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)GPU memory usage
                s = ('%10s' * 2 + '%10.3g' * 5) % ('%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, mloss.sum(), gt_boxes.shape[1], min(ims.shape[2:]))
                pbar.set_description(s)

        # Update scheduler
        scheduler.step()
        final_epoch = epoch + 1 == epochs
        
        if args.rank in [-1, 0]:
            # eval
            if hyps['test_interval']!= -1 and epoch % hyps['test_interval'] == 0 and epoch > 0 :
                if torch.cuda.device_count() > 1:
                    results = evaluate(target_size=args.target_size,
                                       eval_dir=args.eval_dir,
                                       eval_res_dir=args.eval_res_dir,
                                       dataset=args.dataset,
                                       split=args.split,
                                       model=model.module, 
                                       hyps=hyps,
                                       conf = 0.01 if final_epoch else 0.1)    
                else:
                    results = evaluate(target_size=args.target_size,
                                    eval_dir=args.eval_dir,
                                    eval_res_dir=args.eval_res_dir,
                                    dataset=args.dataset,
                                    split=args.split,
                                    model=model,
                                    hyps=hyps,
                                    conf = 0.01 if final_epoch else 0.1) #  p, r, map, f1

            
            # Write result log
            with open(results_file, 'a') as f:
                f.write(s + '%10.3g' * 4 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            ##   Checkpoint
            if args.dataset in ['IC15', ['IC13']]:
                fitness = results[-1]   # Update best f1
            else :
                fitness = results[-2]   # Update best mAP
            if fitness > best_fitness:
                best_fitness = fitness

            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        # 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                        'model': model.module if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) else model,
                        'optimizer': None if final_epoch else optimizer.state_dict()}
            

            # Save last checkpoint
            torch.save(chkpt, last)
            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best) 

            if (epoch % hyps['save_interval'] == 0  and epoch > 100) or final_epoch:
                if torch.cuda.device_count() > 1:
                    torch.save(chkpt, './weights/deploy%g.pt'% epoch)
                else:
                    torch.save(chkpt, './weights/deploy%g.pt'% epoch)

    # end training
    """ [DIST] Clean distributed consumption """
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a detector')
    # # Distributed
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--rank", default=os.getenv('RANK', -1), type=int)
    parser.add_argument("--world_size", default=os.getenv('WORLD_SIZE', 1), type=int)
    parser.add_argument("--master_addr", default=os.getenv('MASTER_ADDR', ""), type=str)
    parser.add_argument("--master_port", default=os.getenv('MASTER_PORT', 0), type=int)
    # config
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--dataset_yaml', type=str, default='./datasets/dota_dataset.yaml', help='hyper-parameter path')
    # network
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')   # 
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--model_dir', type=str, default='/2T/001_AI/1005_RotateRetinaNet/003_Models/IRInsu_20230222')   # [SAI-KEY]     

    # DOTA
    parser.add_argument('--dataset', type=str, default='DOTA')  # 再次也特指一种数据类型
    parser.add_argument('--split',   type=bool, default=False)  
    parser.add_argument('--train_data', type=str, default='/2T/001_AI/1005_RotateRetinaNet/002_Datasets/IRInsu_20230222/trainval.txt')
    parser.add_argument('--eval_dir', type=str, default='/2T/001_AI/1005_RotateRetinaNet/002_Datasets/IRInsu_20230222/val')
    parser.add_argument('--eval_res_dir', type=str, default='/2T/001_AI/1005_RotateRetinaNet/004_Evals/002_Dst')

    parser.add_argument('--training_size', type=int, default=640)
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--load', action='store_true', help='load training from last.pt')
    parser.add_argument('--augment', action='store_true', help='data augment')
    parser.add_argument('--target_size', type=int, default=[640])   

    arg = parser.parse_args()
    print("☯ ", arg)

    # [SAI-KEY] Dataset hyperparameters
    with open(arg.dataset_yaml, mode='r', encoding='utf-8', errors='ignore') as f:
        hyps = yaml.load(f.read())  # load hyps dict
        f.close()
    print("☯ hyps: {}".format(hyps))

    train_model(arg, hyps)