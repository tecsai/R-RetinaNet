from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import codecs
import yaml
from tqdm import tqdm
import numpy as np
from utils.utils import sort_corners, is_image
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption
from utils.utils import show_dota_results
from eval import evaluate
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge, ResultNoMerge

DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }

def generate_colors(dataset):
    num_colors = {'VOC' : 20 ,
            'IC15': 1,
            'IC13': 1,
            'HRSC2016': 1,
            'DOTA':15,
            'UCAS_AOD':2,
            'NWPU_VHR':10
            }
    if num_colors[dataset] == 1:
        colors = [(0, 255, 0)]
    elif num_colors[dataset] == 2:
        colors = [(0, 255, 0), (0, 0, 255)]
    else:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_colors[dataset])]
    return colors

def dota_test(model, 
              target_size, 
              test_dir,
              test_res_dir,
              split,
              conf = 0.01,
              hyps=None):
    # 
    root_data, evaldata = os.path.split(test_dir)  # root_data: /2T/001_AI/1005_RotateRetinaNet/002_Datasets/DOTA_1.0_1.5_2.0/DOTA_V1.0
                                                    # evaldata: val
    if split==True:
        splitdata = evaldata + 'split'  # valsplit
        ims_dir = os.path.join(root_data, splitdata + '/' + 'images')  # 评估图像路径
    else:
        ims_dir = os.path.join(test_dir, 'images')  # 评估图像路径
    test_root_dir = os.path.join(test_res_dir, 'test_outputs')
    res_dir = os.path.join(test_root_dir, 'detections')          # 裁剪图像的检测结果, 以图像名字命名   
    integrated_dir = os.path.join(test_root_dir, 'integrated')   # 将裁剪图像整合后成15个txt的结果, 以类别命名
    merged_dir = os.path.join(test_root_dir, 'merged')           # 将整合后的结果NMS
    label_dir = os.path.join(test_root_dir, 'dota_out')

    test_img_path = os.path.join(test_res_dir,'images')

    
    if  os.path.exists(test_root_dir):
        shutil.rmtree(test_root_dir)
    os.makedirs(test_root_dir)
    for f in [res_dir, integrated_dir, merged_dir, label_dir, test_img_path]: 
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)
    
    ds = DOTADataset(hyps=hyps)
    # loss = torch.zeros(3)
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    # s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    # for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
    for idx, im_name in enumerate(tqdm(ims_list, desc="Test dota")):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf = conf)
        nt += len(dets)
        out_file = os.path.join(res_dir,  im_name[:im_name.rindex('.')] + '.txt')  # 创建以图像名字命名的txt,存储识别结果
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))  # xywha -> xyxyxyxy, res: 排序后的xyxyxyxy
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1],)
                    # (x, y, x, y, x, y, x, y, class, image_name, score)
                )
    '''
    Merge处理, integrated_dir下生成新的文件, [integrated_dir]/Task1_plane.txt中每一行包含形如[Split图像文件名, conf, x1, y1, x2, y2, x3, y3, x4, y4]的信息
    NMS处理, merged_dir下生成新的文件, [merged_dir]/Task1_plane.txt的内容每行形如[完整图像文件名, conf, x, y, x, y, x, y, x, y]
    integrated_dir中的"Split图像文件名"如: P0003__1__0___223.png
    merged_dir中的"Split图像文件名"如: P0003.png
    '''
    if split==True:
        ResultMerge(res_dir, integrated_dir, merged_dir, label_dir)
    else:
        ResultNoMerge(res_dir, integrated_dir, label_dir)  # 仅检测，无需合并结果

    img_path = os.path.join(test_dir, 'images')  # ./val/images
    save_imgs =  True
    if save_imgs:
        show_dota_results(img_path, label_dir, test_img_path)

    # return 0, 0, mAP, 0 


def demo(args):
    with open(args.dataset_yaml, mode='r', encoding='utf-8', errors='ignore') as f:
        hyps = yaml.load(f.read())  # load hyps dict
        f.close()
    print("☯ hyps: {}".format(hyps))
    ds = DATASETS[args.dataset](level = 1, hyps=hyps)
    model = RetinaNet(backbone=args.backbone, hyps=hyps)
    colors = generate_colors(args.dataset)
    if args.weight.endswith('.pt'):
        chkpt = torch.load(args.weight)
        # load model
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'].state_dict())
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))
    model.eval()

    t0 = time.time()
    if not args.dataset == 'DOTA':
        ims_list = [x for x in os.listdir(args.test_dir) if is_image(x)]
        for idx, im_name in enumerate(ims_list):
            s = ''
            t = time.time()
            im_path = os.path.join(args.test_dir, im_name)   
            s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)
            src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # BGR2RGB
            cls_dets = im_detect(model, im, target_sizes=args.target_size)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, thickness=2, color=colors[int(cls-1)])
                    put_label = True
                    plot_anchor = False
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.45
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                        # import ipdb;ipdb.set_trace()

                        cv2.rectangle(src, c1, c2, colors[int(cls-1)], -1)  # filled
                        cv2.putText(src, label, (c1[0], c1[1] -4), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)
                        if plot_anchor:
                            pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                            cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)
            print('%sDone. (%.3fs) %d objs' % (s, time.time() - t, len(cls_dets)))
            # save image

            out_path = os.path.join('outputs' , os.path.split(im_path)[1])
            cv2.imwrite(out_path, src)
    ## DOTA detct on large image
    else:
        dota_test(model, args.target_size, args.test_dir, args.test_res_dir, args.split, conf = 0.3, hyps=hyps)
        
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--dataset_yaml', type=str, default='./datasets/dota_dataset.yaml', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='/2T/001_AI/1005_RotateRetinaNet/003_Models/IRInsu_20230222/best.pt')

    # DOTA 
    parser.add_argument('--dataset', type=str, default='DOTA')
    parser.add_argument('--split',   type=bool, default=False)
    parser.add_argument('--test_dir', type=str, default='/2T/001_AI/1005_RotateRetinaNet/002_Datasets/IRInsu_20230222/test')
    parser.add_argument('--test_res_dir', type=str, default='/2T/001_AI/1005_RotateRetinaNet/004_Evals/002_Dst/test_outputs')

    parser.add_argument('--target_size', type=int, default=[640])
    demo(parser.parse_args())
