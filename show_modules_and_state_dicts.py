import sys
import argparse
import os
import struct
import torch
import os
import cv2
import torch
import codecs
import zipfile
import shutil
import argparse
import sys
sys.path.append('datasets/DOTA_devkit')

from tqdm import tqdm
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image
from utils.map import eval_mAP


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('--dataset_yaml', type=str, default='./datasets/dota_dataset.yaml', help='hyper-parameter path')
    args = parser.parse_args()
    return args.weights


pt_file = parse_args()

# Initialize
device = torch.device("cpu")
with open(arg.dataset_yaml, mode='r', encoding='utf-8', errors='ignore') as f:
    hyps = yaml.load(f.read())  # load hyps dict
    f.close()
print("☯ hyps: {}".format(hyps))
model = RetinaNet(backbone="res50", hyps=hyps)
chkpt = torch.load(pt_file)
# load model
if 'model' in chkpt.keys():
    model.load_state_dict(chkpt['model'].state_dict())

model.to(device).eval()


""" Modules """
f =  open("sai_show_modules.txt","w")
for k, v in model.named_modules():
    f.write("###{}\n".format(k))
    f.write("{}\n".format(v))
f.close()

""" state_dicts """
f =  open("sai_show_state_dicts.txt","w")
for k, v in model.state_dict().items():
    f.write("###{}\n".format(k))
    if k == "model.24.anchor_grid" or k == "model.24.anchors":
        f.write("{}\n".format(v))
f.close()



"""
CAUTION: 直接保存会出现TypeError: cannot pickle 'Stream' object
        目前只能保存ckpt
        参考trt.py来修改一个[SAI-DEBUG]

"""