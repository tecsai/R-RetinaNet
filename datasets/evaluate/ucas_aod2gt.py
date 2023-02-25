import os 
import cv2
import numpy as np
import shutil
import math
from tqdm import tqdm
from decimal import Decimal



def convert_ucas_gt(gt_path, eval_res_dir):
    if os.path.exists(eval_res_dir):
        shutil.rmtree(eval_res_dir)
    os.mkdir(eval_res_dir)
    with open(gt_path,'r') as f:
        files = [x.strip('\n').replace('.png','.txt').replace('AllImages','Annotations') for x in f.readlines()]
    gts = [os.path.split(x)[1] for x in files]
    dst_gt = [os.path.join(eval_res_dir, x) for x in gts]
    print('gt generating...')
    for i, filename in enumerate(tqdm(files)):
        with open(filename,'r',encoding='utf-8-sig') as f:
#             print(filename)
            content = f.read()
            objects = content.split('\n')
            objects = [x for x in objects if len(x)>0]
            for obj in objects:
                name = obj.split()[0]
                box = [ eval(x) for x in  obj.split()[1:9]]
                with open(dst_gt[i],'a') as fd:
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            name,box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]
                        ))



if __name__ == "__main__":
    gt_path = '/data-input/das_dota/UCAS_AOD/test.txt' 
    eval_res_dir = '/data-input/das_dota/datasets/evaluate/ground-truth'

    convert_ucas_gt(gt_path, eval_res_dir)

    

