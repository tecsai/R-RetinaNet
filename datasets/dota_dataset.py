import os
import cv2
import time
import sys
import yaml
sys.path.append("/2T/001_AI/1005_RotateRetinaNet/001_AL/Rotated_RetinaNet_v4.0")

import numpy as np
import torch
import torch.utils.data as data
import random

from utils.augment import *
from utils.utils import plot_gt
from utils.bbox import quad_2_rbox
from utils.utils import Rescale, Normailize, Trans2NCHW

from torchvision.transforms import Compose


class DOTADataset(data.Dataset):
    """ DOTA-v1.0 """
    def __init__(self,
                 dataset= None,  # txt文件，内含完整图片路径列表
                 augment = False,
                 level = 1,
                 hyps = None,
                 only_latin = True):
        self.level = level      
        self.image_set_path = dataset
        self.image_list = []
        if dataset is not None:
            self.tmp_image_list = self._load_image_names()
            self._filter_image_and_annotation()
        self.names = hyps['names']
        if self.level == 1:  # [SAI-KEY]
            self.classes = tuple(['__background__'] + self.names)
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))    
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]  
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # BGR2RGB
        
        # im_path = "/2T/001_AI/1005_RotateRetinaNet/002_Datasets/DOTA_1.0_1.5_2.0/DOTA_V1.0/val/images/P1983.png"
        # im = cv2.imread(im_path, cv2.IMREAD_COLOR)  # BGR2RGB

        roidb = self._load_annotation(im_path)
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]  # gt_inds: shape(collected_ngt,), 取值True或False => np.where返回为tuple
        bboxes = roidb['boxes'][gt_inds, :]  # shape(collected_ngt, 8) 筛选后的gt
        classes = roidb['gt_classes'][gt_inds]  # shape(collected_ngt,) 筛选后的gt
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)

        if self.augment :
            transform = Augment([ HSV(0.5, 0.5, p=0.5),
                                  HorizontalFlip(p=0.5),
                                  VerticalFlip(p=0.5),
                                  Affine(degree=20, translate=0.1, scale=0.2, p=0.5),  
                                  Noise(0.02, p=0.2),
                                  Blur(1.3, p=0.5),
                                ], box_mode = 'xyxyxyxy',)
            im, bboxes = transform(im, bboxes)

        mask = mask_valid_boxes(quad_2_rbox(bboxes, 'xywha'), return_mask=True)  # quad_2_rbox: 四坐标转旋转框标识
        bboxes = bboxes[mask]
        gt_boxes = gt_boxes[mask]
        classes = classes[mask]

        for i, bbox in enumerate(bboxes):
            gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode = 'xyxya')   
            gt_boxes[i, 5] = classes[i]


        ## test augmentation
        # plot_gt(im, gt_boxes[:,:-1], "/2T/000000.jpg", mode = 'xyxya')
        return {'image': im, 'boxes': gt_boxes, 'path': im_path}

    def _filter_image_and_annotation(self):
        for img in self.tmp_image_list:
            root_dir = img.split('/images/')[0]
            # print("root_dir: {}".format(root_dir))
            label_dir = os.path.join(root_dir, 'labelTxt')  # 标注文件存储路径
            _ , img_name = os.path.split(img)  # 提取图像文件名
            filename = os.path.join(label_dir, img_name[:-4]+'.txt')  # 构建标注文件完整路径，如/2T/DOTA_V1.0/train/labelTxt-v1.0/labelTxt/P0000.txt 
            with open(filename,'r',encoding='utf-8-sig') as f:
                valid_box = 0
                content = f.read()
                objects = content.split('\n')
                for obj in objects:  # 逐行处理
                    if len(obj) != 0 :
                        objs = obj.split(' ')
                        if len(objs) == 10:  # [SAI-KEY] 排除imagesource:GoogleEarth和gsd:0.242183449549等开头信息，仅保留标注框信息
                            valid_box += 1
                if valid_box > 0:
                    self.image_list.append(img)
                else:
                    print("Image: {} has no valid boxes".format(img))

    def _load_image_names(self):
        """
        Load the names listed in this dataset's image set file.
        """
        # print("Call _load_image_names")
        image_set_file = self.image_set_path
        """ 检查路径 """
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]
        random.shuffle(image_list)
        # for file in image_list:
        #     print(file)
        return image_list


    def _load_annotation(self, path):
        root_dir = path.split('/images/')[0]
        label_dir = os.path.join(root_dir, 'labelTxt')  # 标注文件存储路径
        _ , img_name = os.path.split(path)  # 提取图像文件名
        filename = os.path.join(label_dir, img_name[:-4]+'.txt')  # 构建标注文件完整路径，如/2T/DOTA_V1.0/train/labelTxt-v1.0/labelTxt/P0000.txt
        boxes, gt_classes = [], []
        with open(filename,'r',encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('\n')
            for obj in objects:  # 逐行处理
                if len(obj) != 0 :
                    objs = obj.split(' ')
                    if len(objs) < 10:  # [SAI-KEY] 排除imagesource:GoogleEarth和gsd:0.242183449549等开头信息，仅保留标注框信息
                        continue
                    *box, class_name, difficult = objs
                    # print("class_name: {}".format(class_name))
                    if difficult == 2:
                        continue
                    box = [ eval(x) for x in  obj.split(' ')[:8] ]
                    label = self.class_to_ind[class_name] 
                    boxes.append(box)
                    gt_classes.append(label)
        
        # print("Image: {} \nboxes.n: {}, gt_classes.n: {}".format(filename, len(boxes), len(gt_classes)))
        return {'boxes': np.array(boxes, dtype=np.int32), 'gt_classes': np.array(gt_classes)}


    def display(self,boxes, img_path):
        img = cv2.imread(img_path)
        for box in boxes:
            coors = box.reshape(4,2)
            img = cv2.polylines(img,[coors],True,(0,0,255),2)	
        cv2.imshow(img_path,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def return_class(self, id):
        id = int(id)
        return self.classes[id]

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     组织函数
    #     调用一次__getitem__得到一组 <一张图片及其label等信息>
    #     torch.utils.data.DataLoader指定batch_size后, 会多次调用__getitem__得到多组<一张图片及其label等信息>
    #     此函数用来组织多组数据
    #     [SAI-KEY] label最终输出是shape(ngt, 6), 其中ngt是一个batch中所有的GT总数
    #     """
    #     imgs, labels, path = zip(*batch)  # transposed
    #     labels_list = []
    #     max_gt_num = 0
    #     for i, l in enumerate(labels):
    #         if l.shape[0]>max_gt_num:
    #             max_gt_num = l.shape[0]
    #         for i in range(l.shape[0]):
    #             l[i, 6] = 1.0
    #     for i, l in enumerate(labels):
    #         gtl = torch.ones(max_gt_num, 7) * -1
    #         gtl[:l.shape[0], :] = l
    #         labels_list.append(gtl)

    #     return {"image": torch.cat(imgs, 0), 'boxes': torch.stack(labels_list, 0), "paths": path}
        
if __name__ == '__main__':
    with open('./datasets/dota_dataset.yaml', mode='r', encoding='utf-8', errors='ignore') as f:
        hyps = yaml.load(f.read())  # load hyps dict
        f.close()
    print("☯ hyps: {}".format(hyps))
    dataset = DOTADataset(dataset="/2T/001_AI/1005_RotateRetinaNet/002_Datasets/IRInsu_20230222/trainval.txt", augment=True, hyps=hyps)
    print(len(dataset))
    data = dataset.__getitem__(9)











