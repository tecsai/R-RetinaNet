import torch
import numpy as np
import numpy.random as npr

from torchvision.transforms import Compose
from utils.utils import Rescale, Normailize, Trans2NCHW

# TODO: keep_ratio

class Collater(object):
    """"""
    def __init__(self, scales, keep_ratio=False, multiple=32):  # scales: 640, keep_ratio=True, multiple=32
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.keep_ratio = keep_ratio
        self.multiple = multiple

    def __call__(self, batch):
        random_scale_inds = npr.randint(0, high=len(self.scales))  # 固定尺度下只有一个640
        target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)  # 设置为32的倍数
        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)
        transform = Compose([Normailize(), Trans2NCHW(unsqueeze=False)])  # 归一化, HWC2CHW, ->NCHW

        images = [sample['image'] for sample in batch]  # 图像列表
        bboxes = [sample['boxes'] for sample in batch]  # bbox列表
        batch_size = len(images)  # batch_size
        
        # 获得batch中所有图像尺寸最大的width和height
        # 最大的width和height不一定来自同一张图片
        max_width, max_height = -1, -1
        for i in range(batch_size):
            im, _ = rescale(images[i])
            height, width = im.shape[0], im.shape[1]
            max_width = width if width > max_width else max_width
            max_height = height if height > max_height else max_height

        padded_ims = torch.zeros(batch_size, 3, max_height, max_width)

        num_params = bboxes[0].shape[-1]   
        max_num_boxes = max(bbox.shape[0] for bbox in bboxes)  # [SAI-KEY] 获取最大box数值
        padded_boxes = torch.ones(batch_size, max_num_boxes, num_params) * -1  # 初始值默认设置为-1
        for i in range(batch_size):
            im, bbox = images[i], bboxes[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)
            if num_params < 9:  # xywha
                bbox[:, :4] = bbox[:, :4] * im_scale
            else:   
                bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))
            padded_boxes[i, :bbox.shape[0], :] = torch.from_numpy(bbox)
        # print("padded_ims.shape: ", padded_ims.shape)
        # print("padded_boxes.shape: ", padded_boxes.shape)
        return {'image': padded_ims, 'boxes': padded_boxes}