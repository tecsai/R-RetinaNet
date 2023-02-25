from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self,
                 pyramid_levels=None,
                 strides=None,
                 sizes=None,
                 ratios=None,  # np.array([0.2, 0.5, 1, 2, 5])
                 scales=None,
                 rotations=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides =  strides
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.rotations = rotations
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 1) for x in self.pyramid_levels]  # [16, 32, 64, 126, 256]
        if ratios is None:
            self.ratios = np.array([1])
        if scales is None:
            self.scales = np.array([2 ** 0])
        if rotations is None:
            self.rotations = np.array([0])
        self.num_anchors = len(self.scales) * len(self.ratios) * len(self.rotations)

    def forward(self, ims):
        ims_shape = np.array(ims.shape[2:])  # feature map的h和w
        image_shapes = [(ims_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]  # FPN输出的各feature map的分辨率
                                                                                            # 输入640, 输出80*80, 40*40, 20*20, 10*10, 5*5
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 5)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):  # p=3、4、5、6、7
            anchors = generate_anchors(
                base_size=self.sizes[idx],  # [ 16,  32, 64, 128, 256]
                ratios=self.ratios,         # [0.2, 0.5,  1,   2,   5]
                scales=self.scales,         # 1
                rotations=self.rotations    # 0
            )
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (ims.size(0), 1, 1))
        all_anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.is_tensor(ims) and ims.is_cuda:
            all_anchors = all_anchors.cuda()
        return all_anchors


def generate_anchors(base_size, ratios, scales, rotations):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales) * len(rotations)  # 所生成总的anchor数
    # initialize output anchors
    anchors = np.zeros((num_anchors, 5))
    # scale base_size
    anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios) * len(rotations))).T # 假设base_size=16, 此处生成shape(5, 2)值全为16的array
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]  # shape(5, )
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales) * len(rotations)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales) * len(rotations))
    # add rotations
    anchors[:, 4] = np.tile(np.repeat(rotations, len(scales)), (1, len(ratios))).T[:, 0]
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel(),
        np.zeros(shift_x.ravel().shape)
    )).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 5))
    return all_anchors


if __name__ == '__main__':
    pass