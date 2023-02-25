import os
import torch
import torch.nn as nn
import re
import glob
import numpy as np

# subname = 'P0131__1__736___0.png'
# pattern1 = re.compile(r'__\d+___\d+')
# x_y = re.findall(pattern1, subname)
# print(x_y)

# x_y_2 = re.findall(r'\d+', x_y[0])
# print(x_y_2)

# pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
# rate = re.findall(pattern2, subname)[0]
# print(rate)


# results_file = './*.py'
# for f in glob.glob(results_file):
#     # os.remove(f)
#     print(f)

# tensor_0 = torch.Tensor(3, 6)
# print(tensor_0)
# print(tensor_0[:, :-1])

scales = np.array([2 ** 0])
print(scales)
print(scales.shape)

sacales_tile = 16 * np.tile(scales, (2, 5)).T
print(sacales_tile)
print(sacales_tile.shape)

t0 = sacales_tile[:, 1] * sacales_tile[:, 0]
print(t0)
print(t0.shape)