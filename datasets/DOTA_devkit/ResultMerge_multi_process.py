"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
import sys
import shutil
sys.path.insert(0,'..')
try: 
    import dota_utils as util
except:
    import datasets.DOTA_devkit.dota_utils as util
import polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial

## the thresh for nms when merge image
nms_thresh = 0.1

def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]  # 得分
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]  # argsort(): 从小到大, [::-1]: 调整为从大到小

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)

        order = order[inds + 1]

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep

def py_cpu_nms(dets, thresh):
    """
    Pure Python NMS baseline.
    dets: shape(n, 9), [x, y, x, y, x, y, x, y, conf]
    """
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]


    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, thresh):
    '''
    nms: 使用py_cpu_nms_poly或py_cpu_nms_poly_fast
    thresh: nms_thresh = 0.1
    '''
    nameboxnmsdict = {x: [] for x in nameboxdict} # 创建一个以文件名命名的新的字典
    for imgname in nameboxdict:
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # nameboxdict[imgname]: [[x, y, x, y, x, y, x, y, conf], [x, y, x, y, x, y, x, y, conf], ...]
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergesingle(dstpath, nms, fullname):
    """
    dstpath: 存储路径
    nms: nms处理函数
    fullname: 文件形如Task1_plane.txt
    """
    name = util.custombasename(fullname)  # 返回文件名，不带后缀，如输入[integrated_outputs]/Task1_plane.txt, 输出Task1_plane
    #print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')  # 构建新的路径，用于存储NMS结果
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()  # 每一行形如 [文件名 conf x1 y1 x2 y2 x3 y3 x4 y4]
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]  # 文件名, 形如'P0003__1__0___223.png'
            splitname = subname.split('__')  # 按照"__"划分，返回列表，形如['P0003', '1', '0___223.png']
            oriname = splitname[0]  # 原始图像名
            pattern1 = re.compile(r'__\d+___\d+')
            #print('subname:', subname)
            x_y = re.findall(pattern1, subname)  # 返回列表，输入'P0003__1__0___223.png', 输出'__0___223'
            x_y_2 = re.findall(r'\d+', x_y[0])  # 返回'__0___0'中含有的数字，如['0', '223']
            x, y = int(x_y_2[0]), int(x_y_2[1]) # x=0, y=223

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
            rate = re.findall(pattern2, subname)[0]  # rate=1

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            origpoly = poly2origpoly(poly, x, y, rate)  # 还原至原图坐标
            det = origpoly
            det.append(confidence)
            det = list(map(float, det))  # 转换成列表，类型为float, 形如[x, y, x, y, x, y, x, y, conf]
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []  # 创建以原始图像名为key的字典项
            nameboxdict[oriname].append(det)
        # 以上步骤是将以类别名命名(如Task1_plane.txt)的文件进行拆分再统计
        # 构建一个字典，字典中的key为文件名, 对应的value为列表，列表中存储[x, y, x, y, x, y, x, y, conf]
        # 如{'P0003': [[x, y, x, y, x, y, x, y, conf], [x, y, x, y, x, y, x, y, conf], ...],
        #    'P0004': [[x, y, x, y, x, y, x, y, conf], [x, y, x, y, x, y, x, y, conf], ...],
        #    ......
        #   }
        # 当前字典所存储的结果属于同一个类

        ''' 
        nms: 使用py_cpu_nms_poly或py_cpu_nms_poly_fast
        '''
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)  # 返回经过NMS处理后的新的字典, 形如nameboxdict
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:  # 遍历每一个图像名
                for det in nameboxnmsdict[imgname]:  # nameboxnmsdict[imgname]: list
                    #print('det:', det)
                    confidence = det[-1]
                    bbox = det[0:-1]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))  # 写入文件，每一行形似[图像文件名, conf, x, y, x, y, x, y, x, y]
                    #print('outline:', outline)
                    f_out.write(outline + '\n')

def mergebase_parallel(srcpath, dstpath, nms):
    """ 
    多进程并行处理 
    在dstpath路径下生成新的文件, 命名如Task1_plane.txt
    [dstpath]/Task1_plane.txt的内容每行形如[图像文件名, conf, x, y, x, y, x, y, x, y]
    """
    pool = Pool(16)  # 开启16进程加速处理
    filelist = util.RecursizeListDir(srcpath)

    mergesingle_fn = partial(mergesingle, dstpath, nms)  # 给函数mergesingle设定两个固定的输入: dstpath和nms，fullname作为动态输入，返回新的函数mergesingle_fn 
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)  # 每个进程自动从filelist取数据送到mergesingle_fn(等价于函数mergesingle)中, 取参数操作是互斥的

def mergebase(srcpath, dstpath, nms):
    """ 
    串行处理
    srcpath: 路径下文件如Task1_plane.txt, 内容存储形如[文件名 conf x1 y1 x2 y2 x3 y3 x4 y4]的信息
    dstpath: 
    nms: 处理函数
    """
    filelist = util.RecursizeListDir(srcpath)  # 列出所有文件，文件形如Task1_plane.txt
    for filename in filelist:
        mergesingle(dstpath, nms, filename)

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'
    mergebase(srcpath, dstpath, py_cpu_nms)

def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms, 
             路径下文件如Task1_plane.txt, 内容存储形如[文件名 conf x1 y1 x2 y2 x3 y3 x4 y4]的信息
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    """ 单进程处理 """
    # mergebase(srcpath, dstpath, py_cpu_nms_poly)
    """ 
    多进程处理 
    NMS处理, dstpath下生成新的文件, [dstpath]/Task1_plane.txt的内容每行形如[图像文件名, conf, x, y, x, y, x, y, x, y]
    """
    
    mergebase_parallel(srcpath, dstpath, py_cpu_nms_poly_fast)







def ResultMerge(outputs,
                integrated_outputs,
                merged_outputs,
                dota_outputs = None): 

    """ 
    outputs: 结果存储txt文件路径
    
    """
    if not os.listdir(outputs):
        raise RuntimeError('No detection results founded in {} ! '.format(outputs))    

    util.detections2Task1(outputs, integrated_outputs)  # 按照类别划分，将同类别的信息写入到以类别命名的文件追踪，如[integrated_outputs]/Task1_plane.txt
                                                        # [integrated_outputs]/Task1_plane.txt中每一行包含形如[文件名, conf, x1, y1, x2, y2, x3, y3, x4, y4]的信息
    mergebypoly(integrated_outputs, merged_outputs)  # 使用多进程，对属于同一个类的数据执行快速NMS，并将结果写到merged_outputs中
                                                     # NMS处理, merged_outputs下生成新的文件, [merged_outputs]/Task1_plane.txt的内容每行形如[图像文件名, conf, x, y, x, y, x, y, x, y]
    if dota_outputs is not None:
        if os.path.exists(dota_outputs):
            shutil.rmtree(dota_outputs)
        os.makedirs(dota_outputs)           
        util.Task2groundtruth_poly(merged_outputs, dota_outputs)

def ResultNoMerge(outputs,
                  integrated_outputs,
                  dota_outputs = None): 

    """ 
    outputs: 结果存储txt文件路径
    
    """
    if not os.listdir(outputs):
        raise RuntimeError('No detection results founded in {} ! '.format(outputs))    

    util.detections2Task1NoMerge(outputs, integrated_outputs)  # 按照类别划分，将同类别的信息写入到以类别命名的文件追踪，如[integrated_outputs]/Task1_plane.txt
                                                        # [integrated_outputs]/Task1_plane.txt中每一行包含形如[文件名, conf, x1, y1, x2, y2, x3, y3, x4, y4]的信息
    if dota_outputs is not None:
        if os.path.exists(dota_outputs):
            shutil.rmtree(dota_outputs)
        os.makedirs(dota_outputs)           
        util.Task2groundtruth_poly(integrated_outputs, dota_outputs)


if __name__ == '__main__':
    root_dir = '/data-tmp/dal/outputs'
    outputs = 'detections'                  ## path to your det_res in txt format
    integrated_outputs = 'integrated'       ## 15 txt files
    merged_outputs = 'merged'               ## 15 txt files after NMS
    dota_outputs = 'dota_outs'              ## split to each picture

    ResultMerge(root_dir,
                outputs,
                integrated_outputs,
                merged_outputs,
                dota_outputs) 



