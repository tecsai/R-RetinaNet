import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps


def xyxy2xywh_a(query_boxes):
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes

# cuda_overlaps
class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func = 'smooth'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss

    def forward(self, classifications, regressions, anchors, annotations, iou_thres=0.5):
        """
        classifications: shape(N, TOTAL_CELLS, NUM_CLASSES)
        regressions: shape(N, TOTAL_CELLS, NUM_REGRESSIONS)
        anchors: shape(N, TOTAL_CELLS, 5)
        annotations: shape(N, max_num_boxes, num_params), 
                     max_num_boxes: 一个batch中, 对应图像标注框最多的数值
                     num_params: 标注信息所含的数据量, 如针对rotate od, 标注形式为(x, y, w, h, theta, class), 此处num_params为5
        iou_thres: 

        NOTE: TOTAL_CELLS = H_0*W_0*NUM_ANCHORS_0 + H_0*W_0*NUM_ANCHORS_0 + H_0*W_0*NUM_ANCHORS_0
        """
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        for j in range(batch_size):
            classification = classifications[j, :, :]  # shape(TOTAL_CELLS, NUM_CLASSES)
            regression = regressions[j, :, :]  # shape(TOTAL_CELLS, NUM_REGRESSIONS)
            bbox_annotation = annotations[j, :, :]  # shape(max_num_boxes, num_params)
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]  # 先排除无效gt_box, 假设有效gt_box数量为num_valid_gt_boxes
                                                                             # shape(valid_gts, 6)
            if bbox_annotation.shape[0] == 0:  # gt_box数量是否为0
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)  # classification每个元素压缩到[1e-4, 1.0-1e-4]之间
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            """
            ious: 交并比矩阵, shape(NUM_ANCHORS, valid_gts)
            """
            ious = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),  # shape(NUM_ANCHORS, 5)
                bbox_annotation[:, :-1].cpu().numpy(),  # shape(valid_gts, 5)
                indicator.cpu().numpy(),  # shape(NUM_ANCHORS, valid_gts)
                thresh=1e-1
            )
            if not torch.is_tensor(ious):
                ious = torch.from_numpy(ious).cuda()
            
            """ 
                找出与每一个Anchor交并比最大的gt_box 
                iou_max: 最大值
                iou_argmax: 最大值索引

                iou_max, iou_argmax: shape(TOTAL_CELLS)
            """
            iou_max, iou_argmax = torch.max(ious, dim=1)
           
            positive_indices = torch.ge(iou_max, iou_thres)  # positive_indices: shape(total_cells, )

            """ 
                找出与每一个gt_box交并比最大的Anchor

                max_gt: shape(valid_gts): gt与对应的最大IOU所指向cell的IOU值
                argmax_gt: shape(valid_gts), 值为cell的索引
            """
            max_gt, argmax_gt = ious.max(0) 
            if (max_gt < iou_thres).any():  # 如果任意存在一个<iou_thres的项
                positive_indices[argmax_gt[max_gt < iou_thres]]=1
              
            # cls loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda()  # cls_targets: shape(TOTAL_CELLS, NUM_CLASSES), 值全部为-1
            cls_targets[torch.lt(iou_max, iou_thres - 0.1), :] = 0 # 小于0.4为背景
            num_positive_anchors = positive_indices.sum()  # 总的正例数
            assigned_annotations = bbox_annotation[iou_argmax, :]  # assigned_annotations: shape(TOTAL_CELLS, 6), (x, y, x, y, theta, cls)  每一个cell对应的gt的索引
            cls_targets[positive_indices, :] = 0  # 首先将正例对应的类别标记全部置0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1  # 然后按照gt信息，将正例对应的类别置1
            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(cls_targets * torch.log(classification+1e-6) + (1.0 - cls_targets) * torch.log(1.0 - classification+1e-6))
            cls_loss = focal_weight * bin_cross_entropy 
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            # reg loss
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]  # Screen out all positive cells, assump that there are NPCELLS cells was screened, here all_rois with shape(NPCELLS, 5)
                gt_boxes = assigned_annotations[positive_indices, :]  # gt boxes corresponding to every anchor cells(all_rois), shape(TOTAL_CELLS, 6), (x, y, x, y, theta, cls)
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)  # In order to calculate regression loss with predicted cells, this we encode gt box with specific method
                reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)

                if not torch.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
            else:
                reg_losses.append(torch.tensor(0).float().cuda())
        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True)
        return loss_cls, loss_reg

    
def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    if  weight is  None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1,5)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

