'''
Function:
    max iou assigner
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ....utils import BBoxUtils


'''define the max iou assigner'''
class MaxIoUAssigner(nn.Module):
    def __init__(self, **kwargs):
        super(MaxIoUAssigner, self).__init__()
        self.cfg = kwargs
    '''forward'''
    def forward(self, rois, targets):
        cfg = self.cfg
        # calculate ious
        gt_bboxes, gt_labels, gt_weight = targets['gt_bboxes'], targets['gt_labels'], targets.get('gt_scores', None)
        overlaps = BBoxUtils.overlaps(rois, gt_bboxes, mode='iou')
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)[0]
        # foreground and background index
        fg_idxs = max_overlaps >= cfg['fg_thresh']
        bg_idxs = max_overlaps < cfg['bg_thresh']
        # assign targets to rois, 0 means background, -1 means ignore
        assigned_gt_labels = torch.zeros(rois.size(0)).fill_(-1)
        assigned_gt_labels[fg_idxs] = gt_labels[gt_assignment, 0][fg_idxs]
        assigned_gt_labels[bg_idxs] = 0
        if gt_weight is not None:
            assigned_weight = gt_weight[gt_assignment, 0]
        else:
            assigned_weight = None
        # return assign result
        return {
            'rois': rois,
            'gt_labels': assigned_gt_labels.type_as(targets['gt_bboxes']),
            'weight': assigned_weight.type_as(targets['gt_bboxes']),
        }