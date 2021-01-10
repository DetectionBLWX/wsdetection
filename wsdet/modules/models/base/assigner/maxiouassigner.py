'''
Function:
    max iou assigner
Author:
    Zhenchao Jin
'''
import torch
from ....utils import BBoxUtils


'''define the max iou assigner'''
class MaxIoUAssigner(object):
    def __init__(self, **kwargs):
        super(MaxIoUAssigner, self).__init__()
        self.cfg = kwargs
    '''forward'''
    def forward(self, rois, targets):
        cfg = self.cfg
        # calculate ious
        gt_bboxes, gt_labels, gt_weight = targets['gt_bboxes'], targets['gt_labels'], targets.get('gt_weight', None)
        overlaps = BBoxUtils.calcIoUs(rois, gt_bboxes)
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        # foreground and background index
        fg_idxs = torch.where(max_overlaps > cfg['fg_thresh'])[0]
        bg_idxs = torch.where(max_overlaps < cfg['bg_thresh'])[0]
        # assign targets to rois, 0 means background, -1 means ignore
        assigned_gt_labels = torch.zeros(rois.size(0)).fill_(-1)
        assigned_gt_labels[fg_idxs] = gt_labels[gt_assignment, 0][fg_idxs]
        assigned_gt_labels[bg_idxs] = 0
        if gt_weight is not None:
            assigned_cls_weight = gt_weight[gt_assignment, 0]
        else:
            assigned_cls_weight = None
        # return assign result
        return {
            'rois': rois,
            'gt_labels': assigned_gt_labels,
            'cls_weight': assigned_cls_weight,
        }