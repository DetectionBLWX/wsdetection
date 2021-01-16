'''
Function:
    some utils related with bboxes
Author:
    Zhenchao Jin
'''
import torch
import numpy as np


'''bbox utils'''
class BBoxUtils():
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
    '''
    Function:
        calculate overlap between two set of bboxes
    Inputs:
        bboxes1: (N, 4), the format is (x1, y1, x2, y2)
        bboxes2: (M, 4), the format is (x1, y1, x2, y2)
        mode: iou/iof/giou
        is_aligned: if True, then m and n must be equal
        eps: a value added to the denominator for numerical stability
    Return:
        Tensor: shape is (N, M)
    '''
    @staticmethod
    def overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        assert mode in ['iou', 'iof', 'giou'], f'unsupported mode {mode}...'
        assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]
        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)
        if is_aligned: assert rows == cols
        if rows * cols == 0:
            if is_aligned:
                return bboxes1.new(batch_shape + (rows, ))
            else:
                return bboxes1.new(batch_shape + (rows, cols))
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        if is_aligned:
            lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
            rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]
            if mode in ['iou', 'giou']:
                union = area1 + area2 - overlap
            else:
                union = area1
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        else:
            lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
            wh = (rb - lt).clamp(min=0)
            overlap = wh[..., 0] * wh[..., 1]
            if mode in ['iou', 'giou']:
                union = area1[..., None] + area2[..., None, :] - overlap
            else:
                union = area1[..., None]
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
                enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        if mode in ['iou', 'iof']: return ious
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area
        return gious