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
        calculate ious
    Inputs:
        bboxes1: (N, 4), the format is (x1, y1, x2, y2)
        bboxes2: (M, 4), the format is (x1, y1, x2, y2)
    Return:
        ious: (N, M)
    '''
    @staticmethod
    def calcIoUs(bboxes1, bboxes2):
        num_bboxes1 = bboxes1.shape[0]
        num_bboxes2 = bboxes2.shape[0]
        left_top = torch.max(
            bboxes1[..., :2].unsqueeze(1).expand(num_bboxes1, num_bboxes2, 2),
            bboxes2[..., :2].unsqueeze(0).expand(num_bboxes1, num_bboxes2, 2),
        )
        right_bottom = torch.min(
            bboxes1[..., 2:].unsqueeze(1).expand(num_bboxes1, num_bboxes2, 2),
            bboxes2[..., 2:].unsqueeze(0).expand(num_bboxes1, num_bboxes2, 2)
        )
        wh = right_bottom - left_top
        wh[wh < 0] = 0
        inter = wh[..., 0] * wh[..., 1]
        bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        bboxes1_area = bboxes1_area.unsqueeze(1).expand(num_bboxes1, num_bboxes2)
        bboxes2_area = bboxes1_area.unsqueeze(0).expand(num_bboxes1, num_bboxes2)
        return inter / (bboxes1_area + bboxes2_area - inter)