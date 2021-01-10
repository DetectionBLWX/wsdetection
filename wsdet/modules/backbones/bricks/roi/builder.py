'''
Function:
    build roi layer
Author:
    Zhenchao Jin
'''
import torch.nn as nn
from mmcv.ops import RoIPool
from mmcv.ops import RoIAlign


'''build roi layer'''
def BuildRoILayer(cfg):
    support_extractors = {
        'roi_pool': RoIPool,
        'roi_align': RoIAlign,
    }
    return support_extractors[cfg['type']](**cfg['opts'])