'''
Function:
    builder the assigner
Author:
    Zhenchao Jin
'''
from .maxiouassigner import MaxIoUAssigner


'''builder the assigner'''
def BuildAssigner(cfg, **kwargs):
    supported_assigners = {
        'max_iou': MaxIoUAssigner,
    }
    assert cfg['type'] in supported_assigners, 'unsupport assigner type %s...' % cfg['type']
    return supported_assigners[cfg['type']](**cfg['opts'])