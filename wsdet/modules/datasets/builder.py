'''
Function:
    build dataset
Author:
    Zhenchao Jin
'''
from .voc import VOCDataset
from .coco import COCODataset


'''build dataset'''
def BuildDataset(dataset_type, **kwargs):
    supported_datasets = {
        'voc': VOCDataset,
        'coco': COCODataset,
    }
    assert dataset_type in supported_datasets, 'unsupport dataset_type %s...' % dataset_type
    return supported_datasets[dataset_type](**kwargs)