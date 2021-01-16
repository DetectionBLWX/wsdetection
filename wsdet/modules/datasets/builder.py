'''
Function:
    build dataset
Author:
    Zhenchao Jin
'''
from .voc import VOCDataset
from .coco import COCODataset


'''build dataset'''
def BuildDataset(mode, logger_handle, dataset_cfg, **kwargs):
    dataset_cfg = dataset_cfg[mode.lower()]
    supported_datasets = {
        'voc': VOCDataset,
        'coco': COCODataset,
    }
    assert dataset_cfg['type'] in supported_datasets, 'unsupport dataset type %s...' % dataset_cfg['type']
    if kwargs.get('get_basedataset', False): return BaseDataset(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    dataset = supported_datasets[dataset_cfg['type']](mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg, **kwargs)
    return dataset