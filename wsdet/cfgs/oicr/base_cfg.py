'''base config file for OICR'''
# config for dataset
DATASET_CFG = {
    'train': {
        'type': '',
        'set': 'trainval',
        'rootdir': '',
        'aug_opts': [
            ('Resize', {'output_size_list': [(480, 480), (576, 576), (688, 688), (864, 864), (1200, 1200)], 'keep_ratio': True}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}),
            ('Padding', {'size_divisor': 32, 'pad_val': 0}),
            ('ToTensor', {}),
        ],
        'proposal_cfg': {
            'filepath': None,
        },
    },
    'test': {
        'type': '',
        'set': 'test',
        'rootdir': '',
        'aug_opts': [
            ('Resize', {'output_size': (1200, 1200), 'keep_ratio': True}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}),
            ('ToTensor', {}),
        ],
        'proposal_cfg': {
            'filepath': None,
        },
    }
}
# config for dataloader
DATALOADER_CFG = {
    'train': {
        'type': 'distributed',
        'batch_size': 4,
        'num_workers': 16,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': True
    },
    'test': {
        'type': 'distributed',
        'batch_size': 4,
        'num_workers': 16,
        'shuffle': False,
        'pin_memory': True,
        'drop_last': False
    }
}
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'sgd',
    'sgd': {
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'max_epochs': -1,
    'params_rules': {},
    'policy': {
        'type': 'step', 
        'opts': {'max_iters': None, 'num_iters': None, 'num_epochs': None, 'adjust_epochs': [17, 23], 'scale_factor': 0.1}
    },
    'adjust_period': 'epoch',
}
# config for losses
LOSSES_CFG = {
    'loss_mid': {
        'binaryceloss': {'scale_factor': 1.0, 'opts': {'ignore_index': None, 'reduction': 'mean'}}
    },
    'loss_icr_stage1': {
        'binaryceloss': {'scale_factor': 0.333, 'opts': {'ignore_index': None, 'reduction': 'mean'}}
    },
    'loss_icr_stage2': {
        'binaryceloss': {'scale_factor': 0.333, 'opts': {'ignore_index': None, 'reduction': 'mean'}}
    },
    'loss_icr_stage3': {
        'binaryceloss': {'scale_factor': 0.333, 'opts': {'ignore_index': None, 'reduction': 'mean'}}
    },
}
# config for model
MODEL_CFG = {
    'num_classes': -1,
    'norm_cfg': {'type': 'batchnorm2d', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'backbone': {
        'type': 'vgg16',
        'series': 'vgg',
        'pretrained': True,
        'outstride': 8,
    },
    'roi_extractor': {
        'type': 'roi_pool',
        'opts': {'output_size': (7, 7), 'spatial_scale': 1.0 / 8.0},
    },
    'assigner': {
        'type': 'max_iou',
        'opts': {
            'bg_thresh': 0.5,
            'fg_thresh': 0.5,
        }
    },
    'head': {
        'multi_instance_det': {'in_features': 4096},
        'instance_cls_refine': {
            'stage1': {'in_features': 4096},
            'stage2': {'in_features': 4096},
            'stage3': {'in_features': 4096},
        }
    },
}
# config for inference
INFERENCE_CFG = {
    
}
# config for common
COMMON_CFG = {
    'train': {
        'backupdir': '',
        'logfilepath': '',
        'loginterval': 50,
        'saveinterval': 1
    },
    'test': {
        'backupdir': '',
        'logfilepath': '',
        'resultsavepath': ''
    }
}