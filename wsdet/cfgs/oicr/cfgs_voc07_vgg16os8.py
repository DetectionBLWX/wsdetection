'''define the config file for voc07 and vgg16os8'''
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG['train'].update(
    {
        'type': 'voc',
        'rootdir': 'data/VOCdevkit/VOC2007',
        'proposal_cfg': {
            'filepath': None,
        }
    }
)
DATASET_CFG['test'].update(
    {
        'type': 'voc',
        'rootdir': 'data/VOCdevkit/VOC2007',
        'proposal_cfg': {
            'filepath': None,
        }
    }
)
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'max_epochs': 24,
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify model config
MODEL_CFG = MODEL_CFG.copy()
MODEL_CFG.update(
    {
        'num_classes': 20,
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'oicr_vgg16os8_voc_train',
        'logfilepath': 'oicr_vgg16os8_voc_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'oicr_vgg16os8_voc_test',
        'logfilepath': 'oicr_vgg16os8_voc_test/test.log',
        'resultsavepath': 'oicr_vgg16os8_voc_test/oicr_vgg16os8_voc_results.pkl'
    }
)