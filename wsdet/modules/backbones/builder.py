'''
Function:
    build the backbone
Author:
    Zhenchao Jin
'''
from .vgg import BuildVGG


'''build the backbone'''
def BuildBackbone(cfg, **kwargs):
    supported_backbones = {
        'vgg': BuildVGG,
    }
    assert cfg['series'] in supported_backbones, 'unsupport backbone type %s...' % cfg['type']
    return supported_backbones[cfg['series']](cfg['type'], **cfg)