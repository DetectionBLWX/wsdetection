'''
Function:
    define the vgg
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .bricks import BuildActivation, BuildNormalization


'''model urls'''
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


'''vgg'''
class VGG(nn.Module):
    archs_dict = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    def __init__(self, arch, in_channels=3, norm_cfg=None, outstride=8, **kwargs):
        self.features = self.makelayers(self.archs_dict[arch], in_channels, outstride, norm_cfg=norm_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
    '''forward'''
    def forward(self, x):
        x = self.features(x)
        return x
    '''make layers for vgg'''
    def makelayers(self, cfg, in_channels, outstride, norm_cfg=None):
        layers, stride, dilation = [], 1, 1
        for v in cfg:
            if v == 'M':
                if stride < outstride:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    stride *= 2
                else:
                    dilation *= 2
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
                if norm_cfg is None:
                    layers += [conv2d, BuildActivation('relu', **{'inplace': True})]
                else:
                    layers += [conv2d, BuildNormalization(norm_cfg['type'], (v, norm_cfg['opts'])), BuildActivation('relu', **{'inplace': True})]
                in_channels = v
        return nn.Sequential(*layers)


'''build vgg'''
def BuildVGG(vgg_type, **kwargs):
    # assert whether support
    supported_vggs = {
        'vgg11': {'arch': 'A'},
        'vgg13': {'arch': 'B'},
        'vgg16': {'arch': 'D'},
        'vgg19': {'arch': 'E'},
        'vgg11_bn': {'arch': 'A'},
        'vgg13_bn': {'arch': 'B'},
        'vgg16_bn': {'arch': 'D'},
        'vgg19_bn': {'arch': 'E'},
    }
    assert vgg_type in supported_vggs, 'unsupport the vgg_type %s...' % vgg_type
    # parse args
    default_args = {
        'outstride': 8,
        'norm_cfg': None,
        'in_channels': 3,
        'pretrained': True,
        'pretrained_model_path': '',
        'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    }
    if vgg_type.endswith('bn'):
        assert 'norm_cfg' in kwargs, 'norm_cfg should be specified if use %s...' % vgg_type
    for key, value in kwargs.items():
        if key in default_args:
            default_args.update({
                key: value,
            })
    # obtain args for instanced vgg
    vgg_args = supported_vggs[vgg_type]
    vgg_args.update(default_args)
    # load weights of pretrained model
    model = VGG(**vgg_args)
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        state_dict = torch.load(default_args['pretrained_model_path'])
        model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        state_dict = model_zoo.load_url(model_urls[vgg_type])
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model