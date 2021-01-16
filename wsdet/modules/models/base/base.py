'''
Function:
    define the base model for all models
Author:
    Zhenchao Jin
'''
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from ...losses import *
from ...backbones import *
from ..base import BuildAssigner


'''base model'''
class BaseModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.mode = kwargs.get('mode')
        assert self.mode in ['TRAIN', 'TEST']
        self.norm_cfg, self.act_cfg = cfg['norm_cfg'], cfg['act_cfg']
        # build backbone_net
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        backbone_cfg.update({'norm_cfg': self.norm_cfg})
        backbone_net = BuildBackbone(backbone_cfg)
        if backbone_cfg['series'] in ['vgg']:
            self.backbone_net_stage1 = backbone_net.features
            self.backbone_net_stage2 = backbone_net.classifier
        else:
            raise ValueError('fail to parse backbone series %s' % backbone_cfg['series'])
        # build roi extractor
        self.roi_extractor = BuildRoILayer(copy.deepcopy(cfg['roi_extractor']))
        # build assigner
        self.assigner = BuildAssigner(copy.deepcopy(cfg['assigner']))
        # build head
        head_cfg = copy.deepcopy(cfg['head'])
        head_cfg.update({'num_classes': cfg['num_classes']})
        self.buildhead(head_cfg)
        # freeze norm layer
        if cfg.get('is_freeze_norm', True): self.freezenormalization()
    '''forward'''
    def forward(self, x, proposals, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''build head'''
    def buildhead(self, cfg):
        raise NotImplementedError('not to be implemented')
    '''return all layers with learnable parameters'''
    def alllayers(self):
        raise NotImplementedError('not to be implemented')
    '''freeze normalization'''
    def freezenormalization(self):
        for module in self.modules():
            if type(module) in BuildNormalization(only_get_all_supported=True):
                module.eval()
    '''calculate the losses'''
    def calculatelosses(self, predictions, targets, losses_cfg):
        assert (len(predictions) == len(targets)) and (len(targets) == len(losses_cfg))
        # calculate loss according to losses_cfg
        losses_log_dict = {}
        for loss_name, loss_cfg in losses_cfg.items():
            losses_log_dict[loss_name] = self.calculateloss(
                prediction=predictions[loss_name],
                target=targets[loss_name],
                loss_cfg=loss_cfg
            )
        loss = 0
        for key, value in losses_log_dict.items():
            value = value.mean()
            loss += value
            losses_log_dict[key] = value
        losses_log_dict.update({'total': loss})
        # convert losses_log_dict
        for key, value in losses_log_dict.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
                losses_log_dict[key] = value.item()
            else:
                losses_log_dict[key] = torch.Tensor([value.item()]).type_as(loss)
        # return the loss and losses_log_dict
        return loss, losses_log_dict
    '''calculate the loss'''
    def calculateloss(self, prediction, target, loss_cfg):
        # define the supported losses
        supported_losses = {
            'celoss': CrossEntropyLoss,
            'sigmoidfocalloss': SigmoidFocalLoss,
            'binaryceloss': BinaryCrossEntropyLoss,
        }
        # calculate the loss
        loss = 0
        for key, value in loss_cfg.items():
            assert key in supported_losses, 'unsupport loss type %s...' % key
            loss += supported_losses[key](
                prediction=prediction, 
                target=target, 
                scale_factor=value['scale_factor'],
                **value['opts']
            )
        # return the loss
        return loss