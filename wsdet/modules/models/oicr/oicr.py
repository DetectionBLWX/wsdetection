'''
Function:
    Multiple Instance Detection Network with Online Instance Classifier Refinement
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ...backbones import *
from ..base import BaseModel
import torch.nn.functional as F


'''OICR'''
class OICR(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(OICR, self).__init__(cfg, **kwargs)
    '''forward'''
    def forward(self, x, proposals, targets=None, losses_cfg=None):
        batch_size = x.size(0)
        assert batch_size == 1, 'only support batch size is 1'
        rois = torch.zeros(batch_size, proposals.size(1), 5).type_as(proposals)
        for idx in range(batch_size):
            rois[idx, :, 0] = idx
            rois[idx, :, 1:] = proposals[idx]
        num_rois_per_image = rois.size(1)
        # extract base feats
        base_feats = self.backbone_net_stage1(x)
        # extract the feats of rois
        rois_feats = self.roi_extractor(base_feats, rois.view(-1, 5), )
        if self.cfg['backbone']['series'] in ['vgg']:
            rois_feats = rois_feats.view(rois_feats.size(0), -1)
        rois_feats = self.backbone_net_stage2(rois_feats)
        # multiple instance detection head
        rois_scores = self.mid_head_rois(rois_feats).view(batch_size, num_rois_per_image, -1)
        rois_probs = F.softmax(rois_scores, dim=1)
        classes_scores = self.mid_head_classes(rois_feats).view(batch_size, num_rois_per_image, -1)
        classes_probs = F.softmax(classes_scores, dim=2)
        preds_mid = rois_probs * classes_probs
        # instance classifier refinement
        preds_icr_list = []
        for icr_head in self.icr_heads:
            scores = icr_head(rois_feats).view(batch_size, num_rois_per_image, -1)
            preds_icr_list.append(scores)
        # return according to the mode
        if self.mode == 'TRAIN':
            # --generate pseudo targets
            assign_results = []
            for pred_idx, preds in enumerate([preds_mid] + preds_icr_list[:-1]):
                if pred_idx > 0: preds = F.softmax(preds, dim=2)
                rois_list, gt_labels_list, weight_list = [], [], []
                for idx in range(batch_size):
                    pseudo_targets = self.generatepseudotargets(rois[idx, :, 1:], preds[idx].clone(), targets['gt_labels'][idx])
                    assign_result = self.assigner(rois[idx, :, 1:], pseudo_targets)
                    rois_list.append(assign_result['rois'].unsqueeze(0))
                    gt_labels_list.append(assign_result['gt_labels'].unsqueeze(0))
                    weight_list.append(assign_result['weight'].unsqueeze(0))
                assign_result = {
                    'rois': torch.cat(rois_list, dim=0).detach(),
                    'gt_labels': torch.cat(gt_labels_list, dim=0).detach(),
                    'weight': torch.cat(weight_list, dim=0).detach(),
                }
                assign_results.append(assign_result)
            # --calculate losses
            all_predictions, all_targets = {}, {}
            for idx, preds in enumerate([preds_mid] + preds_icr_list):
                if idx == 0:
                    preds = preds.sum(1).view(batch_size, -1)
                    preds = torch.clamp(preds, min=0.0, max=1.0)
                    all_predictions.update({'loss_mid': preds})
                    all_targets.update({'loss_mid': targets['gt_labels'].view(-1, self.cfg['num_classes'])})
                else:
                    preds = preds.view(-1, self.cfg['num_classes'] + 1)
                    all_predictions.update({f'loss_icr_stage{idx}': preds})
                    all_targets.update({f'loss_icr_stage{idx}': assign_results[idx-1]['gt_labels'].view(-1)})
                    for key in losses_cfg[f'loss_icr_stage{idx}'].keys():
                        losses_cfg[f'loss_icr_stage{idx}'][key]['opts'].update({
                            'weight': assign_results[idx-1]['weight'].view(-1, 1).expand_as(preds),
                            'ignore_index': -1,
                        })
            return self.calculatelosses(
                predictions=all_predictions,
                targets=all_targets,
                losses_cfg=losses_cfg,
            )
        return sum(preds_icr_list)
    '''build head'''
    def buildhead(self, cfg):
        # multiple instance detection head
        mid_cfg = cfg['multi_instance_det']
        self.mid_head_rois = nn.Linear(mid_cfg['in_features'], cfg['num_classes'])
        self.mid_head_classes = nn.Linear(mid_cfg['in_features'], cfg['num_classes'])
        # instance classifier refinement
        icr_cfg = cfg['instance_cls_refine']
        self.icr_heads = nn.ModuleList()
        for stage_name, stage_cfg in icr_cfg.items():
            self.icr_heads.append(nn.Linear(stage_cfg['in_features'], cfg['num_classes'] + 1))
    '''return all layers with learnable parameters'''
    def alllayers(self):
        return {
            'backbone_net_stage1': self.backbone_net_stage1,
            'backbone_net_stage2': self.backbone_net_stage2,
            'mid_head_rois': self.mid_head_rois,
            'mid_head_classes': self.mid_head_classes,
            'icr_heads': self.icr_heads,
        }
    '''
    Function:
        generate pseudo targets
    Input:
        --bboxes: the shape is (N, 4)
        --probs: the shape is (N, num_classes)
        --labels: the shape is (num_classes,), 0 means background
    '''
    def generatepseudotargets(self, bboxes, probs, labels):
        gt_bboxes, gt_labels, gt_scores = [], [], []
        assert (probs.shape[-1] == labels.shape[0] + 1) or (probs.shape[-1] == labels.shape[0])
        if probs.shape[-1] == labels.shape[0] + 1:
            probs = probs[..., 1:]
        for idx in range(labels.shape[0]):
            if labels[idx] > 0:
                prob = probs[:, idx]
                max_index = torch.argmax(prob)
                gt_bboxes.append(bboxes[max_index, :].reshape(1, -1))
                gt_labels.append((idx + 1) * torch.ones((1, 1)))
                gt_scores.append(prob[max_index].reshape(1, -1))
                probs[max_index, :] = 0
        return {
            'gt_bboxes': torch.cat(gt_bboxes, dim=0),
            'gt_labels': torch.cat(gt_labels, dim=0),
            'gt_scores': torch.cat(gt_scores, dim=0),
        }