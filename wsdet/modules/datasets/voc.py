'''
Function:
    load voc dataset
Author:
    Zhenchao Jin
'''
import os
import pandas as pd
import xml.etree.ElementTree as ET
from .base import BaseDataset
from chainercv.evaluations import eval_detection_voc


'''voc dataset'''
class VOCDataset(BaseDataset):
    num_classes = 21
    classnames = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        super(VOCDataset, self).__init__(mode, logger_handle, dataset_cfg, **kwargs)
        assert dataset_cfg['type'] in ['VOC07', 'VOC12', 'VOC0712']
        df = pd.read_csv(os.path.join(self.set_dir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = [str(_id) for _id in df['imageids'].values()]
        self.imagepaths = [os.path.join(dataset_cfg['rootdir'], 'JPEGImages', f'{imageid}.jpg') for imageid in self.imageids]
        self.annpaths = [os.path.join(dataset_cfg['rootdir'], 'Annotations', f'{imageid}.xml') for imageid in self.imageids]
        self.proposals_dict = self.loadproposals(dataset_cfg['proposal_cfg'])
        if self.proposals_dict is not None: assert len(self.proposals_dict) == len(self.imageids)
    '''pull item'''
    def __getitem__(self, index):
        # read image
        image = cv2.imread(self.imagepaths[index])
        # read annotation
        annotation = ET.parse(self.annpaths[index])
        width = int(annotation.find('size').find('width').text)
        height = int(annotation.find('size').find('height').text)
        gt_boxes, gt_labels = [], []
        for obj in annotation.findall('object'):
            if obj.find('difficult').text != '1':
                bndbox = obj.find('bndbox')
                gt_box = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                gt_box[0] -= 1
                gt_box[1] -= 1
                if (gt_box[2] <= gt_box[0]) or (gt_box[3] <= gt_box[1]): continue
                gt_boxes.append(gt_box)
                gt_labels.append(self.classnames.index(obj.find('name').text))
        gt_boxes, gt_labels = np.stack(gt_boxes).astype(np.float32), np.stack(gt_labels).astype(np.int32)
        # read proposals
        proposals = torch.zeros(1, 4)
        if hasattr(self, 'proposals_dict') and self.proposals_dict is not None:
            proposals = self.proposals_dict[self.imageids[index]]
        # synctransform
        sample = {
            'imageid': self.imageids[index],
            'image': image,
            'proposals': proposals,
            'image_shape': (height, width),
        }
        if self.mode == 'TRAIN':
            sample.update({
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels,
            })
        sample = self.synctransform(sample, transform_type='without_totensor_normalize_pad')
        if self.mode == 'TEST':
            sample.update({
                'gt_boxes', gt_boxes,
                'gt_labels': gt_labels,
            })
        sample = self.synctransform(sample, transform_type='only_totensor_normalize_pad')
        # return sample
        return sample
    '''evaluate'''
    def evaluate(self, predictions, targets, metric_list=['ap', 'map'], eval_cfg=None):
        result = eval_detection_voc(
            pred_bboxes=predictions['pred_bboxes'],
            pred_labels=predictions['pred_labels'],
            pred_scores=predictions['pred_scores'],
            gt_bboxes=targets['gt_bboxes'],
            gt_labels=targets['gt_labels'],
            iou_thresh=eval_cfg['iou_thresh'],
            use_07_metric=eval_cfg['use_07_metric'],
        )
        result_selected = {}
        for metric in metric_list:
            result_selected[metric] = result[metric]
        if 'ap' in result_selected:
            ap_list = result_selected['ap']
            ap_dict = {}
            for idx, item in enumerate(ap_list):
                ap_dict[self.classnames[idx]] = item
            result_selected['ap'] = ap_dict
        return result_selected