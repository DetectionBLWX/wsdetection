'''
Function:
    base class for loading dataset
Author:
    Zhenchao Jin
'''
import torch
from .transforms import *
from scipy.io import loadmat


'''define the base dataset class'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg, **kwargs):
        assert mode in ['TRAIN', 'TEST']
        self.mode = mode
        self.logger_handle = logger_handle
        self.dataset_cfg = dataset_cfg
        self.transforms = Compose(self.constructtransforms(self.dataset_cfg['aug_opts']))
    '''pull item'''
    def __getitem__(self, index):
        raise NotImplementedError('not be implemented')
    '''length'''
    def __len__(self):
        return len(self.imageids)
    '''transform the training/testing data'''
    def synctransform(self, sample, transform_type='all'):
        assert hasattr(self, 'transforms') and self.transforms, 'undefined transforms...'
        assert transform_type in ['all', 'without_totensor_normalize_pad', 'only_totensor_normalize_pad']
        sample = self.transforms(sample, transform_type)
        return sample
    '''load the proposals, format is [x1, y1, x2, y2]'''
    def loadproposals(self, proposal_cfg):
        if proposal_cfg['filepath'] is None: return None
        data, proposals_dict = loadmat(proposal_cfg['filepath']), dict()
        imageids = [str(item[0][0]) for item in data['images']]
        for idx, proposals in enumerate(data['boxes'][0]):
            proposals = proposals.astype(np.float32)
            proposals = np.stack((proposals[:, 1], proposals[:, 0], proposals[:, 3], proposals[:, 2]), axis=1)
            proposals_dict[imageids[idx]] = proposals
        return proposals_dict
    '''construct the transforms'''
    def constructtransforms(self, aug_opts):
        # obtain the transforms
        transforms = []
        supported_transforms = {
            'Resize': Resize,
            'RandomFlip': RandomFlip,
            'PhotoMetricDistortion': PhotoMetricDistortion,
            'Normalize': Normalize,
            'ToTensor': ToTensor,
            'Padding': Padding,
        }
        for aug_opt in aug_opts:
            key, value = aug_opt
            assert key in supported_transforms, 'unsupport transform %s...' % key
            transforms.append(supported_transforms[key](**value))
        # return the transforms
        return transforms
    '''eval the resuls'''
    def evaluate(self, predictions, groundtruths):
        raise NotImplementedError('not be implemented')