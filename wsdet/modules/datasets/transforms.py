'''
Function:
    define the transforms for data augmentations
Author:
    Zhenchao Jin
'''
import cv2
import torch
import random
import numpy as np


'''resize image'''
class Resize(object):
    def __init__(self, output_size=None, output_size_list=None, keep_ratio=True, bbox_clip_border=True, interpolation='bilinear'):
        assert (self.output_size is None) or (self.output_size_list is None)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.output_size_list = output_size_list
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border
        self.interpolation = interpolation
        self.interpolation_codes = {
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'lanczos': cv2.INTER_LANCZOS4,
            'bilinear': cv2.INTER_LINEAR,
        }
    '''call'''
    def __call__(self, sample):
        for key in ['image', 'gt_bboxes', 'proposals']:
            if key in ['image']:
                image = sample[key].copy()
                output_size = self.output_size if self.output_size is not None else random.choice(self.output_size_list)
                image, scale_factor = self.resizeimage(
                    image=image, 
                    output_size=output_size, 
                    keep_ratio=self.keep_ratio, 
                    interpolation=self.interpolation_codes[self.interpolation]
                )
                sample.update({
                    key: image,
                    'scale_factor': scale_factor,
                })
            elif key in ['gt_bboxes', 'proposals']:
                assert 'scale_factor' in sample
                bboxes = sample[key].copy()
                bboxes = self.resizebboxes(
                    bboxes=bboxes, 
                    scale_factor=sample['scale_factor'], 
                    image_shape=sample['image_shape'], 
                    bbox_clip_border=self.bbox_clip_border
                )
                sample[key] = bboxes
        return sample
    '''resize bboxes'''
    def resizebboxes(self, bboxes, scale_factor, image_shape, bbox_clip_border):
        bboxes = bboxes * scale_factor
        if bbox_clip_border:
            bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, image_shape[1])
            bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, image_shape[0])
        return bboxes
    '''resize image'''
    def resizeimage(self, image, output_size, keep_ratio, interpolation):
        h_ori, w_ori = image.shape[:2]
        if keep_ratio:
            scale_factor = max(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            h, w = int(h_ori * float(scale_factor) + 0.5), int(w_ori * float(scale_factor) + 0.5)
            image = cv2.resize(image, dsize=(w, h), dst=None, interpolation=interpolation)
        else:
            image = cv2.resize(image, dsize=output_size, dst=None, interpolation=interpolation)
        h, w = image.shape[:2]
        w_scale, h_scale = w / w_ori, h / h_ori
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        return image, scale_factor


'''random flip'''
class RandomFlip(object):
    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert direction in ['horizontal', 'vertical', 'diagonal']
        self.flip_prob = flip_prob
        self.direction = direction
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.flip_prob: return sample
        for key in sample.keys():
            if key in ['image']:
                image = sample[key].copy()
                image = self.flipimage(image, self.direction)
                sample[key] = image
            elif key in ['gt_bboxes', 'proposals']:
                bboxes = sample[key].copy()
                bboxes = self.flipbboxes(bboxes, sample['image_shape'], self.direction)
                sample[key] = bboxes
        return sample
    '''flip bboxes'''
    def flipbboxes(self, bboxes, image_shape, direction):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = image_shape[:2]
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        return flipped
    '''flip image'''
    def flipimage(self, image, direction):
        if direction == 'horizontal':
            return np.flip(image, axis=1)
        elif direction == 'vertical':
            return np.flip(image, axis=0)
        else:
            return np.flip(image, axis=(0, 1))


'''apply photometric distortion to image sequentially'''
class PhotoMetricDistortion(object):
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
    '''calll'''
    def __call__(self, sample):
        # parse and assert
        image = sample['image'].copy()
        image = image.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            image += delta
        # mode == 0 means do random contrast first and mode == 1 means do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                image *= alpha
        # convert color from BGR to HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # random saturation
        if random.randint(2):
            image[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)
        # random hue
        if random.randint(2):
            image[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            image[..., 0][image[..., 0] > 360] -= 360
            image[..., 0][image[..., 0] < 0] += 360
        # convert color from HSV to BGR
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                image *= alpha
        # randomly swap channels
        if random.randint(2):
            image = image[..., random.permutation(3)]
        # update and return sample
        sample['image'] = image
        return sample


'''normalize image'''
class Normalize(object):
    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    '''call'''
    def __call__(self, sample):
        image = sample['image'].copy()
        image = image.astype(np.float32)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        if self.to_rgb: cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        cv2.subtract(image, mean, image)
        cv2.multiply(image, stdinv, image)
        sample['image'] = image
        return sample


'''convert np.array to torch.Tensor'''
class ToTensor(object):
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image']:
                sample[key] = torch.from_numpy((sample[key].transpose((2, 0, 1))).astype(np.float32))
            elif key in ['gt_bboxes', 'gt_labels', 'proposals']:
                sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample          


'''pad image'''
class Padding(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0, padding_mode='constant'):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.padding_mode = padding_mode
        self.padding_modes = {
            'edge': cv2.BORDER_REPLICATE,
            'symmetric': cv2.BORDER_REFLECT,
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT_101,
        }
    '''call'''
    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image']:
                sample[key] = self.padimage(
                    image=sample[key], 
                    size=self.size, 
                    size_divisor=self.size_divisor, 
                    pad_val=self.pad_val,
                    padding_mode=self.padding_modes[self.padding_mode]
                )
        return sample
    '''pad image'''
    def padimage(self, image, size, size_divisor, pad_val, padding_mode):
        if size is not None:
            pad_size = (0, 0, size[1] - image.shape[1], size[0] - image.shape[0])
        else:
            pad_h = int(np.ceil(image.shape[0] / size_divisor)) * size_divisor
            pad_w = int(np.ceil(image.shape[1] / size_divisor)) * size_divisor
            pad_size = (0, 0, pad_w - image.shape[1], pad_h - image.shape[0])
        image = cv2.copyMakeBorder(image, pad_size[1], pad_size[3], pad_size[0], pad_size[2], padding_mode, value=pad_val)
        return image


'''wrap the transforms'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms
    '''call'''
    def __call__(self, sample, transform_type):
        if transform_type == 'without_totensor_normalize_pad':
            for transform in self.transforms:
                if not (isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding)):
                    sample = transform(sample)
        elif transform_type == 'only_totensor_normalize_pad':
            for transform in self.transforms:
                if isinstance(transform, ToTensor) or isinstance(transform, Normalize) or isinstance(transform, Padding):
                    sample = transform(sample)
        else:
            for transform in self.transforms:
                sample = transform(sample)
        return sample