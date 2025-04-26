import random
import math
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import warnings
import numbers
import torch
import warnings
from collections.abc import Sequence

def resize_bbox(bbox, old_size, new_size):  
    """  
    根据图像的缩放比例调整边界框的大小。  
      
    参数:  
        bbox: 原始边界框，格式为 (x1, y1, x2, y2)  
        old_size: 原始图像的大小，格式为 (width, height)  
        new_size: 新图像的大小，格式为 (width, height)  
          
    返回:  
        调整大小后的边界框，格式为 (x1, y1, x2, y2)  
    """  
    old_w, old_h = old_size  
    new_w, new_h = new_size  
      
    # 计算缩放比例  
    scale_w = new_w / float(old_w)  
    scale_h = new_h / float(old_h)  
      
    # 根据缩放比例调整边界框坐标 
    bbox[0] =  int(bbox[0] * scale_w)
    bbox[1] =  int(bbox[1] * scale_h)
    bbox[2] =  int(bbox[2] * scale_w)
    bbox[3] =  int(bbox[3] * scale_h)
      
    return bbox 

def get_params(img, scale, ratio) :
    _, height, width = F.get_dimensions(img)
    area = width * height
    
    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
    
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class RandomResizedCropWithBbox(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0), interpolation=InterpolationMode.BILINEAR):
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image, target):
        # img: PIL Image
        # boxes: List of boxes [xmin, ymin, xmax, ymax]
        _, height, width = F.get_dimensions(image)
        i, j, h, w = get_params(image, self.scale, self.ratio)
        
        size_old = width,height
        # Perform crop and resize
        image = F.resized_crop(image, i, j, h, w, self.size,self.interpolation)
        size_new = image.size

        # Adjust bounding boxes accordingly
        if target != None:
            bbox = target['boxes']
            if bbox[0] != -100:
                bbox[0] = max(min(max(0, bbox[0] - i), w), 0)
                bbox[1] = max(min(max(0, bbox[1] - j), h), 0)
                bbox[2] = max(min(max(0, bbox[2] - i), w), 0)
                bbox[3]  = max(min(max(0, bbox[3] - j), h), 0)

                bbox = resize_bbox(bbox, size_old, size_new)
                #原图resize成224*224，label对应更改
                target['boxes'] = bbox

        return image, target

class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            width,height  = image.size 
            image = F.hflip(image)
            if target != None:
                bbox = target["boxes"]
                if bbox[0] != -100:
                # bbox: xmin, ymin, xmax, ymax
                    bbox[0] = width - bbox[2]
                    bbox[2] = width - bbox[0]
                    target["boxes"] = bbox
        return image, target

class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target):

        image = F.normalize(image, mean=self.mean, std= self.std)
 
        return image, target