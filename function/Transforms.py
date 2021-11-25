import random

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Divide255(object):
    def __call__(self, img, boxes=None):
        if boxes is not None:
            return img.float() / 255, boxes
        else:
            return img.float() / 255


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes=None):
        if random.random() < self.p:
            c, h, w = img.shape
            img = TF.hflip(img)
            if boxes is not None:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        if boxes is not None:
            return img, boxes
        else:
            return img


class Resize(object):
    def __init__(self, target_shape):
        self.h, self.w = target_shape

    def __call__(self, img, boxes=None):
        c, h, w = img.shape

        resize_ratio = min(1.0 * self.h / h, 1.0 * self.w / w)
        resize_w = int(resize_ratio * w)
        resize_h = int(resize_ratio * h)

        resize_img = TF.resize(img, size=(resize_h, resize_w))
        img = torch.ones(c, self.h, self.w) * 128.0
        dw = int((self.w - resize_w) / 2)
        dh = int((self.h - resize_h) / 2)
        img[:, dh:resize_h+dh, dw:resize_w+dw] = resize_img

        if boxes is not None:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * resize_ratio + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * resize_ratio + dh
            return img, boxes
        else:
            return img


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
