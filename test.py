import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

from models.yolov3 import Yolov3
from function.yololoss import YoloV3Loss
from function.SVHN_Dataset import SVHN_Dataset
from function.Evaluator import Evaluator
import config as cfg

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    img.show()

if __name__ == '__main__':
    M = Yolov3()
    E = Evaluator(M)
    for files in ['29.pth']:
        weight_path = os.path.join('weights', files)
        E.evaluate(mode='Valid', model_path=weight_path, multi_scale=True)
        print(files)
        print(E.calculate_mAP(os.path.join('dataset','DataInfo','Valid_GT.json'),
                        os.path.join('dataset','DataInfo','Valid_PD.json')))
