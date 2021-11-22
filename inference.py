import os

import torch

from models.yolov3 import Yolov3
from function.SVHN_Dataset import SVHN_Dataset
from function.Evaluator import Evaluator
import config as cfg

if __name__ == '__main__':
    model = Yolov3()
    weight_path = os.path.join('weights', '29.pth')
    
    E = Evaluator(model)
    E.evaluate(mode='Test', model_path=weight_path, multi_scale=True)
