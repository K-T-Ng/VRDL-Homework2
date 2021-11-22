import torch.nn as nn
from torchvision.models.detection.faster_rcnn import (fasterrcnn_resnet50_fpn,
                                                      FastRCNNPredictor)


def MakeModel(num_classes=11):
    # use fasterrcnn_resnet50_fpn with pretrained in torchvision
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
