import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from .HelperFunction import convert_bbox_type
import function.Transforms as Transforms
import config as cfg

class SVHN_Dataset(Dataset):
    def __init__(self, root='dataset', mode='Train', img_size=416):
        self.root = root
        self.mode = mode
        
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = cfg.DATA["NUM"]
        self.img_size = img_size

        if mode in ['Train', 'Kmeans']:
            self.ImgFolder = os.path.join(root, 'Training')
            Train_GT = os.path.join(root, 'DataInfo', 'Train_GT.json')
            self.Dataset = self._parse_train(Train_GT)
            
        elif mode == 'Valid':
            self.ImgFolder = os.path.join(root, 'Training')
            Valid_GT = os.path.join(root, 'DataInfo', 'Valid_GT.json')
            self.Dataset = self._parse_train(Valid_GT)
            
        else:
            self.ImgFolder = os.path.join(root, 'Testing')
            self.Dataset = self._parse_test(self.ImgFolder)

    def __len__(self):
        return len(self.Dataset)

    def __getitem__(self, index):
        item = self.Dataset[index]
        img_path = os.path.join(self.ImgFolder, item[0])
        
        img = read_image(os.path.join(self.ImgFolder, item[0]))
        
        if self.mode in ['Valid', 'Test']:
            return img_path, img

        boxes = torch.tensor(list(item[1:]))
        img, boxes = Transforms.RandomHorizontalFlip()(img, boxes)
        img, boxes = Transforms.Resize((self.img_size, self.img_size))(img, boxes)
        img, boxes = Transforms.Divide255()(img, boxes)
        
        slbl, mlbl, llbl, sbox, mbox, lbox= self._create_label(boxes)
        
        label_sbbox = torch.from_numpy(slbl).float()
        label_mbbox = torch.from_numpy(mlbl).float()
        label_lbbox = torch.from_numpy(llbl).float()
        sbboxes = torch.from_numpy(sbox).float()
        mbboxes = torch.from_numpy(mbox).float()
        lbboxes = torch.from_numpy(lbox).float()

        if self.mode == 'Kmeans':
            return img, boxes

        return img, label_sbbox, label_mbbox, label_lbbox,\
               sbboxes, mbboxes, lbboxes

    def _create_label(self, bboxes):
        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = []
        for i in range(3):
            label.append(np.zeros(( int(train_output_size[i]),
                                    int(train_output_size[i]),
                                    anchors_per_scale,
                                    5+self.num_classes )))

        bboxes_xywh = [np.zeros((50, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot = Transforms.LabelSmooth()(one_hot, self.num_classes)

            # LTRB to XYWH
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            # scaling to the grid (by dividing the stride)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] /\
                               strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2].astype(np.int32)) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = iou_xywh(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                # iou_mask(1, 3) is the iou value that
                # supposed the (predefined) anchor centered at that grid
                # and the iou of anchor and GT > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # if iou large enough, assign anchor
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0 # conf. score
                    label[i][yind, xind, iou_mask, 5:] = one_hot

                    bbox_ind = int(bbox_count[i] % 50)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = one_hot

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    def _parse_train(self, path):
        '''
        return list of list, e.g.
        [ ['1.png', [L,T,R,B,cls], [L,T,R,B,cls]] ,...]
        '''
        with open(path, 'r') as f:
            json_data = json.load(f)
        Dataset = {}
        for d in json_data:
            img_id = str(d['image_id']) + '.png'
            cls_id = int(d['category_id'])
            bbox = convert_bbox_type(d['bbox'], 'LTWH', 'LTRB')
            if img_id not in Dataset:
                Dataset[img_id] = [img_id]
            Dataset[img_id].append([*bbox, cls_id])
        return list(Dataset.values())

    def _parse_test(self, path):
        List = os.listdir(path)
        return [[name] for name in List] # in order to match training pattern

def iou_xywh(A, B):
    '''
        Calculate the iou between A and B
        A and B are (N, 4) ndarray, indicating (center_x, center_y, W, H)
    '''
    A, B = np.array(A), np.array(B)
    areaA = A[..., 2] * A[..., 3]
    areaB = B[..., 2] * B[..., 3]

    # convert them to LTRB format
    A = np.concatenate([A[..., :2] - A[..., 2:] * 0.5,
                        A[..., :2] + A[..., 2:] * 0.5], axis=-1)
    B = np.concatenate([B[..., :2] - B[..., 2:] * 0.5,
                        B[..., :2] + B[..., 2:] * 0.5], axis=-1)

    # intersection
    inter_LT = np.maximum(A[..., :2], B[..., :2])
    inter_RB = np.minimum(A[..., 2:], B[..., 2:])
    inter = np.maximum(inter_RB - inter_LT, 0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    union_area = areaA + areaB - inter_area
    return 1.0 * inter_area / union_area
    
