import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from function.Transforms import Resize, Divide255
from function.SVHN_Dataset import SVHN_Dataset
from function.tools import xywh2xyxy, nms
from function.HelperFunction import convert_bbox_type
from function.mAP_metric import mAP_metric


class Evaluator(object):
    def __init__(self, model):
        self.classes = cfg.DATA["CLASSES"]
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape = cfg.TEST["TEST_IMG_SIZE"]

        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"

    
    def evaluate(self, mode='Valid', weight_path=None):
        assert mode in ['Valid', 'Test'], "No such mode"
        
        dataset = SVHN_Dataset(mode=mode)
        Loader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False)

        JsonList = []
        self.model.load_state_dict(torch.load(weight_path))
        self.model.to(self.device)
        self.model.eval()
        for (i, data) in enumerate(Loader):
            img_path, img = data
            
            # (1)
            # get the image id
            # img_path = ('dataset\\Training\\11203.png', )
            img_id = img_path[0].split('\\')[-1]
            img_id = int(img_id.split('.')[0])

            # (2)
            # record original image size
            org_img = img.clone()
            _, _, org_h, org_w = org_img.shape

            # (3)
            # resize image and throw into model
            img = Resize((self.val_shape, self.val_shape))(img[0])
            img = Divide255()(img).unsqueeze(0)
            img = img.to(self.device)
            
            with torch.no_grad():
                _, p_d = self.model(img)
            pred_bbox = p_d.squeeze().cpu().numpy()

            pred_coor = xywh2xyxy(pred_bbox[:, :4])
            pred_conf = pred_bbox[:, 4]
            pred_prob = pred_bbox[:, 5:]
            
            # (4)
            # now, pred_coor has scale (val_shape, val_shape)
            # scaling them to (org_h, org_w)

            # (4.1)
            # Inverse Resize (please see Transform.Resize)
            resize_ratio = min(1.0 * self.val_shape / org_w,
                               1.0 * self.val_shape / org_h)
            dw = (self.val_shape - resize_ratio * org_w) / 2
            dh = (self.val_shape - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
            
            # (4.2)
            # crop the boxes that out of image range
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                        np.minimum(pred_coor[:, 2:], [org_w-1, org_h-1])], axis=-1)
            
            # (4.3)
            # remove all invalid boxes (L > R or T > D)
            invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                         (pred_coor[:, 1] > pred_coor[:, 3]))
            pred_coor[invalid_mask] = 0

            # (4.4)
            # remove invalid range boxes
            bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
            scale_mask = np.logical_and((0 < bboxes_scale), (bboxes_scale < np.inf))

            # (4.5)
            # remove those boxes with confidence score < self.conf_thres
            classes = np.argmax(pred_prob, axis=-1)
            scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
            score_mask = scores > self.conf_thresh

            mask = np.logical_and(scale_mask, score_mask)

            coors = pred_coor[mask]
            scores = scores[mask]
            classes = classes[mask]

            bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
            bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

            # (4.6) dump into json list
            # bboxes is a (N, 6) array: [L, T, R, B, score, cls_id]
            for bbox in bboxes:
                coord = convert_bbox_type(list(bbox[:4]), 'LTRB', 'LTWH')
                score = float(bbox[4])
                cls_id = int(bbox[5])

                ddict = {'image_id':img_id,
                         'category_id': cls_id,
                         'score': score,
                         'bbox': coord}
                JsonList.append(ddict)
            if i % 100 == 0:
                print(i)

        JsonObj = json.dumps(JsonList, indent=4)
        if mode == 'Valid':
            save_path = os.path.join('dataset', 'DataInfo', 'Valid_PD.json')
        else:
            save_path = os.path.join('dataset', 'DataInfo', 'answer.json')

        with open(save_path, 'w') as fp:
            fp.write(JsonObj)
        
        return
        
    def calculate_mAP(self, GT_path, PD_path):
        def read_Json(path, mode='GT'):
            assert mode in ['GT', 'PD'], "No such choice for mode"
            
            with open(path, 'r') as fp:
                input_list = json.load(fp)

            output_list = []
            for bbox_dict in input_list:
                # convert every bbox dict to bbox list
                img_id = bbox_dict['image_id']
                cls_id = bbox_dict['category_id']
                bbox = convert_bbox_type(bbox_dict['bbox'], 'LTWH', 'LTRB')

                if mode == 'GT':
                    output_list.append([img_id, cls_id, *bbox])
                else:
                    output_list.append([img_id, cls_id, bbox_dict['score'], *bbox])
            return output_list

        Valid_GT = read_Json(GT_path, 'GT')
        Valid_PD = read_Json(PD_path, 'PD')
        mAP = mAP_metric(Valid_PD, Valid_GT)
        return mAP
