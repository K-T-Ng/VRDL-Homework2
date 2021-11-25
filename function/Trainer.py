import os
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import config as cfg
from models.yolov3 import Yolov3
from function.SVHN_Dataset import SVHN_Dataset
from function.yololoss import YoloV3Loss
from function.CosineDecayLR import CosineDecayLR


class Trainer(object):
    def __init__(self):
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"

        self.TrainDs = SVHN_Dataset(mode='Train',
                                    img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.TrainLoader = DataLoader(self.TrainDs,
                                      batch_size=cfg.TRAIN["BATCH_SIZE"],
                                      num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                      shuffle=True)

        self.yolov3 = Yolov3().to(self.device)
        self.yolov3.load_darknet_weights(cfg.DARKNET_WEIGHT)

        self.optimizer = optim.SGD(self.yolov3.parameters(),
                                   lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"],
                                   weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.lr_scheduler = CosineDecayLR(
            self.optimizer,
            T_max=self.epochs*len(self.TrainDs),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.TrainDs))

    def train(self):
        for ep in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (imgs, lbl_s, lbl_m, lbl_l, box_s, box_m, box_l) in \
                    enumerate(self.TrainLoader):
                self.lr_scheduler.step(len(self.TrainDs)*ep + i)

                imgs = imgs.to(self.device)
                lbl_s = lbl_s.to(self.device)
                lbl_m = lbl_m.to(self.device)
                lbl_l = lbl_l.to(self.device)
                box_s = box_s.to(self.device)
                box_m = box_m.to(self.device)
                box_l = box_l.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                    p, p_d, lbl_s, lbl_m, lbl_l, box_s, box_m, box_l)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_items = torch.tensor(
                    [loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                del imgs, lbl_s, lbl_m, lbl_l, box_s, box_m, box_l

                if i % 10 == 0:
                    print(ep, i, mloss)

                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.TrainDs.img_size = random.choice(range(10, 20))*32
                    print('image size becomes {}'.format(
                        self.TrainDs.img_size))

            if ep > 9 and ep % 1 == 0:
                torch.save(self.yolov3.state_dict(),
                           os.path.join('weights', str(ep)+'.pth'))

    def load_weight(self, path):
        self.yolov3.load_state_dict(torch.load(path, map_location=self.device))
        return

    def save_weight(self, path):
        torch.save(self.yolov3.state_dict(), path)
        return

    def _max_params(self):
        ans = 0
        for p in self.yolov3.parameters():
            if torch.any(torch.isnan(p)):
                print('nan, fuck up')
                return
            if p.data.max().item() > ans:
                ans = p.data.max().item()
        return ans
