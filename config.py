import os

DARKNET_WEIGHT = os.path.join('models', 'yolov3.weights')

DATA = {"CLASSES": [' 0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        "NUM": 10}

# model
'''
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# big obj
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }
'''
MODEL = {"ANCHORS": [[(2.83, 6.20), (4.43, 8.92), (4.24, 14.38)],  # small obj
                     [(3.06, 5.75), (3.03, 8.39), (4.02, 6.96)],   # medium obj
                     [(1.96, 4.71), (2.53, 4.40), (3.21, 5.20)]],  # big obj
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

# train
TRAIN = {
         "TRAIN_IMG_SIZE": 448,
         "BATCH_SIZE": 6,
         "MULTI_SCALE_TRAIN": True,
         "IOU_THRESHOLD_LOSS": 0.5,
         "EPOCHS": 30,
         "NUMBER_WORKERS": 0,
         "MOMENTUM": 0.9,
         "WEIGHT_DECAY": 0.0005,
         "LR_INIT": 1e-4,
         "LR_END": 1e-6,
         "WARMUP_EPOCHS": 2,  # or None
         "GRADIENT_CLIP": 2000,
         "CLAMP": 3,
         }


# test
TEST = {
        "TEST_IMG_SIZE": 608,
        "BATCH_SIZE": 1,
        "NUMBER_WORKERS": 0,
        "CONF_THRESH": 0.03,
        "NMS_THRESH": 0.5,
        }
