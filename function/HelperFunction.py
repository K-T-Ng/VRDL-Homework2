import json

import numpy as np
import scipy.io as sio


def numpy2int(*ndarrays):
    return list(int(ndarray.squeeze()) for ndarray in ndarrays)


def convert_bbox_type(bbox, input_type='LTWH', output_type='LTRB'):
    '''
    convert bounding box from input type to output type
    Each bounding box (bbox) can be described by 4 elements
    {
        'L', 'R': Left/Right endpoint
        'T', 'B': Top/Bottom endpoint
        'W', 'H': Width/Height of the bbox
        'X', 'Y': Col/Row id of the center of the bbox
    }

    Parameters:
        bbox: list containing 4 int or float
        input_type: string with 4 characters, indicate the type of input bbox
        output_type: string with 4 characters, indicate the type of output bbox
    Output:
        output_bbox: list containing 4 floats, with output_type order
    '''

    # Here, we only implemented 3 types of bbox
    # 1. LTWH: COCO format (left, top, width, height)
    # 2. LTRB: torchvision faster rcnn format (left, top, right, bottom)
    # 3. XYWH: YOLO format (Center col, Center row, width, height)
    assert input_type in ['LTWH', 'LTRB', 'XYWH'], "No such input_type"
    assert output_type in ['LTWH', 'LTRB', 'XYWH'], "No such output_type"

    box = list(map(float, bbox))
    if input_type == 'LTWH':
        L, T, W, H = box
        R, B, X, Y = L+W, T+H, L+W/2, T+H/2
    elif input_type == 'LTRB':
        L, T, R, B = box
        W, H, X, Y = R-L, B-T, (L+R)/2, (T+B)/2
    else:  # input_type == 'XYWH':
        X, Y, W, H = box
        L, T, R, B = X-W/2, Y-H/2, X+W/2, Y+H/2

    if output_type == 'LTWH':
        return [L, T, W, H]
    elif output_type == 'LTRB':
        return [L, T, R, B]
    else:  # output_type == 'XYWH':
        return [X, Y, W, H]
