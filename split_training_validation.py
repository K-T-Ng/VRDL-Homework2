import os
import json
import random

import scipy.io as sio

from functions.HelperFunction import numpy2int

if __name__ == '__main__':
    # collect information from ./dataset/DataInfo/digitStruct.mat
    path = os.path.join('dataset', 'DataInfo', 'digitStruct.mat')
    ImagesInfo = sio.loadmat(path)
    ImagesInfo = ImagesInfo['digitStruct'][0]

    # bbox_list collect infos for each image, info includes
    #   image_id: int
    #   category_id: list[int], indicate the label id (0~9) for each bbox
    #   bbox: list[list[int]], indicate the bbox position (with LTWH type)
    bbox_list = [] # result
    for img_name, bbox in ImagesInfo:
        img_name, bbox = img_name[0], bbox[0]
        ddict = {
                    "image_id": int(img_name[:-4]),
                    "category_id": [],
                    "bbox": []
                }
        for box in bbox:
            height, left, top, width, label = numpy2int(*box)
            ddict["category_id"].append(label % 10)
            ddict["bbox"].append([left, top, width, height])
        bbox_list.append(ddict)
    
    # shuffle
    random.shuffle(bbox_list)

    # split it into training and validation data
    NumValid = 3402
    Train, Valid = bbox_list[:-NumValid], bbox_list[-NumValid:]

    # Convert them into the format that match the submission format (COCO)
    # i.e. multiple bbox in one dict -> one bbox in one dict
    def _convert_COCO_json(bbox_list):
        Obj = []
        for data in bbox_list:
            for cls_id, bbox in zip(data["category_id"], data["bbox"]):
                ddict = {
                            "image_id": data["image_id"],
                            "category_id": cls_id,
                            "bbox": bbox
                        }
                Obj.append(ddict)
        return json.dumps(Obj, indent=4)
    
    Train_Obj, Valid_Obj = _convert_COCO_json(Train), _convert_COCO_json(Valid)

    # write them into .dataset/DataInfo/{Train,Valid}_GT.json
    with open(os.path.join('dataset', 'DataInfo', 'Train_GT.json'), 'w') as fp:
        fp.write(Train_Obj)

    with open(os.path.join('dataset', 'DataInfo', 'Valid_GT.json'), 'w') as fp:
        fp.write(Valid_Obj)

    
