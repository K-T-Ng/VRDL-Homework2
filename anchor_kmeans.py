import os
import json
import random

from PIL import Image
from function.SVHN_Dataset import SVHN_Dataset


def yolo_kmeans_metric(box, centroid):
    '''
    box and centroid are 2 boxes of the form (0, 0, W, H)
    i.e, both box and centroid centered at (0, 0) and it's W and H are
         normalized by dividing the image W, H

    The metric is defined by
        d(box, centroid) = 1 - iou(box, centroid)
    '''
    box_area = box[2] * box[3]
    centroid_area = centroid[2] * centroid[3]

    # since both boxes centered at 0
    inter_w = min(box[2], centroid[2])
    inter_h = min(box[3], centroid[3])
    inter_area = inter_w * inter_h

    union_area = box_area + centroid_area - inter_area
    return 1 - inter_area / union_area

if __name__ == '__main__':
    Dataset = SVHN_Dataset(mode='Kmeans')

    bbox_list = []
    for i in range(len(Dataset)):
        img, boxes = Dataset[i]

        for box in boxes:
            L, T, R, B, _ = box
            bbox_list.append([0, 0, float(R-L), float(B-T)])

        if i % 1000 == 0:
            print(i)
    ''' k means goes here '''

    n_anchors = 9
    tolerance, dloss, loss = 1e-6, 1, -100
    max_iter, iter_id = 500, 0

    # initalize
    centroids = random.sample(bbox_list, n_anchors)

    # iterate until convergence
    while dloss > tolerance and iter_id < max_iter:
        new_loss, new_groups, new_centroids = 0, [], []
        for i in range(n_anchors):
            new_groups.append([])
            new_centroids.append([0, 0, 0, 0])

        for box in bbox_list:
            min_dist = 1

            for c_id, centroid in enumerate(centroids):
                dist = yolo_kmeans_metric(box, centroid)
                if dist < min_dist:
                    min_dist = dist
                    gp_index = c_id

            new_loss += min_dist
            new_groups[gp_index].append(box)
            new_centroids[gp_index][2] += box[2]
            new_centroids[gp_index][3] += box[3]

        for i in range(n_anchors):
            new_centroids[i][2] /= len(new_groups[i])
            new_centroids[i][3] /= len(new_groups[i])

        dloss = abs(new_loss - loss)
        loss = new_loss
        centroids = new_centroids
        iter_id += 1
        print(iter_id, dloss)

    # collect those width and heigth
    ANCHORS = []
    for centroid in centroids:
        ANCHORS.append((centroid[2], centroid[3]))

    ANCHORS.sort(key=lambda x: x[0]*x[1])
    for anchor in ANCHORS:
        print(anchor[0], anchor[1])

    '''
    (22.61, 49.62)   --stride=8 --> (2.82625, 6.2025)
    (35.47, 71.36)   --stride=8 --> (4.43375, 8.92)
    (33.95, 115.04)  --stride=8 --> (4.24375, 14.38)
    (48.97, 91.93)   --stride=16--> (3.060625, 5.745625)
    (48.52, 134.27)  --stride=16--> (3.0325, 8.391875)
    (64.33, 111.38)  --stride=16--> (4.020625, 6.96125)
    (62.71, 150.82)  --stride=32--> (1.9596875, 4.713125)
    (80.02, 140.86)  --stride=32--> (2.525625, 4.401875)
    (102.62, 166.35) --stride=32--> (3.206875, 5.1984375)
    '''
