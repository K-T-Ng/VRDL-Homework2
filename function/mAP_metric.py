from collections import Counter

import torch


def IOU(A, B):
    '''
    Calculate intersection over union between two bounding box

    Parameters:
        A, B: list[float] [Left, Top, Right, Bottom]
    Output:
        iou_val = iou between A and B
    '''
    area_A = (A[2]-A[0])*(A[3]-A[1])
    area_B = (B[2]-B[0])*(B[3]-B[1])
    
    # the intersect box
    L, R = max(A[0], B[0]), min(A[2], B[2])
    T, B = max(A[1], B[1]), min(A[3], B[3])
    area_I = max(R-L, 0) * max(B-T, 0)

    return area_I / (area_A + area_B - area_I + 1e-7)
    

def AP_metric(PD_boxes, GT_boxes, iou_threshold=0.5, num_classes=10):
    '''
    Calculate average precision at iou threshold

    Parameters:
        PD_boxes = [[img_id, cls_id, score, L, T, R, B], ...]
        GT_boxes = [[img_id, cls_id, L, T, R, B], ...]
        iou_threshold: float, decide the box become TP or FP
        num_classes: int, cls_id should be in range 0 ~ num_class-1
    '''
    AP_list = [] # record the AP for each class
    eps = 1e-7 # for computation stability

    for cls in range(num_classes):
        # collect those box with cls_id == cls
        PD = [box for box in PD_boxes if box[1] == cls]
        GT = [box for box in GT_boxes if box[1] == cls]

        # count how may cls object in each image (img_id)
        # e.g. {1: 4, 22: 1, 3682: 3}
        # indicates image (1, 22, 3682) has (4, 1, 3) bboxes with cls_id = cls
        GT_counter = Counter([gt[0] for gt in GT])

        # GT_counter becomes
        # {1   : torch.tensor([0,0,0,0]),
        #  22  : torch.tensor([0]),
        #  3682: torch.tensor([0,0])}
        for img_id, amount in GT_counter.items():
            GT_counter[img_id] = torch.zeros(amount)

        # sort the predict bbox by the confidnece
        PD.sort(key=lambda box: box[2], reverse=True)

        # Preallocate TP, FP and (TP+FN)
        TP = torch.zeros(len(PD))
        FP = torch.zeros(len(PD))
        All_P = len(GT)

        # loop over all predictions bbox
        for pid, pd in enumerate(PD):
            # ensure that we are dealing with the same img_id
            gt_boxes = [box for box in GT if box[0] == pd[0]]

            best_iou = 0
            # loop over those ground truth bbox
            for gid, gt in enumerate(gt_boxes):
                iou = IOU(pd[-4:], gt[-4:])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gid

            if best_iou > iou_threshold:
                #print(pd[0], best_gt_idx, best_iou, GT_counter[pd[0]])
                if GT_counter[pd[0]][best_gt_idx] == 0:
                    # good iou, haven't predict yet
                    TP[pid] = 1
                    GT_counter[pd[0]][best_gt_idx] = 1
                else:
                    # good iou, but predict this box before
                    FP[pid] = 1
            else:
                # bad iou
                FP[pid] = 1

        # cumsum TP, FP
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        # recall and precision
        recalls = TP_cumsum / (All_P + eps)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps)

        # add an number before, and do numerical integral to calculate AP
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        AP_list.append(torch.trapz(precisions, recalls))
    return sum(AP_list) / len(AP_list)
        
def mAP_metric(PD_boxes, GT_boxes, num_classes=10):
    '''
    Calculate mean average precision at iou threshold
    mean the (average precision at 0.5:0.05:0.95)

    Parameters:
        PD_boxes = [[img_id, cls_id, score, L, T, R, B], ...]
        GT_boxes = [[img_id, cls_id, L, T, R, B], ...]
        num_classes: int, cls_id should be in range 0 ~ num_class-1
    '''
    AP_list = []
    for ds in range(10):
        AP_list.append(AP_metric(PD_boxes, GT_boxes, 0.5+0.05*ds, num_classes))
    return sum(AP_list) / len(AP_list)
