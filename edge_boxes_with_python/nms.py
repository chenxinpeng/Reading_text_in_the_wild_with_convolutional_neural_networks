import numpy as np
    

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return empty list
    if len(boxes) == 0:
        return []

    for i in range(len(boxes)):
        boxes[i][0] = float(boxes[i][0])
        boxes[i][1] = float(boxes[i][1])
        boxes[i][2] = float(boxes[i][2])
        boxes[i][3] = float(boxes[i][3])
    
    # initlize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])
    print type(area[0])

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]

