import shlex
import os, sys
import numpy as np
import scipy.io
from PIL import Image, ImageDraw
import glob
import pickle


'''
## Calculate Intersection of two rectangles
'''
def rectint(XA1, YA1, XA2, YA2, XB1, YB1, XB2, YB2):
    a = max(XA1, XB1)
    b = max(YA1, YB1)
    c = min(XA2, XB2)
    d = min(YA2, YB2)

    return max(0, c-a) * max(0, d-b)
    #return abs((c-a)*(d-b))
    

'''
## Load data saved in pickle file before
'''
file_boxes = open('boxes_edgeBoxes_without_NMS.pkl', 'rb')
file_boxes_image_file_names = open('boxes_image_file_names.pkl', 'rb')

boxes = pickle.load(file_boxes)
image_filenames = pickle.load(file_boxes_image_file_names)

'''
## save
'''
file_boxes_IOU_gt05_withoutNMS = open('boxes_IOU_gt05_withoutNMS.pkl', 'wb')
boxes_IOU_gt05 = []

file_map_gtBox_BBoxes = open('map_gtBox_BBoxes.pkl', 'wb')
map_gtBox_BBoxes = []

'''
## caculate the recall
'''
gt_file_dir = '/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/train_gt' 
IOUs = []
for img in image_filenames:
    boxes_IOU_gt05_temp = []

    map_gtBox_BBoxes_eachImg = []
    # get the index of the boxes corresponding to the image_filenames
    idx = image_filenames.index(img)

    img_basename = os.path.basename(img)
    img_name, temp1 = os.path.splitext(img_basename)
    gt_boxes = []
    gt_txt = open(gt_file_dir + '/gt_' + img_name + '.txt')
    for line in gt_txt.readlines():
        gt_line = line.strip().split(',')
        gt_boxes.append(gt_line)
    count=0
    for i in range(len(boxes[idx])):
        XA1, YA1 = int(boxes[idx][i][1]), int(boxes[idx][i][0])
        XA2, YA2 = int(boxes[idx][i][3]), int(boxes[idx][i][2])
        S_A12 = (int(boxes[idx][i][3]) - int(boxes[idx][i][1]))  * (int(boxes[idx][i][2]) - int(boxes[idx][i][0]))
        for j in range(len(gt_boxes)):
            XB1, YB1 = int(gt_boxes[j][0]), int(gt_boxes[j][1])
            XB2, YB2 = int(gt_boxes[j][2]), int(gt_boxes[j][3])
            S_B12 = (int(gt_boxes[j][2]) - int(gt_boxes[j][0])) * (int(gt_boxes[j][3]) - int(gt_boxes[j][1]))

            S_intersect = rectint(XA1, YA1, XA2, YA2, XB1, YB1, XB2, YB2)
            IoU = float(S_intersect) / float(S_A12 + S_B12 - S_intersect)
            if IoU >= 0.5:
            	boxes_IOU_gt05_temp.append(boxes[idx][i])
            	map_gtBox_BBoxes_eachImg.append([boxes[idx][i], gt_boxes[j]])
                count = count + 1
                break
    print count, len(boxes[idx])
    IOUs.append((count + 1) / float(len(boxes[idx]) + 1))
    boxes_IOU_gt05.append(boxes_IOU_gt05_temp)
    map_gtBox_BBoxes.append(map_gtBox_BBoxes_eachImg)

sum_IOUs = 0.0
for item in IOUs:
    sum_IOUs = sum_IOUs + float(item)
print sum_IOUs, len(IOUs)
print "The Recall: ", (sum_IOUs / len(IOUs))

pickle.dump(boxes_IOU_gt05, file_boxes_IOU_gt05_withoutNMS, -1)
pickle.dump(map_gtBox_BBoxes, file_map_gtBox_BBoxes, -1)

file_boxes_IOU_gt05_withoutNMS.close()
file_boxes.close()
file_boxes_image_file_names.close()
file_map_gtBox_BBoxes.close()