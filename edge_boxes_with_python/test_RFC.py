import os, sys
import pickle
import numpy as np
from PIL import Image, ImageDraw



## Load data saved in pickle file before
file_boxes = open('boxes_IOU_gt05_withoutNMS.pkl', 'rb')
file_boxes_image_file_names = open('boxes_image_file_names.pkl', 'rb')

boxes = pickle.load(file_boxes)
image_filenames = pickle.load(file_boxes_image_file_names)

for img in image_filenames:
    print img, type(img)
    im = Image.open(img)
    draw = ImageDraw.Draw(im)
    idx = image_filenames.index(img)

    for i in range(len(boxes[idx])):
        x1, y1, x2, y2 = int(boxes[idx][i][1]), int(boxes[idx][i][0]), int(boxes[idx][i][3]), int(boxes[idx][i][2])
        draw.rectangle((x1, y1, x2, y2), outline='red')
        im.save(os.path.join('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/bbox/', os.path.basename(img)))


del draw

file_boxes.close()
file_boxes_image_file_names.close()



