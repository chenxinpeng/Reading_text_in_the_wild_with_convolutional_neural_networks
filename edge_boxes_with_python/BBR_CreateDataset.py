import os, sys, glob
import numpy as np
import pickle
import random

from PIL import Image, ImageDraw
import cv2

CAFFE_PATH = '/home/chenxp/caffe/python'
sys.path.append(CAFFE_PATH)
import caffe
import caffe.io
from caffe.proto import caffe_pb2
import lmdb


# Size of images
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32

# train_lmdb, validation_lmdb path
train_imgs_lmdb = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python/train_imgs_lmdb'
validation_imgs_lmdb = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python/validation_imgs_lmdb'

train_labels_lmdb = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python/train_labels_lmdb'
validation_labels_lmdb = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python/train_labels_lmdb'

os.system('rm -rf ' + train_imgs_lmdb)
os.system('rm -rf ' + train_labels_lmdb)
os.system('rm -rf ' + validation_imgs_lmdb)
os.system('rm -rf ' + validation_labels_lmdb)

# read image
file_train_data = open('boxes_image_file_names.pkl', 'rb')
train_data = pickle.load(file_train_data)

# read Bounding Boxes that IOU greater than 0.5
file_map_gtBox_BBoxes = open('map_gtBox_BBoxes.pkl', 'rb')
map_gtBox_BBoxes = pickle.load(file_map_gtBox_BBoxes)

# Image transform
def transform_img(img, img_width=IMAGE_WIDTH, img_height = IMAGE_HEIGHT):
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    return img

'''
def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=1,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=img)
'''

# Calculate the inflated area
def crop_img(img, x1, y1, x2, y2):
    h, w = img.shape[0], img.shape[1]
    #print w, h
    xx1 = max(0, x1 - (x2 - x1) / 4)
    yy1 = max(0, y1 - (y2 - y1) / 4)
    xx2 = min(w, x2 + (x2 - x1) / 4)
    yy2 = min(h, y2 + (y2 - y1) / 4)
    
    return img[yy1:yy2, xx1:xx2]




'''
'''
## Train LMDB
'''
print '\nCreating train lmdb data...'
imgs_train_data, labels_train_data = [], []
for img_idx, img_path in enumerate(train_data):
    if img_idx % 6 != 0:
        continue
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    eachImg_map_gtBox_BBoxes = map_gtBox_BBoxes[img_idx]
      
    for i in range(len(eachImg_map_gtBox_BBoxes)):
        each_BBox_map_gtBox_BBoxes = eachImg_map_gtBox_BBoxes[i]
        Candidate_Box = each_BBox_map_gtBox_BBoxes[0]
        BBox_x1, BBox_y1 = int(Candidate_Box[1]), int(Candidate_Box[0])
        BBox_x2, BBox_y2 = int(Candidate_Box[3]), int(Candidate_Box[2])
        #print BBox_x1, BBox_y1, BBox_x2, BBox_y2
        BBox_img = crop_img(gray_img, BBox_x1, BBox_y1, BBox_x2, BBox_y2)
        #print BBox_img.shape
        BBox_img = transform_img(BBox_img)
        imgs_train_data.append(BBox_img)

        GroundTruth_Box = each_BBox_map_gtBox_BBoxes[1]
        GT_Box_x1, GT_Box_y1 = GroundTruth_Box[0], GroundTruth_Box[1]
        GT_Box_x2, GT_Box_y2 = GroundTruth_Box[2], GroundTruth_Box[3]       
        labels_train_data.append(np.array([GT_Box_x1, GT_Box_y1, GT_Box_x2, GT_Box_y2]))

counter_img_train = 0
counter_label_train = 0

db_imgs_train = lmdb.open(train_imgs_lmdb, map_size=int(1e12))
with db_imgs_train.begin(write=True) as txn_img_train:
    for img in imgs_train_data:
        datum = caffe.io.array_to_datum(np.expand_dims(img, axis=0))
        txn_img_train.put("{:0>10d}".format(counter_img_train), datum.SerializeToString())
        counter_img_train = counter_img_train + 1
print("Processed {:d} images".format(counter_img_train))

db_labels_train = lmdb.open(train_labels_lmdb, map_size=int(1e12))
with db_labels_train.begin(write=True) as txn_label_train:
    for label in labels_train_data:
        datum = caffe.io.array_to_datum(np.expand_dims(label, axis=0))
        txn_label_train.put("{:0>10d}".format(counter_label_train), datum.SerializeToString())
        counter_label_train = counter_img_train + 1
print("Processed {:d} labels".format(counter_label_train))

db_imgs_train.close()
db_labels_train.close()


'''
## Validation LMDB
'''
print '\nCreating validation lmdb data...'
imgs_validation_data, labels_validation_data = [], []
for img_idx, img_path in enumerate(train_data):
    if img_idx % 6 == 0:
        continue
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    eachImg_map_gtBox_BBoxes = map_gtBox_BBoxes[img_idx]
      
    for i in range(len(eachImg_map_gtBox_BBoxes)):
        each_BBox_map_gtBox_BBoxes = eachImg_map_gtBox_BBoxes[i]
        Candidate_Box = each_BBox_map_gtBox_BBoxes[0]
        BBox_x1, BBox_y1 = int(Candidate_Box[1]), int(Candidate_Box[0])
        BBox_x2, BBox_y2 = int(Candidate_Box[3]), int(Candidate_Box[2])

        BBox_img = crop_img(gray_img, BBox_x1, BBox_y1, BBox_x2, BBox_y2)
        BBox_img = transform_img(BBox_img)
        imgs_validation_data.append(BBox_img)

        GroundTruth_Box = each_BBox_map_gtBox_BBoxes[1]
        GT_Box_x1, GT_Box_y1 = GroundTruth_Box[0], GroundTruth_Box[1]
        GT_Box_x2, GT_Box_y2 = GroundTruth_Box[2], GroundTruth_Box[3]       
        labels_validation_data.append(np.array([GT_Box_x1, GT_Box_y1, GT_Box_x2, GT_Box_y2]))

# Counter
counter_img_validation = 0
counter_label_validation = 0

db_imgs_train = lmdb.open(validation_imgs_lmdb, map_size=int(1e12))
with db_imgs_validation.begin(write=True) as txn_img_validation:
    for img in imgs_validation_data:
        datum = caffe.io.array_to_datum(np.expand_dims(img, axis=0))
        txn_img_validation.put("{:0>10d}".format(counter_img_validation), datum.SerializeToString())
        counter_img_validation = counter_img_validation + 1
print("Processed {:d} images".format(counter_img_validation))

db_labels_validation = lmdb.open(validation_labels_lmdb, map_size=int(1e12))
with db_labels_validation.begin(write=True) as txn_label_validation:
    for label in labels_validation_data:
        datum = caffe.io.array_to_datum(np.expand_dims(label, axis=0))
        txn_label_validation.put("{:0>10d}".format(counter_label_validation), datum.SerializeToString())
        counter_label_validation = counter_img_validation + 1
print("Processed {:d} labels".format(counter_label_validation))

db_imgs_validation.close()
db_labels_validation.close()
'''



        