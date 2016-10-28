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

import h5py

# Size of images
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 32

# train hdf5, validation hdf5 path
dirname = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python/hdf5'
train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname,'test.h5')

os.system('rm -rf ' + train_filename)
os.system('rm -rf ' + test_filename)

# read image
file_train_data = open('boxes_image_file_names.pkl', 'rb')
imgs_names = pickle.load(file_train_data)

# read Bounding Boxes that IOU greater than 0.5
file_map_gtBox_BBoxes = open('map_gtBox_BBoxes.pkl', 'rb')
map_gtBox_BBoxes = pickle.load(file_map_gtBox_BBoxes)

# Image transform
def transform_img(img, img_width=IMAGE_WIDTH, img_height = IMAGE_HEIGHT):
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    return img

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
## Train
'''
print '\nCreating train HDF5 data...'

labels_train_data = []
imgs_train_data = []
for img_idx, img_path in enumerate(imgs_names):
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
        #print BBox_x1, BBox_y1, BBox_x2, BBox_y2
        BBox_img = crop_img(gray_img, BBox_x1, BBox_y1, BBox_x2, BBox_y2)
        #print BBox_img.shape
        BBox_img = transform_img(BBox_img)
        imgs_train_data.append(np.reshape(BBox_img, (1, BBox_img.shape[0], BBox_img.shape[1])))
 
        GroundTruth_Box = each_BBox_map_gtBox_BBoxes[1]
        GT_Box_x1, GT_Box_y1 = int(GroundTruth_Box[0]), int(GroundTruth_Box[1])
        GT_Box_x2, GT_Box_y2 = int(GroundTruth_Box[2]), int(GroundTruth_Box[3])       
        labels_train_data.append(np.array([GT_Box_x1, GT_Box_y1, GT_Box_x2, GT_Box_y2]))

print np.array(imgs_train_data).shape
with h5py.File(train_filename, 'w') as f:
    f.create_dataset('data', data = np.asarray(imgs_train_data).astype(np.float32))
    f.create_dataset('label', data = np.asarray(labels_train_data).astype(np.float32))
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')


'''
## Test
'''
print '\nCreating test HDF5 data...'

labels_validation_data = []
imgs_validation_data = []
for img_idx, img_path in enumerate(imgs_names):
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

        BBox_img = crop_img(gray_img, BBox_x1, BBox_y1, BBox_x2, BBox_y2)
        BBox_img = transform_img(BBox_img)
        imgs_validation_data.append(np.reshape(BBox_img, (1, BBox_img.shape[0], BBox_img.shape[1])))

        GroundTruth_Box = each_BBox_map_gtBox_BBoxes[1]
        GT_Box_x1, GT_Box_y1 = int(GroundTruth_Box[0]), int(GroundTruth_Box[1])
        GT_Box_x2, GT_Box_y2 = int(GroundTruth_Box[2]), int(GroundTruth_Box[3])       
        labels_validation_data.append(np.array([GT_Box_x1, GT_Box_y1, GT_Box_x2, GT_Box_y2]))

with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data = np.asarray(imgs_validation_data).astype(np.float32))
    f.create_dataset('label', data = np.asarray(labels_validation_data).astype(np.float32))
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')

print np.array(imgs_validation_data).shape
file_train_data.close()
file_map_gtBox_BBoxes.close()
