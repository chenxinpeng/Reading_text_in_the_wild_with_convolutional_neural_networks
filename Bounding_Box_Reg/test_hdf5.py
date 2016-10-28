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
from caffe import layers as L
from caffe import params as P

import h5py


f = h5py.File('train.h5', 'r')
train_data = f['data']

print train_data.shape
f.close()

f2 = h5py.File('test.h5', 'r')
test_data = f2['label']

print type(test_data[507])
print type(test_data)