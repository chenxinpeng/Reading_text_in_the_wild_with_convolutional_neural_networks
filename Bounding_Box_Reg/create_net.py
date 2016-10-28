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

def Bounding_Box_Reg(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=64, pad=2, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=128, pad=2, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv4 = L.Convolution(n.pool3, kernel_size=3, num_output=512, pad=1, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1   = L.InnerProduct(n.pool4, num_output=4000, weight_filler=dict(type='xavier'))
    n.dp1   = L.Dropout(n.ip1, dropout_ratio=0.5)
    n.ip2   = L.InnerProduct(n.ip1, num_output=4, weight_filler=dict(type='xavier'))
    n.loss  = L.EuclideanLoss(n.ip2, n.label)

    return n.to_proto()

with open('BBR_train.prototxt', 'w') as f:
    f.write(str(Bounding_Box_Reg('train.h5', 16)))

with open('BBR_test.prototxt', 'w') as f:
    f.write(str(Bounding_Box_Reg('test.h5', 16)))
