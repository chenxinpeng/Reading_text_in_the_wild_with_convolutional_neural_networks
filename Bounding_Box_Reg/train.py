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

caffe.set_device(0)
caffe.set_mode_gpu()

solver=None
solver = caffe.SGDSolver('BBR_solver.prototxt')

solver.step(1)

solver.net.forward()
solver.test_nets[0].forward()