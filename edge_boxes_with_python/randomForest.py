import os, sys
import glob
import numpy as np
from PIL import Image

from skimage.feature import hog
from skimage import data, color, exposure

import pickle

'''
## Resize parameters
'''
Width = 100
Height = 32

image_lists = glob.glob('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/train-textloc/*.jpg')
image_filenames = []
for item in image_lists:
    image_filenames.append(item)

gt_file_dir = '/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/train_gt'

HOG_feature = np.zeros((1145, 7200))
HOG_labels = np.zeros(1145)

# Positive
count = 0
for img in image_filenames:
    # read ground truth txx file
    img_basename = os.path.basename(img)
    img_name, temp1 = os.path.splitext(img_basename)
    
    gt_boxes = [] # x1, y1, x2, y2, score
    gt_txt = open(gt_file_dir + '/gt_' + img_name + '.txt')
    for line in gt_txt.readlines():
        gt_line = line.strip().split(',')
        gt_boxes.append(gt_line)

    im = Image.open(img).convert('L') # convert to gray scale
    
    for coordinateXY in gt_boxes:
        crop_box = (int(coordinateXY[0]), int(coordinateXY[1]), int(coordinateXY[2]), int(coordinateXY[3]))
        cropped_im = im.crop(crop_box) # Crop the boxes area of ground truth
        cropped_im = cropped_im.resize((Width, Height), Image.ANTIALIAS) # resize the image to (128, 100)
        fd, hog_image = hog(np.array(cropped_im)/255.0, orientations=36, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
        HOG_feature[count] = fd
        HOG_labels[count] = 1
        count = count + 1
print count

# Negative
image_neg_dir = glob.glob('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/rf_neg_imgs/*.jpg')
image_neg_lists = []
for item in image_neg_dir:
    image_neg_lists.append(item)

for img in image_neg_lists:
    neg_im = Image.open(img).convert('L') # convert to gray scale

    resize_neg_img = neg_im.resize((Width, Height), Image.ANTIALIAS) # resize the image to (32, 100)
    neg_fd, neg_hog_image = hog(np.array(resize_neg_img)/255.0, orientations=36, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
    HOG_feature[count] = neg_fd
    HOG_labels[count] = 0

print np.shape(HOG_feature)	
print np.shape(HOG_labels)

# Train HOG features using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, max_depth=64)
rfc.fit(HOG_feature, HOG_labels)

# Perform cross-validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(rfc, HOG_feature, HOG_labels, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

output = open('randomModel.pkl', 'wb')
pickle.dump(rfc, output, -1)
output.close()


'''
## Predict, Save the through the Random Forest Classifier 
## boxes_edgeBoxes_RFC.pkl
'''

## Load data saved in pickle file before
file_boxes = open('boxes_edgeBoxes_with_NMS.pkl', 'rb')
file_boxes_image_file_names = open('boxes_image_file_names.pkl', 'rb')

boxes = pickle.load(file_boxes)
image_filenames = pickle.load(file_boxes_image_file_names)

file_boxes_RFC = open('boxes_edgeBoxes_RFC.pkl', 'wb')

boxes_RFC = []
for img in image_filenames:
    im = Image.open(img).convert('L')
    idx = image_filenames.index(img)
    boxes_RFC_Temp = []
    boxes_features_RFC_Temp = np.zeros((len(boxes[idx]), 7200))
    for i in range(len(boxes[idx])):
        '''
        ## IMPORTANT: PAY ATTENTION!!!
        ## The index order of boxes is: Ymin, Xmin, Ymax, Xmax
        '''
        XA1, YA1 = int(boxes[idx][i][1]), int(boxes[idx][i][0])
        XA2, YA2 = int(boxes[idx][i][3]), int(boxes[idx][i][2])  
        crop_box = (XA1, YA1, XA2, YA2)
        cropped_im = im.crop(crop_box) # Crop the area of boxes proposal
        cropped_im = cropped_im.resize((Width, Height), Image.ANTIALIAS) # resize the image to (128, 100)

        '''
        ## Extract the proposal boxes's HOG feature
        '''
        fd, hog_image = hog(np.array(cropped_im)/255.0, orientations=36, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=True)
        
        boxes_features_RFC_Temp[i] = fd
    
    '''
    ## Predict the prob of proposal, if the prob greater than 0.5, then add to the boxes_edgeBoxes_RFC.pkl
    '''
    prob = rfc.predict_proba(boxes_features_RFC_Temp)
    print prob
    for j in range(len(prob)):
        if prob[j][1] >= 0.5:
            boxes_RFC_Temp.append(boxes[idx][i])
    boxes_RFC.append(boxes_RFC_Temp)

pickle.dump(boxes_RFC, file_boxes_RFC, -1)

file_boxes.close()
file_boxes_RFC.close()
file_boxes_image_file_names.close()

