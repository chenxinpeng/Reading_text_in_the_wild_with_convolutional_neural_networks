import tempfile
import subprocess
import shlex
import os, sys
import numpy as np
import scipy.io
from PIL import Image, ImageDraw
import glob
import pickle

# import the Non-maximum Suppression
PYTHONPATH = '/home/chenxp/Documents/hitachi/paper1/edge_boxes_with_python'
sys.path.append(PYTHONPATH)
import nms

script_dirname = os.path.abspath(os.path.dirname(__file__))

np.set_printoptions(threshold='nan')

def get_windows(image_fnames, cmd='edge_boxes_wrapper'):
    """
    Run MATLAB EdgeBoxes code on the given image filenames to
    generate window proposals.

    Parameters
    ----------
    image_filenames: strings
        Paths to images to run on.
    cmd: string
        edge boxes function to call:
            - 'edge_boxes_wrapper' for effective detection proposals paper configuration.
    """
    # Form the MATLAB script command that processes images and write to
    # temporary results file.
    f, output_filename = tempfile.mkstemp(suffix='.mat')
    os.close(f)
    fnames_cell = '{' + ','.join("'{}'".format(x) for x in image_fnames) + '}'
    command = "{}({}, '{}')".format(cmd, fnames_cell, output_filename)
    #print(command)

    # Execute command in MATLAB.
    mc = "sudo /usr/local/MATLAB/R2015b/bin/matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(command)
    pid = subprocess.Popen(shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")
    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    # Remove temporary file, and return.
    os.remove(output_filename)
    if len(all_boxes) != len(image_fnames):
        raise Exception("Something went wrong computing the windows!")
    
    #print(all_boxes[0])
    return all_boxes

if __name__ == '__main__':
    """
    Run a demo.
    """
    import time

    image_lists = glob.glob('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/train-textloc/*.jpg')
    image_filenames = []
    for item in image_lists:
        image_filenames.append(item)
    
    t = time.time()

    # [boxes: ymin, xmin, ymax, xmax, score]
    boxes = get_windows(image_filenames)
  
    # do Non-maximum Suppression
    boxes2 = [] # stored the bounding box after NMS
    for i in range(len(boxes)):
        boxes2.append(nms.non_max_suppression(boxes[i], 0.8))
    
    # test: Plot the bounding boxes in the image
    for img in image_filenames:
        im = Image.open(img)
        draw = ImageDraw.Draw(im)
        idx = image_filenames.index(img)
        for i in range(len(boxes2[idx])):
            x1, y1, x2, y2 = int(boxes2[idx][i][1]), int(boxes2[idx][i][0]), int(boxes2[idx][i][3]), int(boxes2[idx][i][2])
            draw.rectangle((x1, y1, x2, y2), outline='red')
        im.save(os.path.join('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/bbox2/', os.path.basename(img)))

    del draw
    
    for img in image_filenames:
        im = Image.open(img)
        draw2 = ImageDraw.Draw(im)
        idx = image_filenames.index(img)
        for i in range(len(boxes[idx])):
            x1, y1, x2, y2 = int(boxes[idx][i][1]), int(boxes[idx][i][0]), int(boxes[idx][i][3]), int(boxes[idx][i][2])
            draw2.rectangle((x1, y1, x2, y2), outline='red')
        im.save(os.path.join('/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/bbox/', os.path.basename(img)))

    del draw2


        # save boxes and image_filenames to each picklefile: boxes_edgeBoxes.pkl, boxes_image_file_names.pkl
    file_boxes = open('boxes_edgeBoxes_with_NMS.pkl', 'wb')
    file_boxes_image_file_names = open('boxes_image_file_names.pkl', 'wb')
    pickle.dump(boxes2, file_boxes, -1)
    pickle.dump(image_filenames, file_boxes_image_file_names, -1)
    file_boxes.close()
    file_boxes_image_file_names.close()

    file_boxes_without_NMS = open('boxes_edgeBoxes_without_NMS.pkl', 'wb')
    pickle.dump(boxes, file_boxes_without_NMS, -1)
    file_boxes_without_NMS.close()


    '''
    ## Warning: The code below is moved to the python file: calRecall.py!
    ##
    # caculate the recall
    gt_file_dir = '/home/chenxp/Documents/hitachi/paper1/sceneData/ICDAR2011/train_gt'
    IOUs = []
    for img in image_filenames:
        # get the index of the boxes corresponding to the image_filenames
        idx = image_filenames.index(img)

        img_basename = os.path.basename(img)
        img_name, temp1 = os.path.splitext(img_basename)
        gt_boxes = []
        gt_txt = open(gt_file_dir + '/gt_' + img_name + '.txt')
        for line in gt_txt.readlines():
            gt_line = line.strip().split(',')
            gt_boxes.append(gt_line)
        count = 0
        for i in range(len(boxes[idx])):
            for j in range(len(gt_boxes)):
                XA1, YA1 = int(boxes[idx][i][1]), int(boxes[idx][i][0])
                XA2, YA2 = int(boxes[idx][i][3]), int(boxes[idx][i][2])
                S_A12 = (int(boxes[idx][i][3]) - int(boxes[idx][i][1]))  * (int(boxes[idx][i][2]) - int(boxes[idx][i][0]))

                XB1, YB1 = int(gt_boxes[j][0]), int(gt_boxes[j][1])
                XB2, YB2 = int(gt_boxes[j][2]), int(gt_boxes[j][3])
                S_B12 = (int(gt_boxes[j][2]) - int(gt_boxes[j][0])) * (int(gt_boxes[j][3]) - int(gt_boxes[j][1]))

                S_intersect = rectint(XA1, YA1, XA2, YA2, XB1, YB1, XB2, YB2)
                IoU = float(S_intersect) / float(S_A12 + S_B12 - S_intersect)
                if abs(IoU) >= 0.5:
                    count = count + 1

        print count, len(boxes[idx])
        IOUs.append((count + 1) / float(len(boxes[idx]) + 1))

    sum_IOUs = 0.0
    for item in IOUs:
        sum_IOUs = sum_IOUs + float(item)
    print sum_IOUs, len(IOUs)
    print("EdgeBoxes processed {} images in {:.3f} s".format(len(image_filenames), time.time() - t))
    print "The Recall: ", (sum_IOUs / len(IOUs))
    '''


