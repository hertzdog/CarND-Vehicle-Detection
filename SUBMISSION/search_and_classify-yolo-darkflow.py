import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import math
from lesson_functions import *
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip
import json


import sys
yolo_path="/Users/franz/Desktop/GNULINUX/darkflow/"
sys.path.append(yolo_path)

from net.build import TFNet
yolo_config_path = yolo_path + "cfg/yolo.cfg"
yolo_load_path = yolo_path + "bin/yolo.weights"
yolo_threshold = 0.25

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": yolo_threshold}

tfnet = TFNet(options)

imgcv = cv2.imread("/Users/franz/Desktop/GNULINUX/Udacity/ND013/CarND-Vehicle-Detection/test_images/test1.jpg")
result = tfnet.return_predict(imgcv)

def process_image(image):
    hot_box_list = []
    # check if the colorspace has to be changed
    # run darknet on the image
    resultjson = tfnet.return_predict(image)
    hot_box_list = jsontobbox (resultjson)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,hot_box_list)
    heat = apply_threshold(heat,0)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image, labels)
    #draw_img=draw_boxes(image, hot_box_list)
    return draw_img

def jsontobbox(json_result_from_darkflow):
    box_list = []
    for element in json_result_from_darkflow:
        if element["label"] == "car":
            startx = np.int(element["topleft"]["x"])
            starty = np.int(element["topleft"]["y"])
            endx = np.int(element["bottomright"]["x"])
            endy = np.int(element["bottomright"]["y"])
            box_list.append(((startx, starty), (endx, endy)))
    return box_list

jsontobbox(result)
print(result)

print('Processing the video...')

out_dir='output_images/'
input_dir = '../Udacity/ND013/CarND-Vehicle-Detection/'

#input_file='test_video.mp4'
input_file='project_video.mp4'

output_file=input_dir + out_dir+'processed_'+input_file
clip = VideoFileClip(input_dir+input_file)
out_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
out_clip.write_videofile(output_file, audio=False)

print("Done")
