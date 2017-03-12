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


import sys
yolo_path="/Users/franz/Desktop/GNULINUX/darkflow/"
sys.path.append(yolo_path)

from net.build import TFNet
yolo_config_path = yolo_path + "cfg/yolo.cfg"
yolo_load_path = yolo_path + "bin/yolo.weights"
yolo_threshold = 0.5

options = {"model": "/Users/franz/Desktop/GNULINUX/darkflow/cfg/yolo.cfg", "load": "/Users/franz/Desktop/GNULINUX/darknet/yolo.weights", "threshold": yolo_threshold}

tfnet = TFNet(options)

imgcv = cv2.imread("/Users/franz/Desktop/GNULINUX/Udacity/ND013/CarND-Vehicle-Detection/test_images/test1.jpg")
result = tfnet.return_predict(imgcv)
print(result)


# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

#LOAD FROM pickle_file of the model
data_file = 'TrainedModel.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

svc = data['svc']
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']


# PROCESSING THE VIDEO
ystart = 400
ystop = 660
y_start_stop = [ystart, ystop] # Min and max in y to search in slide_window()
frames_per_hot_boxes=6
hot_box_list_history=[]
number_of_boxes = []

def process_image(image):
    hot_box_list = []
    for i in range(1):
        current_window=np.int(math.pow(2,(7+i)))
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                        xy_window=(current_window, current_window), xy_overlap=(0.5, 0.5))
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        hot_box_list = hot_box_list + hot_windows

    #print ("Hot boxes in current frame: ", len(hot_box_list))
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    global number_of_boxes
    number_of_boxes.append(len(hot_box_list))
    global hot_box_list_history
    hot_box_list_history = hot_box_list_history + hot_box_list


    if len(number_of_boxes) > frames_per_hot_boxes:
        for counter in range(number_of_boxes[0]):
            hot_box_list_history.pop(0)
        number_of_boxes.pop(0)
    #print ("Number of frames in memory with boxes: ", len(number_of_boxes))

    #print ("Total number of boxes of the last", frames_per_hot_boxes, "frames : ", len(hot_box_list_history))
    # Add heat to each box in box list
    #heat = add_heat(heat,hot_box_list)
    heat = add_heat(heat,hot_box_list_history)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_boxes(np.copy(image), hot_box_list_history, color=(0, 0, 255), thick=3)
    draw_img = draw_labeled_bboxes(draw_img, labels)



    return draw_img

print('Processing the video...')

out_dir='./output_images/'
input_file='project_video.mp4'
#input_file='test_video.mp4'
output_file=out_dir+'processed_'+input_file

clip = VideoFileClip(input_file)
out_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
out_clip.write_videofile(output_file, audio=False)


print("Done")
