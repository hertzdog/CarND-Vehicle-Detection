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

def process_image(image):
    hot_box_list = []
    for i in range(3):
        current_window=np.int(math.pow(2,(5+i)))
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                        xy_window=(current_window, current_window), xy_overlap=(0.5, 0.5))
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        hot_box_list = hot_box_list + hot_windows
    draw_image = draw_boxes(image, hot_box_list, color=(0, 255, 255), thick=6)
    return draw_image

print('Processing the video...')

out_dir='./output_images/'
input_file='project_video.mp4'
output_file=out_dir+'processed_'+input_file

clip = VideoFileClip(input_file)
out_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
out_clip.write_videofile(output_file, audio=False)

import Queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
        print "Starting " + self.name
        process_data(self.name, self.q)
        print "Exiting " + self.name

def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print "%s processing %s" % (threadName, data)
        else:
            queueLock.release()
        time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = ["One", "Two", "Three", "Four", "Five"]
queueLock = threading.Lock()
workQueue = Queue.Queue(10)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()
print "Exiting Main Thread"

print("Done")
