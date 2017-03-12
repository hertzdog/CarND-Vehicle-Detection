# Vehicle Detection

The Project
---
The project is available at: https://github.com/hertzdog/CarND-Vehicle-Detection and is part of the Udacity Autonomous Veichle Driving Nano Degree course in which I am involved in.

The final movie is here: 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

In the file *lesson_functions.py* I defined the function *get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)* which computes the hog features of an image, using several parameters, as was explained during the lessons.

* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.

I also added the color transformation, using *color_space = 'YCrCb'* and *color_space = 'HLS'* as color spaces: the former was giving best results. A lot of effort was done in analyzing colours, and a python notebook called [color-spaces-detection.ipnyb](https://github.com/hertzdog/Detection-tracking-color/blob/master/color-spaces-detection.ipynb) was made for exploring color-spaces.

Binned color features have been included.

* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

The training has been done in train_model.py: features have been normalized and the selextion for training and testing has been randomized. I trained the linear SVM using all channels of images converted to YCrCb space. I included spatial features color features as well as all three YCrCb channels. The final feature vector has a length of 6156 elements. For color binning patches of spatial_size=(16,16) were generated and color histograms were implemented using hist_bins=32.

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

The sliding windows has been implemented, using windows of different sizes: 32, 64 and 128 and spanning only the image part where cars can be found, because of the different size of veichles on the road. The result has been passed in a heatmap and the final box was done by the label function.
[image1]: ./output_images/figure_1_YCrCb.png
[image2]: ./output_images/figure_2_YCrCb.png

Even if test images seems good, the application on video was poor, especially for detecting the white car.

* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

Working on video, the following change was made:
1. Define the number of frames on which perform the heat map: started with one, I was "satisfied" with 6
2. Take into account boxes of the last 6 frames for bulding the heatmap, with a threshold of 5



* Estimate a bounding box for vehicles detected.

In the output directory there are many videos on which I did tests. [video](./output_images/YCrCb_processed_project_video_YCrCb_15_2.mp4)



The video shows the boxes detected in the frame (blue line) and the result of the heatmpab and labeling.
Even if I tested a lot of color combination, thresholds, frames I was not satisfied.

Then I decided to apply the [Darknet Yolo Project](https://pjreddie.com/darknet/yolo/).
I found a TensorFlow implementation of it.
Starting from there I designed an algorithm for detecting cars using the Convolutional Neural Network already trained: the approach was faster (4x, even on my laptop) and giving more satisfying results. Specifically I set the threshold at 0.3 and added a heatmap in order to have only one box per car.

For running the project I had to create a new Conda Environment with TensorFlow 1.0 and all related packages. At the end I was very excited about this final project: I started following the guidelines provided, exploring color-spaces, and at the end I was able to create a new conda Environment and succesfully using a very fast algroithm on my project!

Here is my final project on github (https://github.com/hertzdog/darkflow) which was used for generating the video otput.
