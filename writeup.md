
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Features
First I will talk about the kinds of features that can be extracted from an image for use in a machine learning algorithm.  There are three basic types, color histograms, spatial binning, and histogram of oriented gradients.

### Feature Extraction - Color Histograms
The first of three types of feature extraction was a color histogram.  This type of feature extraction counts the number of pixels falling within a certain range, or bin.  This is done for each of the 3 channels.  Below are the RGB histograms for an image of a car, a section of roag, a tree, and some sky.  You can see that the car histograms are more complex and should be easy to delineate from the other three in a machine learning process.  The code generating these images is in the jupyter notebook vehicle_detection.ipynb in the code cell with the same title as this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_cutout1.jpg" />
<br />
<b>Car Image RGB Color Histogram</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_road.jpg" />
<br />
<b>Road Image RGB COlor Histogram</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_tree.jpg" />
<br />
<b>Tree Image RGB Color Histogram</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_sky.jpg" />
<br />
<b>Sky Image RGB Color Histogram</b>
</p>
<br />

### Feature Extraction - Spatial Binning
Spatial binning is an attempt to use raw pixel values as a way to determine if an image is a vehicle or not.  The image is typically reduced in size to xomething manageable like 32X32.  Otherwise, the feature set can become quite long.  Even with 32X32 image and 3 channels, this yields 32X32X3=3072 features!  Below are the plots of the pixel values for a 32X32 vewrsion of the same images used above.  Again the car graph is quite different from the other 3 and should be useful for a machine learning approach.  The code that generated these images is in the jupyter notebook vehicle_detection.ipynb in the code cell with the same title as the this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_cutout1.jpg" />
<br />
<b>Car Image Spatial Bin</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_road.jpg" />
<br />
<b>Road Image Spatial Bin</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_tree.jpg" />
<br />
<b>Tree Image Spatial Bin</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_sky.jpg" />
<br />
<b>Sky Image Spatial Bin</b>
</p>
<br />

### Feature Extraction - Histogram of Oriented Gradients (HOG)
The last type of feature extraction looks at the gradient of the pixels - the direction of change of pixel values. 
Each of the figures below shows an original image and its histogram of gradients.  Again, the vehicle HOG is quite different from the other three and a good candidate for machine learning.  The code that generated these images is in the jupyter notebook vehicle_detection.ipynb in the code cell with the same title as the this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_cutout2.jpg" />
<br />
<b>Car Image HOG</b>
</p>
<br />

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_road.jpg" />
<br />
<b>Road Image HOG</b>
</p>
<br />

<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_tree.jpg" />
<br />
<b>Tree Image HOG</b>
</p>
<br />


## Window Search
In developing a search strategy, first off, I don't want to search  anywhere cars are not likely to appear.  So obviously, avoid the sky, trees, buildings, etc.  Also, I opted for keeping the search simple.  Not too many offsets, etc.  The window search development code is in the jupyter notebook vehicle_detection.ipynb in the code cell with the same title as the this section.

#### Y Axis Search
After reviewing several frames from the video, I decided to start Y axis searching at 400 - elimnating sky, trees, etc.  The larger the window dimension, the farther down the Y axis I went.  I did not see a need to conduct the search all the way to the bottom of the image.

#### X Axis Search 
I chose not to narrow the search along the X axis.  This is because the road can swerve right or left.  Also, as in the project video, the car is in the left most lane of 3 lanes and cars in the right most lane.  In such cases, cars can appear at the far edges of the image.  There are clear examples of this in the figures below.

#### Other Considerations 
A final consideration in window search is computation costs.  The more windows, the longer it takes to process a frame.  So any additional windows in the search need to provide a definite benefit for the cost.  Several attempts at adding windows, new dimensions, etc. did not yield any significant increase in results.  

My final solution (although tweaked later) is as folows

| Y start | Y stop | Window Size | Overlap |
|-----|-----|-----|-----|
| 400 | 496 | 64 X 64 | 0.5 |
| 416 | 560 | 96 X 96 | 0.5 |
| 432 | 624 | 128 X 128 | 0.5 |

This is illustrated in the figures below.  The code that generated these images is in the jupyter notebook lane_detection.ipynb in the code cell with the same title as the this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test1_64.jpg"  width="320"/>
<br />
<b>Search Pattern Window Size 64X64</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test1_96.jpg"  width="320"/>
<br />
<b>Search Pattern Window Size 96X96</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test1_128.jpg"  width="320"/>
<br />
<b>Search Pattern Window Size 128X128</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test1.jpg"  width="320"/>
<br />
<b>Full Search Pattern</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test13.jpg" width="320"/>
<br />
<b>Full Search Pattern for Another Frame</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/windows_test5.jpg"  width="320"/>
<br />
<b>Yet Another Frame 0.75 Overlap</b>
</p>
<br />

## Data

I had 2 kinds of test data.  Single frames for initial testing and training data for training the model.

#### Test Data 
For single frame test data, I opted to use actual frames from the project video.  Six were provided in the project repository.  I generated 7 others from different points in the video.  These were based on trouble spots I experienced in my previous attempt at this project.  These images were used in developing the window search and parameter tuning for the model.  The code I used to generate these additional frames is not in the vehicle_detection.ipynb.

Here are 2 of the frames I generated and used for initial testing.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/test7.jpg"  width="320"/>
<br />
<b>Example Frame from Video Used for Development</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/test11.jpg"  width="320"/>
<br />
<b>Another Example Frame from Video Used for Development</b>
</p>
<br />

#### Training Data
For training data, I used all png images of cars and non cars provided by the project repository.  These are sourced from GTI and KITTI.  This provided 8792 examples of cars and 8968 examples of non cars.  The code that I used to read in all of the image names is in the jupyter notebook lane_detection.ipynb in the code cell with the same title as the this section. Some examples of these are shown below. 

| Label | Example Images |
|-------|-----|
| Cars | <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/202.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/21.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/269.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/36.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0037.png"  />  <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0078.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0172.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0236.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0353.png"  /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/vehicles/image0361.png"  />|
| Non Cars | <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra124.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra133.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra136.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra14.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra250.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/extra40.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/image26.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/image764.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/image820.png" /> <img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/training_images_sample/non-vehicles/image911.png" />  |


## Search and Classify

Fitting a model for the pipeline was a 4 step process:  1) selecting feature parameters, 2) extracting feature vectors from the training data using those parameters, 3) fitting a linear SVM model using those feature vectors, 3) applying the trained model to the test images (video frames) and visually inspecting for detections and false positives.  Also, since this process involves a window search, I varied that as well.  Some models fit better under one window size, but not another.  All of the code used for this section is in the jupyter notebook vehicle_detection.ipynb in the code cell with the same title as the this section.

### Parameter Selection

I used all three feature types from above:  color histogram, spatial binning, and HOG.  I had no real intuitions about what parameter values might work so I tweaked most of them.  This exhaustive approach is not good at all since the permutations explode and the cycle time to train the mdoel each time can be time consuming.  I noticed color channel seemed to have the most effect.  I even tried using only HOG features.  Success of these experiments varied quite a bit but no real success.  Whenever I had good detection, I also had several false positives.  Seemed like more than I could reliably filter out.  I then tried the parameter set from Vehicle Detection Lesson, Exercise 35 since this seemed to work well on the test image in that exercise.  This also had a good effect on my test imags so I settled on that.  The final parameter set is as follows:

| Parameter | Value | Explanation |
|-------|-------|
| color_space | YCrCb | Color channels of the image |
| orientations | 9 | Number of HOG orientations |
| pix-per_cell | 8 | HOG pixels per cell |
| cell_per_block | 2 | HOG cells per block |
| hog_channel | ALL | which color channel to use, or ALL |
| spatial_size | (32,32) | size of spatially binned image |
| hist_bins | 32 | number of bins for color histogram |
| spatial_feat | True | use spatial features or not |
| hist_feat | True | use color histogram features or not |
| hog_feat | True | use HOG features or not |


### Fitting the Model

First I extracted the features from the training images using the parameter set above.  This took 90.02 seconds.  Each vector had 8460 features in it.  Then I normalized the vectors so that features with larger magnitude values would not have undue influence.  Then I split the total training feature set into 80% for training, and 20% for testing.  Then I trained a sklearn.svm.LinearSVC with the 80% training data.  This took 53.11 seconds.  I then used the 20% training data to calculate an accuracy of 0.9901.  Finally, I could optionally save the model to file so it could be used later without having to regenerate it.

### Test Images

Since training the model with the full data set is so time consuming, during my experimentation phase, I would only use a sub sample of the data, typically 1000 images.  This reduced my cycle time since the model had to be retrained with each change of the parameters.  Below are some of my test images with the searches resulting from the final model.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/search_classify_test1.jpg"  width="320"/>
<br />
<b>Test Image Cars Detected</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/search_classify_test10.jpg"  width="320"/>
<br />
<b>Test Image Car Detected</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/search_classify_test2.jpg"  width="320"/>
<br />
<b>No False Positives</b>
</p>
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/search_classify_test6.jpg"  width="320"/>
<br />
<b>Two Cars Detected</b>
</p>
<br />

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

## Discussion

My final video does track the cars and all false positives have been removed.  However, the white car is not adequately identified in all cases.  My model is not very good at detecting the white car in some frames, especially those with light colored pavement.  The result is that the bounding box for the white car sometimes clips the full dimensions of the car.  While the car is identified, its centroid is inaccurate and this could lead to poor decisions for my car.

One possible approach to fix this is to fine tuning the window search strategy more thoroughly.  More overlapping searches at differing sizes and offsets might help.  Another might be to augment training data with more white cars - particularly ones taken from video frames.

What I would like to do, however, is abandon the entire approach so far (linear SVM with derived features) and try a neural network instead.  The window search piece can be reused, as well as the training data set.  This would be a very interesting exercise!
 

