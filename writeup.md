
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
The first of three types of feature extraction was a color histogram.  This type of feature extraction counts the number of pixels falling within a certain range, or bin.  This is done for each of the 3 channels.  Below are the RGB histograms for an image of a car, a section of roag, a tree, and some sky.  You can see that the car histograms are more complex and should be easy to delineate from the other three in a machine learning process.  The code generating these images is in the jupyter notebook lane_detection.ipynb in the code cell with the same title as this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_cutout1.jpg" />
<br />
<b>Car Image RGB Color Histogram</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_road.jpg" />
<br />
<b>Road Image RGB COlor Histogram</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_tree.jpg" />
<br />
<b>Tree Image RGB Color Histogram</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hist_rgb_sky.jpg" />
<br />
<b>Sky Image RGB Color Histogram</b>
</p>
<br />

### Feature Extraction - Spatial Binning
Spatial binning is an attempt to use raw pixel values as a way to determine if an image is a vehicle or not.  The image is typically reduced in size to xomething manageable like 32X32.  Otherwise, the feature set can become quite long.  Even with 32X32 image and 3 channels, this yields 32X32X3=3072 features!  Below are the plots of the pixel values for a 32X32 vewrsion of the same images used above.  Again the car graph is quite different from the other 3 and should be useful for a machine learning approach.  The code that generated these images is in the jupyter notebook lane_detection.ipynb in the code cell with the same title as the this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_cutout1.jpg" />
<br />
<b>Car Image Spatial Bin</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_road.jpg" />
<br />
<b>Road Image Spatial Bin</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_tree.jpg" />
<br />
<b>Tree Image Spatial Bin</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/spat_bin_sky.jpg" />
<br />
<b>Sky Image Spatial Bin</b>
</p>
<br />

### Feature Extraction - Histogram of Oriented Gradients (HOG)
The last type of feature extraction looks at the gradient of the pixels - the direction of change of pixel values. 
Each of the figures below shows an original image and its histogram of gradients.  Again, the vehicle HOG is quite different from the other three and a good candidate for machine learning.  The code that generated these images is in the jupyter notebook lane_detection.ipynb in the code cell with the same title as the this section.

<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_cutout2.jpg" />
<br />
<b>Car Image HOG</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_road.jpg" />
<br />
<b>Road Image HOG</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_tree.jpg" />
<br />
<b>Tree Image HOG</b>
</p>
<br />
<br />
<p align="center">
<img src="https://github.com/TheOnceAndFutureSmalltalker/vehicle_detection/blob/master/out_images/hog_sky.jpg" />
<br />
<b>Sky Image HOG</b>
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

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

