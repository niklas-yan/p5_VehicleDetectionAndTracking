# Vehicle Detection and Tracking
### Yulong Li  
---
## Some Words at Beginning  

I know the main goal of this project is using the HOG features approach to train a classifier. However, I had been working on this approach for the whole week and couldn't get a good result. I tried different combanitions of parameters including 'orient', 'pix_per_cell', color space and etc., but none of them gave me a clean and accurate pridiction on test images, even though I always got 98% to 99% test accuracy when training the classifier. I also suspect that the Hog Sub-sampling Window Search method may distort the window image, so the window feature may not be very clean. I even tried suggestions mentioned in the forum to use confidence score over 99% confidence and played around with the heat map, but still getting too many false positives and partially detected vehicles.  

So eventually, I decided to switch to CNN approach in this project, because I think the performance of CNN is much better than HOG - fewer false positives and more accurate predictions.  

Anyway, I still attached the codes for the HOG approach in this repo: [p5_HOG_SVM.py](https://github.com/yulongl/p5_VehicleDetectionAndTracking/blob/master/p5_HOG_SVM.py) trains the Linear SVC and [p5_HOG_test.py](https://github.com/yulongl/p5_VehicleDetectionAndTracking/blob/master/p5_HOG_test.py) implements Hog Sub-sampling Window Search method to test on sample frames. It'll be great if reviewer could help me to figure out how to improve the codes to get better results.  

**This project writeup will introduce the approch of CNN with Keras, instead of HOG feature extraction.**  

---
### 1. Training and Testing Data  

The given GTI and KITTI datasets and some extractions from the project video are used for vehicle class. The given 'Extras' dataset is used for nonvehicle class. Traning and testing data are split by the ratio of 0.8:0.2.

### 2. CNN Model Architecture

CNN Model Architecture:  
![model_1](https://github.com/yulongl/p5_VehicleDetectionAndTracking/blob/master/images/model_1.png)  

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

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

