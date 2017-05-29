# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carNonCarHOG.png
[image2]: ./output_images/HOGparams.png
[image3]: ./output_images/HOGparams1.png
[image4]: ./output_images/HOGparams2.png
[image5]: ./output_images/HOGparams3.png
[image6]: ./output_images/allWindows.png
[image7]: ./output_images/testResult.png
[image8]: ./output_images/testOptimized.png
[image9]: ./output_images/processed_video.gif
[image10]: ./output_images/videopipeline.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **HOG feature parameters** section of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes and their HOG features visualized with cmap 'hot':

![alt text][image1]

Here get_hog_features defined in the lession gets called to output the visualization. One can see that the contour of the car back is well captured whereas for non car image, there were only few long arms of gradient magnitude; so HOG can be used as a good differentiator for the classification problem.

#### 2. Explain how you settled on your final choice of HOG parameters.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

The visualization shows the pix per cell @ 8 and 16 with 6 and 9 orients respectively. All we can visually is that 9 orients does a better job in getting a dominant arm in the orientation bins. To better decide the HOG parameters I did some experiments to explore number of features generated for different parameters:




```
HOG dimension:  (7, 7, 2, 2, 6)
HOG features: (1176,)
distance of car and non car HOG feature vectors: 2.38234371541
HOG dimension:  (8, 8, 1, 1, 6)
HOG features: (384,)
distance of car and non car HOG feature vectors: 3.76111336373
HOG dimension:  (7, 7, 2, 2, 9)
HOG features: (1764,)
distance of car and non car HOG feature vectors: 2.40785095273
HOG dimension:  (8, 8, 1, 1, 9)
HOG features: (576,)
distance of car and non car HOG feature vectors: 3.6547053891
HOG dimension:  (3, 3, 2, 2, 6)
HOG features: (216,)
distance of car and non car HOG feature vectors: 0.79194006214
HOG dimension:  (4, 4, 1, 1, 6)
HOG features: (96,)
distance of car and non car HOG feature vectors: 1.41870932594
HOG dimension:  (3, 3, 2, 2, 9)
HOG features: (324,)
distance of car and non car HOG feature vectors: 0.794630196639
HOG dimension:  (4, 4, 1, 1, 9)
HOG features: (144,)
distance of car and non car HOG feature vectors: 1.48928671611
```

So we can conclude that the length of the raveled feature vector is the product of orients, blocks per dimension, i.e (# of cells per dimension - cells per block + 1) and cells per block; in our case the pix per cell and cells per block are the same for x and y.

As the pix per cell grows, the cells per dimension becomes less. From the comparison we can see that the euclidien distance between car and noncar feature vector grows as the cells per dimension grows.

the change in # orients doesn't change the distance much.

So I chose pix per cell as 8 and cells per block as 2. cells per block 2 means each pixel in the cell has 2 votes in 2 different blocks each. this makes the HOG feature more generalized to variations in gradients. 64/8 - 2 + 1 = 7 blocks per axis, this gives us a feature vector of 7 x 7 x 2 x 2 x 6 = 1176, which is relatively long so that it differentiates the car and non car well while it doesn't introduce too much computation. For number of orients, I started with 6 but later on raised it to 9 as the [HOG paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwjf3Nrl05PUAhVYwGMKHSB5CeYQFgguMAE&url=http%3A%2F%2Fvc.cs.nthu.edu.tw%2Fhome%2Fpaper%2Fcodfiles%2Fhkchiu%2F201205170946%2FHistograms%2520of%2520Oriented%2520Gradients%2520for%2520Human%2520Detection.pdf&usg=AFQjCNHV8aVX7YHcipUtIK4Qe9PeHNhWxw) suggest 9 orientation bins work the best with SVM for object detection tasks.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM classifier from all the data provided: midclose, far, left, right, GTI and KITTI; also I used the StandardScalar from sklearn to help normalize the data. From initial data set exploration, see section **Generate a list of images to read in**:

the car and non car data numbers are close and they both have more than 8k images. So there is no need to augment the dataset for now. Given the large number of inputs. So it may take a long time to try various combinations of parameters on the entire dataset; the way I evaluate a parameter combination was that I run it on a 1000 randomly selected images training dataset with 80% for training and 20% for validation. 


For features, all the 3 types of features are employed, i.e. color bin spatial, histogram and HOG to capture as much spatial, color and gradient information as possible. I started with RGB channel 0 with HOG parameters: 6 orients, 8 pix per cell and 2 cells per block. that gave me 0.94 validation accuracy. which is not good enough as 6% percent of false positive translates to a lot of wrong windows when the video is long enough. So I did some exploration for different color spaces and HOG features. My best results came after I adjusted the color channel to YCrCb and number of orients in HOG to 9; the accuracy rose to 0.9913. Details see section: **Train the model, tweak feature selection to optimize validation accuracy**.



```
85.50593495368958 seconds to complete feature extraction
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8412
6.93 Seconds to train SVC...
Test Accuracy of SVC =  0.9913
My SVC predicts:  [ 0.  0.  0.  0.  0.  0.  1.  0.  1.  1.]
For these 10 labels:  [ 0.  0.  0.  0.  0.  0.  1.  0.  1.  1.]
0.00179 Seconds to predict 10 labels with SVC
```

To find out the best parameters, sklearn has GridSearch we can use to cover the search for the optimal parameters. Here I did some online research, I found this article with some more feature extraction schema tried by Mohan [here](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10);


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Please see the section **Test the classifier on test images, tweak parameters to optimize car detection**: since the training dataset has far and midclose images so in order to capture the similar sub image I used 3 sliding window size both all at 0.5 overlapping. The result in the next section shows it worked pretty well to capture all the cars while only introduce little false positives.
```
ystart: 400, we only need to check the part below the tree top.
yend: 720 - 64 = 656, that is a smallest window high to the bottom.
R: xy_window=(64, 64), xy_overlap=(0.5, 0.5)
G: xy_window=(96, 96), xy_overlap=(0.5, 0.5)
B: xy_window=(128, 128), xy_overlap=(0.5, 0.5)
```

![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are the test results on the test_images/*

![alt text][image7]

In previous section, it took close to 2 seconds to go through 430 windows in a single frame, which is not viable for processing a 1.2k frames video later.

So I need to optimize the computation, I used the method mentioned in the lessson which instead of computing HOG for each subimage and then change the sliding windows size, we can scale the image first and compute HOG on the entire image once then only do a subsampling when it comes to each window's HOG. Note that this method is not exact the same as previous flow since the gradient of the pixels on the window boundary differs in these two scenario. but the result should not differ that much given the fact that HOG takes the dominance after bucketizing the weighted gradients by orientation. See the details in section **Optimize the computation by scaling + HOG entire image once + subsampling**.

classication time before and after:

```
before:
1.9314320087432861 seconds to process one image searching 430 windows
1.9900920391082764 seconds to process one image searching 430 windows
1.9429800510406494 seconds to process one image searching 430 windows
1.7981958389282227 seconds to process one image searching 430 windows
1.7815701961517334 seconds to process one image searching 430 windows
1.7474641799926758 seconds to process one image searching 430 windows

after:
0.39569687843322754 seconds to process 294 windows
0.5054891109466553 seconds to process 294 windows
0.36809301376342773 seconds to process 294 windows
0.4032270908355713 seconds to process 294 windows
0.35880517959594727 seconds to process 294 windows
0.354233980178833 seconds to process 294 windows
```

with this, I set the 'cells per step' to 2 which equals to 0.75 overlap for the 8 cells per axis to enjoy the speed up.

From the test result, the accuracy of classifier remains the same:
![alt text][image8]



---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's my processed video:

![alt text][image9]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Heatmap is used to record the likelyhood of a subimage contains the car. The more detected windows laid on one pixel, the higher the heat. After this we can use this heat map to filter out false positives by setting heat threshold. so that those detection with thin coverage will be ruled out. After that `scipy.ndimage.measurements.label()` is called to identify individual clusters of detections, the labels returned will contain the nubmer of clusters and their bounding boxes with which we can use the draw the rectangles around the car.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the test frames:

![alt text][image10]

I also did some more precedure to filter out false positives in video pipeline;

* I didn't choose the searching in the margin method used in lane finding. Because I don't have the confidence to handle the case where two more cars are close by. Depending on the speed and distance, it is hard to come up with a good searching criteria.
* using different scales, i.e window sizes for each frame, I chose [2.0, 1.5, 0.8] to capture cars close by and relatively far away. With this I get more windows detected that can help me figure out false positives by setting threshold.

* look back to the previous 4 frames and accumlate the heat maps. This further reduces the possibility of false positives. Note that one should not use a very deep buffer as it hurts the agility of the bounding box. i.e. the box remains for a longer time when the car goes away.

* there are some false positives introduce by the scale 0.8 which is a smaller window size which is prone to wrong detection. So I also added some logic in the ```draw_labeled_bboxes``` function to avoid drawing those small boxes which apparently cannot hold a car.

* all this false positive removal takes an trial and error approach; but the time it takes to process the video is not trivial. so what I did was the cut the project video into shorter pieces and adding more stats so that I can isolate the problem and debug and try new threshold / buffer size / scale combinations quickly.

So in the final processed video there are minimal false positives included, some of them are actually incoming vehicles on the oppsite direction.

Please details in section **Video pipeline**



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are several challenges in the project:

* the search for optimal: there are so many variables to tweak: different types of features, parameters for each type of feature; like the lesson mentioned, traditional computer vision methods give those who love tweaking feature selection and parameters the chance to understand the connection between a feature and overall model performance. Fortunately this training time in this dataset and the SVC model we use are still tweakable in limited time so I got the chance to try different combos. Otherwise GridSearch should be the way to go.

* the processing time: even after optimizing the HOG feature calculation, the processing time for the video file is considerable. since I will need the previous N frames information so going muliple thread to process the video is not that easy. this makes the fight to false positives in video pipeline a time consuming process. So I had to divide the video into smaller chunks. I should have compiled a GPU version of opencv if time allows..

The current pipeline will still pick up some cars seen in the opposite traffic. Also it stops tracking the car when it goes beyond tree top on y axis; To make it more robust, I will need more negative tranining example for tree background, car front etc. Also the training should focus on smaller window sizes.

---

### 2nd Submission

Thanks to a great reviewer. I've made the following changes inspired by the review comments:

* instead of svc.predict, decision_function is used along with the threshold to better select only detection with high confidence; this actually removed the majority of the false positives if threshold set properly.

* limit the scan range to right half of the screen. as in this video the car is driving on the inner most lane so it is ok to make this assumption that all the car in the same traffic direction are found in the right half of the screen.

* tried opencv HOGDescriptor; since the compute() method of this descriptor requires 8 bit integer input which is the default format for jpg; I will need to figure out a way to use it in both the png training data and jpg test data; since I already used the HOG once + subsampling method to optimize the classifier performance and the time spent on HOG is not the majority; so I didn't spend too much time on this. It turns out avoiding unnecessary window scan is the best way to accelerate the video pipeline. I got 4-5x speed up after reducing the scan area size and choose window size based on distance.

* use different window size based on the x coordinates. this is a great insight from the reviewer, that car size tend to become smaller when it approaches the center of the image. I adopted it and saved lots of computation, improved the detection accuracy without introducing any false positives. Now the solution uses [1.0 1.2 1.4] scale for 400 pixels on x axis with 200 pixels overlapping. this ensures smooth transition of window size.

* fixed the false negative issue in the video. Now the video has a very clean and stable output.

