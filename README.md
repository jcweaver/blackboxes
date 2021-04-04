# W207 Final Project : Facial Keypoint Detection 
# Team: Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter
W207 Tuesday @4pmPT

ref: https://www.kaggle.com/c/facial-keypoints-detection


## Summary

This repo contains work performed by Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter for the [Kaggle Facial Keypoint Detection](https://www.kaggle.com/c/facial-keypoints-detection) challenge as our final project for W207. This notebook contains several different neural networks with the XX approach yielding the best result of XX.   


### The Challenge

The objective of this task is to predict keypoint positions such as nose_tip and right_eye_center on face images. This can be used as a building block in several applications, such as:

* tracking faces in images and video
* analysing facial expressions
* detecting dysmorphic facial signs for medical diagnosis
* biometrics / face recognition

Detecting facial keypoints is a very challenging problem.  Facial features vary greatly from one individual to another, and even for a single individual, there is a large amount of variation due to 3D pose, size, position, viewing angle, and illumination conditions. Computer vision research has come a long way in addressing these difficulties, but there remains many opportunities for improvement.

## Performance Criteria

The Kaggle facial detection challenge asks participants to identify the (x,y) coordinates for up to 15 facial keypoints on a given Test dataset that contains 1783 images. The images are sized 96x96x1 and are in grayscale.  Each facial point must be submitted as their individual (x,y) coordinates for scoring which equates to up to 30 values (2x15) for an image that contains all 15 facial keypoints. 

### Performance

This solution acheives a best score of **TBD RMSE** using the model, which places X place on the locked leaderboard. 

## Project Approach
To reach the final output of the project and to support the summary findings above, TBD

### Completed Tasks

For this project the team performed the following tasks 

1. **Getting ready for Project 4!**
- Files used in Project 4 - go through the files that are used in Project 4.  
  * training.csv - Train file
  * test.csv - TBD
  * IdlookupTable.csv - TBD
  * SampleSubmission - TBD

2. **EDA**

To begin our project, we spent time exploring our data through 4 main ways: 
1. Loading the data to become familiar with the structure and contents
2. Identifying duplicate images and exploring potential approaches to handling
3. Identifying and counting the amount of missing data fields and exploring potential approaches to handling.
4. Identifying, comparing, and exploring potential approaches to handling outliers.

We ultimately decided to try many approaches to removing different sets of outliers and duplicates and try our final models on all of these clean sets.

3. **Clean Data**

As we decided to try our models on several versions of clean data with different approaches to dropping outliers and/or duplicates, we generally used the following clean data files on our models:
a. "clean_o_outliers" : removing only overlap outliers, which is the set of outliers that are also duplicates
b. "clean_w_outliers" : removing only the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)
c. "clean_all_outliers" : removing all outliers which are all images with any keypoint that is located more than 2 standard deviations away from the mean for that keypoint.
d. "clean_duplicates" : removing all duplicates from the data
e. "clean_o_dups" : removing all duplicates and overlap outliers
f. "clean_w_dups" : removing all duplicates and the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)
g. "clean_wo_dups" : removing all duplicates, overlap outliers, and the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)

4. **Training Pipeline**

4.1 **Baseline Modelling**

Though our final goal was to utilize neural networks for predicting the facial keypoints, we decided to first use some simpler machine learning models to develop a baseline, help us develop our final data pipeline and fine tune our EDA and data cleaning processes. As this is a regression problem, we selected several models from the SciKitLearn library. In general, none of these models performed particularly well based upon the mean squared error and $\R^2$. Moreover, they were particularly slow to run. 

**Datasets Used:**

a. Raw dataset - No modifications, all missing values filled in with mean of respective column.<br />
b. Duplicates removed - All duplicates removed with missing values filled in with the mean of the respective column.<br />
c. Augmented dataset - Utilized a custom-built linear model to predict the value of the missing data points.<br />

**Models**

a. Ordinary Least Squares Regression<br />
b. Ridge Regression<br />
c. Lasso Regression<br />
d. Decision Tree Regression<br />
e. K-Nearest Neighbors Regression (with K=5 and K=7 nearest neighbors)<br />
f. Random Forest Regression (with 5 estimators)<br />

In general, performance varied between models and datasets, with the linear, lasso and ridge regression models, and decision tree regression, giving very poor (negative $\R^2$) performance. Best performance was achieved using K-Nearest Neighbors regression with 7 neighbors and using the raw dataset. This might be because the raw dataset, as opposed to the duplicate-removed dataset gives more training examples. Random forest regression performed slightly worse than the K-Nearest Neighbors models, but still better than the others, with optimal performance given using the augmented dataset. Another interesting finding from this initial modelling process is that angled faces, eyeglasses and childrens faces seemed to pose the biggest predictive challenge for these baseline models.

* TBD
* TBD
* TBD
* TBD
6. Inference Pipeline
7. Stacking??

The models that were implemented as part of this challenge are:

## Models

### LeNet5




 



## Navigating the Files in this Repository

Below is a list of files found in this repository along with a brief description.

|File | Description |
|:----|:------------|
|[`EDA/EDA_Final.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/EDA_Final.ipynb)|A EDA notebook that describes in detail the steps taken during the EDA phase.|
|[`EDA/Data_Clean`](https://github.com/jcweaver/blackboxes/blob/master/Data_Clean.ipynb)|A notebook to clean the data a number of different ways.|
|[`EDA/Augment_Missing_Data`](https://github.com/jcweaver/blackboxes/blob/master/EDA/Augment_Missing_Data.ipynb)|A notebook to try a linear approach to fill in missing data.|
|[`models/initial_models.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/initial_models.ipynb)| TBD|

