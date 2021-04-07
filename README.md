# W207 Final Project : Facial Keypoint Detection 
# Team: Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter
W207 Tuesday @4pmPT

ref: https://www.kaggle.com/c/facial-keypoints-detection


## Summary

This repo contains work performed by Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter for the [Kaggle Facial Keypoint Detection](https://www.kaggle.com/c/facial-keypoints-detection) challenge as our final project for W207. This notebook contains several different neural networks with the Lenet5 inspired approach yielding the best result of 3.23581 placing at position 72 on the leaderboard

![](https://i.imgur.com/kbpD4Eo.jpg) 


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

This solution acheives a best score of **3.23581** using the Lenet5 inspired model, which places 72 place on the locked leaderboard. 

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
1. "clean_o_outliers" : removing only overlap outliers, which is the set of outliers that are also duplicates
2. "clean_w_outliers" : removing only the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)
3. "clean_all_outliers" : removing all outliers which are all images with any keypoint that is located more than 2 standard deviations away from the mean for that keypoint.
4. "clean_duplicates" : removing all duplicates from the data
5. "clean_o_dups" : removing all duplicates and overlap outliers
6. "clean_w_dups" : removing all duplicates and the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)
7. "clean_wo_dups" : removing all duplicates, overlap outliers, and the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons)


![](https://i.imgur.com/S7FhUkH.jpg)

4. **Training Pipeline**

4.1 **Baseline Modelling**

Though our final goal was to utilize neural networks for predicting the facial keypoints, we decided to first use some simpler machine learning models to develop a baseline, help us develop our final data pipeline and fine tune our EDA and data cleaning processes. As this is a regression problem, we selected several models from the SciKitLearn library. In general, none of these models performed particularly well based upon the mean squared error and $\R^2$. Moreover, they were particularly slow to run. 

4.1.1 **Datasets Used for Baseline Models:**

a. Raw dataset - No modifications, all missing values filled in with mean of respective column.<br />
b. Duplicates removed - All duplicates removed with missing values filled in with the mean of the respective column.<br />
c. Augmented dataset - Utilized a custom-built linear model to predict the value of the missing data points.<br />

4.1.2 **Baseline Model Results**

a. Ordinary Least Squares Regression<br />
b. Ridge Regression<br />
c. Lasso Regression<br />
d. Decision Tree Regression<br />
e. K-Nearest Neighbors Regression (with K=5 and K=7 nearest neighbors)<br />
f. Random Forest Regression (with 5 estimators)<br />

In general, performance varied between models and datasets, with the linear, lasso and ridge regression models, and decision tree regression, giving very poor (negative $\R^2$) performance. Best performance was achieved using K-Nearest Neighbors regression with 7 neighbors and using the raw dataset. This might be because the raw dataset, as opposed to the duplicate-removed dataset gives more training examples. Random forest regression performed slightly worse than the K-Nearest Neighbors models, but still better than the others, with optimal performance given using the augmented dataset. Another interesting finding from this initial modelling process is that angled faces, eyeglasses and childrens faces seemed to pose the biggest predictive challenge for these baseline models.

The notebook for the baseline models can be found [`here`](https://github.com/jcweaver/blackboxes/blob/master/models/initial_models.ipynb)

4.2 **Final Models**

The models that were implemented as part of this challenge are:

|Model| Description |
|:----|:------------|
|[`models/Lenet5_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/Lenet5_Model.ipynb)|A notebook of models inspired by Lenet5.|
|[`models/JCW.Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/JCW_Model.ipynb)|Joanie to fill in.|
|[`models/SP_model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/SP_model.ipynb)|A notebook of a model inspired by Daniel Nouri's approach to this challenge.|



4.2.1 **Model 1 (JN)/LeNet5 Inspired**

This model produces a number of Lenet5 inspired Models and Predictions based on varying degrees of cleaned Train data and augmentation. The inspiration came from the Mediaum article ["Lenet5 in 9 lines of code using Keras"](https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086).
Several layers were added to arrive at the final version of 15 layers following a large amount of testing. It takes as input a 2D input dataset (96,96,1) followed by a 2 sets of [2D convolution layers + rectified linear unit (RELU) + 2DAverage Pooling] x 3 Dense layers.  

The following variations of models were created:

- baseline, no augmentation 
- Approach 1: 7 previously mentioned versions of cleaned train data (clean section) set were used to create models and predictions (best result came from this test)
- Approach 2: 7 previously mentioned versions of cleaned train data (clean section) set were used + varying the layers in the model used to create models and predictions
- Approach 3: 7 previously mentioned versions of cleaned train data (clean section) set were used + varying layers + image augmentation (brightness and dim) to create models and predictions
- Approach 4: 7 previously mentioned versions of cleaned train data (clean section) set were used + varying layers + image augmentation (horizontal flip) to create models and predictions

All training for all models was fixed at a 128 batch size, 300 epochs with a patience set at 30 (with early stop set). Adam optimization was used with an initial learning rate of 0.01 and later changed to 0.001, beta of 0.9, beta2 of 0.999 and epsilon=1e-8 (following standards and examples)

`act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)`

`lss = 'mean_squared_error'`

The best result was from approach 2 from the model using the clean data with all outliers removed yielding: 3.23581  

Placing at position 72 on the leaderboard

![](https://i.imgur.com/kbpD4Eo.jpg)

The best performing model plot can be see below: 

![](https://i.imgur.com/ltAzPj6.png)


4.2.2 **Model 2 (JCW)**

4.2.4 **Model 3 (SP)**

This model was based upon a blog post entitled ["Achieving Top 23% in Kaggle's Facial Keypoints Detection with Keras + Tensorflow"](https://fairyonice.github.io/achieving-top-23-in-kaggles-facial-keypoints-detection-with-keras-tensorflow.html), by Shinya Yuki, which itself is an adaptation of [Daniel Nouri's approach](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) to this challenge using the now deprecated Lasagne package for CNN's. The original package was released prior to the release of the Keras library, so Yuki's version represents an update. The model was adapted to take data that had been pre-processed using our EDA and data cleaning pipeline. 

The model is essentially the same, taking a 4D input dataset (1, 96, 96, 1) and has 3 x 2D convolution layers with 32-64-128 filters (doubles each layer). Each convolutional layer is followed by a 2x2 max-pooling layer and there are 2 fully-connected layers. The densely-connected layers have 500 units each. The model uses rectified linear unit (‘ReLU’) activation in each layer, a Nesterov-accelerated gradient descent (SGD) optimizer with a learning rate of 0.01 and a momentum parameter of 0.9. The model was trained using batches of 128 examples, and for 300 epochs. The only way the model was modified from the examples was that for simplicity, the dropout layer functionality was omitted.

This model achieved modest perfomance in terms of the metrics of interest, and performed best using the cleaned dataset with overlapping outliers (Kaggle score 4.15) and the cleaned dataset with duplicates (Kaggle score 4.33), which is ~150th position on the leaderboard. Augmented data did not improve its performance past the scores listed. 

The model plot can be seen below:

![](https://github.com/jcweaver/blackboxes/blob/master/images/Model%20SP/model_flow.png)

* TBD
* TBD
* TBD
* TBD
6. Inference Pipeline






## Navigating the Files in this Repository

Below is a list of files found in this repository along with a brief description.

|File | Description |
|:----|:------------|
|[`EDA/EDA_Final.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/EDA_Final.ipynb)|A EDA notebook that describes in detail the steps taken during the EDA phase.|
|[`EDA/Data_Clean`](https://github.com/jcweaver/blackboxes/blob/master/Data_Clean.ipynb)|A notebook to clean the data a number of different ways.|
|[`EDA/Augment_Missing_Data`](https://github.com/jcweaver/blackboxes/blob/master/EDA/Augment_Missing_Data.ipynb)|A notebook to try a linear approach to fill in missing data.|
|[`models/initial_models.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/initial_models.ipynb)| TBD|

