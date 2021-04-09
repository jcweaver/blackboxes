# W207 Final Project : Facial Keypoint Detection 
# Team: Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter
W207 Tuesday @4pmPT

ref: https://www.kaggle.com/c/facial-keypoints-detection


## Summary

This repo contains work performed by Joanie Weaver, Sandip Panesar, Jackie Nichols, Rakesh Walisheter for the [Kaggle Facial Keypoint Detection](https://www.kaggle.com/c/facial-keypoints-detection) challenge as our final project for W207. This notebook contains several different neural networks with the Lenet5 inspired approach yielding the best result of **2.48797** placing at position 51 on the leaderboard


![](https://i.imgur.com/BiDsWBP.jpg) 


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

This solution acheives a best score of **2.48797** using the Lenet5 inspired model, which places 51 place on the locked leaderboard. 

## Project Approach
To reach the final output of the project and to support the summary findings above, we decided to each work on a model that inspired us.  A framework was developed to make creating models and predictions easier for the team. 

### Completed Tasks

For this project the team performed the following tasks 

1. **Getting ready for Project 4!**
- Files used in Project 4 - go through the files that are used in Project 4.  
  * training.csv - list of training 7049 images. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.
  * test.csv - list of 1783 test images. Each row contains ImageId and image data as row-ordered list of pixels
  * IdlookupTable.csv - list of 27124 keypoints Each row contains RowId, ImageId, FeatureName, Location
  * SampleSubmission - list of 27124 keypoints to predict. Each row contains a RowId, ImageId, FeatureName, Location.

 

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

For our final models we applied two different approaches for preparing the data. 

The first approach simply uses all 30 keypoints, vairious versions of clean data, applies any augmentation, then creates, trains and fits a model before predictions are completed and saved to a CSV file. 

![](https://i.imgur.com/jlxPolW.png)

The second approach involves using all 30 keypoints in the same fashion as above, but also doing the same for 8 keypoints. The predictions from the 30 keypoints and the 8 keypoints are then combined into a single prediction CSV per clean train file . For example, if you run this process on the 7 train pickle files, you will have 

- 7 prediction CSV files (1 for each pickle file) for the 30 keypoints
- 7 prediction CSV files (1 for each pickle file) for the 8 keypoints

These prediction files are then combined resulting in 7 combined CSV files

![](https://i.imgur.com/1UUDUSy.jpg)

The models that were implemented as part of this challenge are:

|Model| Description |
|:----|:------------|
|[`models/Lenet5_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/Lenet5_Model.ipynb)|A notebook of models inspired by Lenet5.|
|[`models/JCW.Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/JCW_Model.ipynb)|A notebook of a model inspired by Sinya Yuki's approach.|
|[`models/SP_model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/SP_model.ipynb)|A notebook of a model inspired by Daniel Nouri's approach to this challenge.|

Note: Each notebook above contains output after each cell making the files quite large. If you'd like to view the model files without output to simple view the code please navigate to:

|Model (No Output)| Description |
|:----------------|:------------|
|[`models_no_output/Lenet5_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/Lenet5_Model.ipynb)|A notebook of models inspired by Lenet5.|
|[`models_no_output/JCW.Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/JCW_Model.ipynb)|A notebook of a model inspired by Sinya Yuki's approach.|
|[`models_no_output/SP_model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/SP_model.ipynb)|A notebook of a model inspired by Daniel Nouri's approach to this challenge.|



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

The best result was from approach 2 from the model using the clean data with all outliers removed yielding: 2.48797  

Placing at position 51 on the leaderboard

![](https://i.imgur.com/BiDsWBP.jpg)

The best performing model plot can be see below: 

![](https://i.imgur.com/ltAzPj6.png)


4.2.2 **Model 2 (JCW)**

This model was based upon a blog post originally written in Japanese and I followed the Google Translate English version. The English translated title is ["Implement Kaggle Facial Keypoints Detection in Keras"](https://elix-tech.github.io/ja/2016/06/02/kaggle-facial-keypoints-ja.html#conv).

The model follows the suggested approach, taking a 4D input dataset (1, 96, 96, 1) and has 3 x 2D convolution layers with 32-64-128 filters (doubles each layer). Each convolutional layer is followed by a 2x2 max-pooling layer and there are 2 fully-connected layers. The densely-connected layers have 512 units each. The model uses rectified linear unit (‘ReLU’) activation in each layer, an Adam optimization with a learning rate of 0.001, and a lss using mean squared error.

I tried a few other approaches beyond the base model I described above.

Approach 1: Adding BatchNormalization layers to my model
This did appeared to make all of the result worse

Approach 2: Setting the use_bias parameter in Conv2D to False
There was some success with this but it didn't reach a new high score

Approach 3: Combining 1 & 2
This made predictions worse

Approach 4: Appending flipped images to the data
This appeared to make predictions better and received a new high score.

Approach 5: Brightening all the data
This appeared to make predictions worse

Approach 6: Brightening the data with flipped data appended
There was some success with this but it didn't reach a new high score

Approach 7: Appending the data with flipped images and brightened images

Approach 8: Run a model on data that has all 8 keypoints and output a model that only predicts 8 keypoints. Use this model to make predictions for all of the test cases that only require 8 keypoints and then fill in the missing data for the remaining test cases from a prior predictions file.

The best score this model achieved was when it was run using the training data with the flipped version of the training data appended. The training data file for this was with the set "clean_wo_dups", which was removing all duplicates, overlap outliers, and the worst outliers, which are the 4 mislabelled images and 4 worst images (two collages which are duplicates and two cartoons). This had a Kaggle score of 3.65.

The best overall score was running a model on data that had all 8 keypoints and outputing only 8 predictions then using data from another predictions file for the remaining test cases that required more than 8 keypoints. This model had a score of 3.40 on Kaggle.

The base model plot is below:
![](https://github.com/jcweaver/blackboxes/blob/master/images/clean_wo_dups_jcw_layerplot.png)

4.2.4 **Model 3 (SP)**

This model was based upon a blog post entitled ["Achieving Top 23% in Kaggle's Facial Keypoints Detection with Keras + Tensorflow"](https://fairyonice.github.io/achieving-top-23-in-kaggles-facial-keypoints-detection-with-keras-tensorflow.html), by Shinya Yuki, which itself is an adaptation of [Daniel Nouri's approach](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) to this challenge using the now deprecated Lasagne package for CNN's. The original package was released prior to the release of the Keras library, so Yuki's version represents an update. The model was adapted to take data that had been pre-processed using our EDA and data cleaning pipeline. 

The model takes a 4D input dataset (1, 96, 96, 1) and has 3 x 2D convolution layers with 32-64-128 filters (doubles each layer). Each convolutional layer is followed by a 2x2 max-pooling layer and there are 2 fully-connected layers. The densely-connected layers have 500 units each. The model uses rectified linear unit (‘ReLU’) activation in each layer, a Nesterov-accelerated gradient descent (SGD) optimizer with a learning rate of 0.01 and a momentum parameter of 0.9. The model was trained using batches of 128 examples, and for 300 epochs. 

This model performed best using Approach 4 (see below), using a method that combines predictions of models trained using 8 keypoints of data (i.e. the variables with the most complete/least missing data) together with predictions from models trained on 30 keypoints for the samples in the testing dataset that required all 30. The top 3 positions were taken with scores generated using this methodology. 

In terms of approach, the model and parameters themselves were not changed, but I tested it using various data transformations:

Approach 1: All datasets as they were.</br>
Approach 2: Flipped images.</br>
Approach 3: Concatenated original datasets + flipped images (i.e. double size).</br>
Approach 4: Combination of 8 Keypoints and 30 Keypoints predictions from Approach 1 (see JCW and JN for more information on methodology).</br>
Approach 5: Combination of 8 Keypoints and 30 Keypoints predictions from Approach 2.</br>
Appeoach 6: Combination of 8 Keypoints and 30 Keypoints predictions from Approach 3.</br>

![](https://github.com/jcweaver/blackboxes/blob/master/images/Model%20SP/top_3_models.png)

The model plot can be seen below:

![](https://github.com/jcweaver/blackboxes/blob/master/images/Model%20SP/model_flow.png)


5. **Transformations**

To try to improve our models, we developed a few transformations. These are in the utils/transform.py file and their effects on images are displayed in the EDA/EDA_Final file.

* HorizontalFlip : This function applies a horizontal flip to the images and properly adjusts the new keypoint positions.
* Bright/Dim : This function applies brightening or dimming to images.

Here is a visual of these transformations:
![](https://github.com/jcweaver/blackboxes/blob/master/images/transformations.png)


## How to Run one of our models
To run one of the models, you'll need to take the following steps:
1. Download the data files from Kaggle and place in the data folder.
2. Run the EDA/EDA notebook file. This generates initial pickle files of the data which are faster to load.
3. Run the EDA/Data_Clean notebook file. This cleans the data and generates a variety of clean pickle files for the training data to use.
4. Choose one of the model notebook files in the models folder to run.
5. Specify where the utils directory is on your machine.
6. Specify where the clean training files are on your machine.
7. Specify where the new model files should be saved if you need to load or reference them later.
8. Run the model creation block in the model notebook file you have open.
9. Specify where predictions should be saved.
10. Run the prediction code block in the model notebook file you have open.
11. Submit any of the predictions to Kaggle: https://www.kaggle.com/c/facial-keypoints-detection/submit

## Navigating the Files in this Repository

**Outline of Repo Structure:**
```
.
├── EDA
│   ├── Augment_Missing_Data.ipynb
│   ├── Data_Clean.ipynb
│   └── EDA_Final.ipynb
├── Predictions
│   ├── JCW_Model
│   │   ├── clean_wo_dups_jcwPred.csv
│   │   ├── clean_wo_dups_jcwPred_flipped_append.csv
│   │   ├── clean_wo_dups_jcwPred_nobatch.csv
│   │   ├── clean_wo_dups_jcwPred_nobatch_nobias.csv
│   │   ├── combined_clean_all_outliers_spPred.csv
│   │   ├── combined_clean_o_dups_Lenet5Pred.csv
│   │   ├── combined_clean_o_outliers_jcwPred.csv
│   │   ├── combined_clean_w_outliers_jcwPred.csv
│   │   └── combined_clean_wo_dups_jcwPred.csv
│   ├── LeNet5
│   │   ├── clean_all_outliers_Lenet5Pred.csv
│   │   ├── clean_duplicates_Lenet5Pred.csv
│   │   ├── clean_o_dups_Lenet5Pred.csv
│   │   ├── clean_o_outliers_Lenet5Pred.csv
│   │   ├── clean_w_dups_Lenet5Pred.csv
│   │   ├── clean_w_outliers_Lenet5Pred.csv
│   │   └── clean_wo_dups_Lenet5Pred.csv
│   └── SP_Model
│       ├── concatenated
│       │   ├── clean_all_outliers_spPred.csv
│       │   ├── clean_duplicates_spPred.csv
│       │   ├── clean_o_dups_spPred.csv
│       │   ├── clean_o_outliers_spPred.csv
│       │   ├── clean_w_dups_spPred.csv
│       │   ├── clean_w_outliers_spPred.csv
│       │   └── clean_wo_dups_spPred.csv
│       ├── flipped_only
│       │   ├── clean_all_outliers_spPred.csv
│       │   ├── clean_duplicates_spPred.csv
│       │   ├── clean_o_dups_spPred.csv
│       │   ├── clean_o_outliers_spPred.csv
│       │   ├── clean_w_dups_spPred.csv
│       │   ├── clean_w_outliers_spPred.csv
│       │   └── clean_wo_dups_spPred.csv
│       └── raw_datasets
│           ├── clean_all_outliers_spPred.csv
│           ├── clean_duplicates_spPred.csv
│           ├── clean_o_dups_spPred.csv
│           ├── clean_o_outliers_spPred.csv
│           ├── clean_w_dups_spPred.csv
│           ├── clean_w_outliers_spPred.csv
│           └── clean_wo_dups_spPred.csv
├── README.md
├── cleantrain
│   ├── clean_all_outliers.p
│   ├── clean_duplicates.p
│   ├── clean_o_dups.p
│   ├── clean_o_outliers.p
│   ├── clean_w_dups.p
│   ├── clean_w_outliers.p
│   └── clean_wo_dups.p
├── data
│   ├── IdLookupTable.csv
│   ├── SampleSubmission.csv
│   ├── kaggle_files.zip
├── deliverables
│   ├── README.md
│   └── initial_modelling.pdf
├── images
│   ├── Lenet5\ Results
│   │   ├── 3rd_best.jpg
│   │   ├── Lenet_flow.png
│   │   ├── Lenet_layerplot.png
│   │   ├── all_clean_output.jpg
│   │   ├── all_outliers.jpg
│   │   ├── aug_clean_all_outliers.jpg
│   │   ├── best_score.jpg
│   │   ├── combined_wf.jpg
│   │   ├── duplicates.jpg
│   │   ├── layer_bd_score.jpg
│   │   ├── layer_hf_output.jpg
│   │   ├── layers_output.jpg
│   │   ├── layers_score.jpg
│   │   ├── o_duplicates.jpg
│   │   ├── o_outliers.jpg
│   │   ├── overall_best_score.jpg
│   │   ├── raw_model_output.jpg
│   │   ├── raw_submission.jpg
│   │   ├── top_three_results.jpg
│   │   ├── w_duplicates.jpg
│   │   ├── w_outliers.jpg
│   │   ├── w_outliers2.jpg
│   │   └── wo_duplicates.jpg
│   ├── Model\ SP
│   │   ├── model_flow.png
│   │   └── top_3_models.png
│   ├── all_clean_files.jpg
│   ├── clean_wo_dups_jcw_layerplot.png
│   └── transformations.png
├── models
│   ├── JCW_Model.ipynb
│   ├── Lenet5_Model.ipynb
│   ├── SP_model.ipynb
│   └── initial_models.ipynb
├── models_no_output
│   ├── JCW_Model.ipynb
│   ├── Lenet5_Model.ipynb
│   ├── SP_model.ipynb
│   └── initial_models.ipynb
└── utils
    ├── load_models.py
    ├── predict_models.py
    └── transform_data.py

```

Below is a list of files found in this repository along with a brief description.

|File | Description |
|:----|:------------|
|[`EDA/EDA_Final.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/EDA/EDA_Final.ipynb)|A EDA notebook that describes in detail the steps taken during the EDA phase.|
|[`EDA/Data_Clean`](https://github.com/jcweaver/blackboxes/blob/master/EDA/Data_Clean.ipynb)|A notebook to clean the data a number of different ways.|
|[`EDA/Augment_Missing_Data`](https://github.com/jcweaver/blackboxes/blob/master/EDA/Augment_Missing_Data.ipynb)|A notebook to try a linear approach to fill in missing data.|
|[`models/initial_models.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/initial_models.ipynb)| Notebook includes all the initials models that were tried with cleaned data|
|[`models/JCW_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/JCW_Model.ipynb)| Notebook with models and results built by Joanie|
|[`models/Lenet5_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/Lenet5_Model.ipynb)| Notebook with models and results built by Jackie|
|[`models/Sandip_model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models/SP_model.ipynb)| Notebook with models and results build by Sandip|
|[`models_no_output/initial_models.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/initial_models.ipynb)| Notebook includes all the initials models that were tried with cleaned data without outputs|
|[`models_no_output/JCW_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/JCW_Model.ipynb)| Notebook with models built by Joanie excluding outputs|
|[`models_no_output/Lenet5_Model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/Lenet5_Model.ipynb)| Notebook with models built by Jackie excluding outputs|
|[`models_no_output/SP_model.ipynb`](https://github.com/jcweaver/blackboxes/blob/master/models_no_output/SP_model.ipynb)| Notebook with models built by Sandip excluding outputs|
|[`utils/load_models.py`](https://github.com/jcweaver/blackboxes/blob/master/utils/load_models.py)| Custom Python Utility-tool to load, train and fit models|
|[`utils/predict_models.py`](https://github.com/jcweaver/blackboxes/blob/master/utils/predict_models.py)| Custom Python Utility-tool to run predictions with a given model and data set, and save to CSV|
|[`utils/transform_data.py`](https://github.com/jcweaver/blackboxes/blob/master/utils/transform_data.py)| Custom Python Utility-tool apply transformations on a given data set|
|[`data`](https://github.com/jcweaver/blackboxes/blob/master/data)| Directory with datasets Note: some too large to store|
|[`data/train.p`](https://github.com/jcweaver/blackboxes/blob/master/data/train.p)| Pickle-file with Cleaned Train-data|
|[`data/test.p`](https://github.com/jcweaver/blackboxes/blob/master/data/test.p)| Pickle-file with Cleaned Test-data|
|[`data/traindup.p`](https://github.com/jcweaver/blackboxes/blob/master/data/traindup.p)| Pickle-file with Cleaned train-data and duplicates removed.|
|[`data/testdup.p`](https://github.com/jcweaver/blackboxes/blob/master/data/testdup.p)| Pickle-file with Cleaned test-data and duplicates removed.|
|[`deliverables/initial_modelling.pdf`](https://github.com/jcweaver/blackboxes/blob/master/deliverables/initial_modelling.pdf)| PDF document with details of initial models|
|[`deliverables/w207 P4 initial_modelling.pptx`](https://github.com/jcweaver/blackboxes/blob/master/deliverables/w207%20P4%20initial_modelling.pptx)| Check-in Presentation slide-deck.|
|[`deliverables/w207 P4 initial_modelling.pptx`](https://github.com/jcweaver/blackboxes/blob/master/deliverables/w207%20P4%20initial_modelling.pptx)| Final Presentation slide-deck. TBD-NEED TO ADD THIS|
|[`Predictions/JCW_Model`](https://github.com/jcweaver/blackboxes/blob/master/Predictions/JCW_Model)| Prediction Submission for Models built by Joanie|
|[`Predictions/LeNet5`](https://github.com/jcweaver/blackboxes/blob/master/Predictions/LeNet5)| Prediction Submission for Models built by Jackie|
|[`Predictions/SP_Model`](https://github.com/jcweaver/blackboxes/blob/master/Predictions/SP_Model)| Prediction Submission for Models built by Sandip|
|[`images`](https://github.com/jcweaver/blackboxes/blob/master/images)| Screenshots and graph plots|
