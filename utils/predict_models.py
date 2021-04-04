#######################################
# Load Imports
#######################################

import pandas as pd
import numpy as np
#from utils import transform_data
import transform_data
import argparse
import pickle
import os
from keras.models import Model, model_from_json
from keras import backend as K
import tensorflow as tf

#######################################################################################
# PredictModels class
# 
# This class takes a model and test set and makes predictions
# It contains the following:
#
# Private functions:
# __init__ - initialize the class
# __load_model_from_file - loads a model from json file
# __generate_prediction - creates a csv file of predictions
#
# Public functions:
# print_paths - print the paths set for data files
# predict_standard - generic function that prepares and generates predictions 
#
######################################################################################
class PredictModels(object):
    ############################# PRIVATE FUNCTIONS #################################################
    def __init__(self, model_dir,pred_dir, ids, verbose = False):

        # validate that the constructor parameters were provided by caller
        #I don't think we need the pickle path in here...but keep this code because

        if (not model_dir):
            raise RuntimeError('Please provide a path to the model files.')

        #Get a clean path...didn't work for me on first try because of spaces.
        model_dir = str(model_dir).replace('\\', '/').strip()
        if (not model_dir.endswith('/')):
            model_dir = ''.join((model_dir, '/'))

        #Let's make sure all is good in the world and we can find the path
        if (not os.path.isdir(model_dir)):
            raise RuntimeError("Path of model files '%s' is invalid. Please resolve and try again." % model_dir)

        self.__model_dir = model_dir
        self.__pred_dir = pred_dir
        self.__ids = ids


    ##################################################################################
    # __load_model_from_file
    # Load the model from a specfied json file
    #
    ##################################################################################
    def __load_model_from_file(self,model_name, model_file, model_json, verbose = False):

        if not os.path.isfile(model_file):
            raise RuntimeError(f"Model file {model_file} does not exist.")
        if not os.path.isfile(model_json):
            raise RuntimeError(f"Model file {model_json} does not exist." )

        #Load
        if verbose:
            print(f"Loading model: {model_name}")
        json_file = open(model_json, "r")
        model_json_data = json_file.read()
        json_file.close()
        model = model_from_json(model_json_data)
        model.load_weights(model_file)

        return model

    ##################################################################################
    # __generate_prediction
    # Genereate some prediction files that we can submit to Kaggle
    #
    ##################################################################################
    def __generate_prediction(self, model_name, Y, test, columns="Full", verbose = True):

        id_lookup = self.__ids

        
        if columns == "Full":
            #All 30 points
            train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
            'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
            'right_eye_inner_corner_y', 'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
        else:
            #8 points only
            train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']

        print(id_lookup.shape)
        print("before melt:", Y.shape)
        Y = pd.DataFrame(Y, columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
        Y = pd.melt(Y.reset_index(), id_vars=['index'])
        print("after melt:",Y.shape)
        Y.columns = ['image_id', 'feature_name','location']
        Y = id_lookup.drop(columns=['location']).merge(Y, on=['image_id','feature_name'], how = 'inner').drop(columns=['image_id','feature_name'])
        Y.columns = ['RowId','Location']

        print("after merge:",Y.shape)

        new_arr = []
        for i in range(len(Y.Location.values)):
            new_arr.append( ((Y.Location.values[i] * 48.) + 48.))

        Y.Location = new_arr

        Y.Location = np.clip(Y.Location, 0.0, 96.0)

        # write the predictions file
        #"../Data/Jackie_Lenet5Pred2.csv"
        new_file = ''.join((self.__pred_dir,model_name,"Pred.csv"))
        print(new_file)
        Y.to_csv(new_file, index = False)
        print("Predictions written ")

    ############################# PUBLIC FUNCTIONS #################################################
    
    ##################################################################################
    # print_paths
    # Print the paths set for data files
    #
    ##################################################################################
    def print_paths(self):
        print("Model dir:", self.__model_dir)
        print("Pickle dir:", self.__pred_dir)

    ##################################################################################
    # predict_standard
    # Generic function that prepares and generates predictions
    # Inspired by https://www.kaggle.com/balraj98
    ##################################################################################
    def predict_standard(self, model_name,model_file, model_json, test, scale = True, X=None, verbose = False, columns = "Full"):

        model_file_name = "".join([self.__model_dir, model_file])
        model_json_file = "".join([self.__model_dir, model_json])

        clean_test = test.drop(['index', 'check_sum'], axis=1, errors='ignore')
        data_transform = transform_data.TransformData(verbose=True)

        #We scaled train we need to scale test
        if scale:
            test_scaled = data_transform.ScaleImages(test, True)
            X, test_subset = data_transform.SplitTest(test_scaled,self.__ids, verbose = True)

        # Create or load the model
        if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
            if verbose:
                print("Model file not found:", model_name)
            raise RuntimeError("Prediction cancelled. Files not found")

        if verbose:
            print("Loading model:", model_file_name)

        
        model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = False)
        if verbose:
            print("Predicting %d (x,y) coordinates" % (len(X)))

        if verbose:
            print("Predicting model:", model_file_name)
        # predict
        Y = model.predict(X, verbose = verbose)

        if verbose:
            print("Predictions complete!")

        #Generate Predictions
        self.__generate_prediction(model_name, Y, test, columns=columns)
        return Y


    