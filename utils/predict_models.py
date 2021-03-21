#######################################
# Load Imports
#######################################

import pandas as pd
import numpy as np
import transform_data
import argparse
import pickle
import os
from keras.models import Model, model_from_json
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


class PredictModels(object):

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


    #######################################
    # Load Models
    #######################################
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

    def __generate_prediction(self, model_name, Y, test, columns="Full", verbose = True):

        id_lookup = self.__ids

        if columns == "Full":
            train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
            'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
            'right_eye_inner_corner_y', 'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
        else:  
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


    def print_paths(self):
        print("Model dir:", self.__model_dir)
        print("Pickle dir:", self.__pred_dir)

    #######################################
    # Predict Models
    #######################################
    def predict_lenet5(self, model_name,model_file, model_json, test, scale = True, X=None, verbose = False):
        
        model_file_name = "".join([self.__model_dir, model_file])
        model_json_file = "".join([self.__model_dir, model_json])

        clean_test = test.drop(['index', 'check_sum'], axis=1, errors='ignore')
        data_transform = transform_data.TransformData(verbose=True)
        if scale:
            test_scaled = data_transform.ScaleImages(test, True)
            X, test_subset = data_transform.SplitTest(test_scaled,self.__ids, verbose = True)
        
        # Create or load the model
        if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
            if verbose: 
                print("LeNet5 model file not found. ")
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" %(model_file_name, model_json_file))

        if verbose: 
            print("Loading model:", model_file_name)
        
        # predict
        model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = False)
        if verbose: 
            print("Predicting %d (x,y) coordinates" % (len(X)))
        
        if verbose: 
            print("Predicting model:", model_file_name)
        Y = model.predict(X, verbose = verbose)

        if verbose: 
            print("Predictions complete!")
        
        #Generate Predictions
        self.__generate_prediction(model_name, Y, test, columns="Full")
        return Y