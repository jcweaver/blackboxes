import pandas as pd
import numpy as np
import zlib
from math import sin, cos, pi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import time
from scipy import ndimage
import pickle
import os.path

#######################################################################################
# TransformData class
# 
# This class is used to transform data in a number of ways
# It contains the following:
#
# Private functions:
# __init__ - initialize the class
# __get_coordinate_columns - returns a list of (x,y) coordinate columns
# __get_coordinate_dict_flipped - return the list of (x,y) coordinate columns flipped
# horizontally 
#
# Public functions:
# FlipHorizontal - flips the images horizontally 
# Bright_Dim - brighten and/or dim an image
# ScaleImages - Scale the Images by 255
# Split Train - Split the train data into X and Y and in the proper shape
# Split Test - Split the train data into X and Y and in the proper shape
######################################################################################
class TransformData(object):
    ############################# PRIVATE FUNCTIONS #################################################
    #Let's make this easy for everyone to use since we all have different paths for our files
    #pickles_path - file where all the yummy pickle files are
    def __init__(self, scale = 255.0, reshape = 96, verbose = False, prop = 0.1):
        
        # validate that the constructor parameters were provided by caller
        
        self.__scale_image_by = scale
        self.__reshape_image = reshape
        
        #Prop is a proportion that can be used for shifting images
        self.prop = prop


    #Rakesh to add rotate, etc. 
    ##################################################################################
    #  __get_coordinate_columns
    #   
    # Return the list of (x,y) coordinate columns 
    #   
    ##################################################################################  
    def __get_coordinate_columns(self, df, x = True, y = True):

        if x & y:
            coordinates = [c for c in df.columns if c.endswith('_x') | c.endswith('_y')]
        elif x:
            coordinates = [c for c in df.columns if c.endswith('_x')]
        else:
            coordinates = [c for c in df.columns if c.endswith('_y')]
        return coordinates

    ##################################################################################
    #  __get_coordinate_dict_flipped
    #   
    # Return the list of (x,y) coordinate columns flipped horizontally
    #   
    ##################################################################################  
    #it might be easier to work wtih a dictionary as these are pairs??
    def __get_coordinate_dict_flipped(self):
        coord = {
            'left_eye_inner_corner_x':'right_eye_inner_corner_x',
            'left_eye_center_x':'right_eye_center_x',
            'left_eye_outer_corner_x':'right_eye_outer_corner_x',
            'left_eyebrow_inner_end_x':'right_eyebrow_inner_end_x',
            'left_eyebrow_outer_end_x':'right_eyebrow_outer_end_x',
            'mouth_left_corner_x':'mouth_right_corner_x',
            'right_eye_inner_corner_x':'left_eye_inner_corner_x',
            'right_eye_center_x':'left_eye_center_x',
            'right_eye_outer_corner_x':'left_eye_outer_corner_x',
            'right_eyebrow_inner_end_x':'left_eyebrow_inner_end_x',
            'right_eyebrow_outer_end_x':'left_eyebrow_outer_end_x',
            'mouth_right_corner_x':'mouth_left_corner_x',
            'left_eye_inner_corner_y':'right_eye_inner_corner_y',
            'left_eye_center_y':'right_eye_center_y',
            'left_eye_outer_corner_y':'right_eye_outer_corner_y',
            'left_eyebrow_inner_end_y':'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_y':'right_eyebrow_outer_end_y',
            'mouth_left_corner_y':'mouth_right_corner_y',
            'right_eye_inner_corner_y':'left_eye_inner_corner_y',
            'right_eye_center_y':'left_eye_center_y',
            'right_eye_outer_corner_y':'left_eye_outer_corner_y',
            'right_eyebrow_inner_end_y':'left_eyebrow_inner_end_y',
            'right_eyebrow_outer_end_y':'left_eyebrow_outer_end_y',
            'mouth_right_corner_y':'mouth_left_corner_y'}

        return coord

    
    ############################# PUBLIC FUNCTIONS #################################################


    ##################################################################################
    #  FlipHorizonal
    #  Used to flip images horizontaly 
    #  Inspired from
    #  https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection
    #
    ##################################################################################  
    def FlipHorizontal(self, train, verbose = False):
        #Flip the images horizontaly and adjust the labels. 
        
        adj_train = train.copy()
        
        #this will only do all the keypoints. Not sure about this..
        #removes null rows
        adj_train = adj_train[(adj_train.isnull().sum(axis = 1) == 0)]

        #https://www.techbeamers.com/python-map-function/ with lambda 
        # horizontally flip the images - use the map function
        adj_train.image = adj_train.image.map(lambda x: np.flip(x.reshape(self.__reshape_image,self.__reshape_image), axis=1).ravel() )

        cols = self.__get_coordinate_columns(adj_train, True, False)
        
        # shift all 'x' values by linear mirroring
        for c in cols:
            mod = adj_train[c].values
            mod = np.clip(np.float(0 - 95) - mod, np.float(0), np.float(95))
            adj_train[c] = mod

        #Get the columns so we can reorder later
        cols = adj_train.columns
        if verbose:
            print(cols)

        #ug this only works if we also rename the columns...
        adj_train.rename(columns=self.__get_coordinate_dict_flipped(), inplace=True)
        #need to verify this...
        # change the column order back to original
        adj_train = adj_train[cols]

        if verbose: 
            print(f"New Horizontal shape: {adj_train.shape}")
        return adj_train


    ##################################################################################
    #  Bright_Dim
    #  Used to brighten or dim an image
    #  Inspired from
    #  https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection
    #
    #  https://scipy-lectures.org/advanced/image_processing/
    #  https://www.tutorialspoint.com/scipy/scipy_ndimage.htm
    ##################################################################################
    def Bright_Dim(self, train, level_of_brightness = 1, level_to_dim = 1, verbose = False):
        
        #Used to brighten
        bright_train = train.copy()

        # Used for dimming
        dim_train = bright_train.copy()

        if level_of_brightness == 1:
            print("Skipping brightness")
        else:
            if verbose: 
                print(f"Number of images to be brightened: {bright_train.shape[0]}")
            #https://www.techbeamers.com/python-map-function/ with lambda 
            #Apply a level of brightness with min =0 and max = 1 for every image in bright_train
            bright_train.image = bright_train.image.map(lambda x: np.clip(x * level_of_brightness, 0.0, 1.0))
        
        
        if level_to_dim == 1:
            print("Skipping dim")
        else:
            if verbose: 
                print(f"Number of images to be dimmed: {dim_train.shape[0]}")
            #https://www.techbeamers.com/python-map-function/ with lambda 
            #Apply a level of dim with min =0 and max = 1 for every image in dim_train    
            dim_train.image = dim_train.image.map(lambda x: np.clip(x * level_to_dim, 0.0, 1.0))
            #Append dimmed images to brightened images. 
            bright_train = bright_train.append(dim_train, ignore_index = True).reset_index().drop(columns=['index'])

        if verbose: 
            print(f"Completed brighten and dim. Number of observations added to train: {bright_train.shape[0]}")
        return bright_train   


    ##################################################################################
    #  ScaleImages
    #  Scale images by 255
    #  Inspired by
    #  https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection#Exploring-Data
    #
    ##################################################################################
    def ScaleImages(self, df, verbose = False):
        #For most CNN we will need to scale the images by 255. 
        if verbose: 
            print("Scaling images")
        if 'image' in df.columns:
            image_scaled = df.image.values
            image_scaled = image_scaled / self.__scale_image_by
            df.image = image_scaled
            if verbose: 
                print("Scaling complete.")
        else:
            print("Scaling not complete. Column 'image' missing.")
        
        #Return the df. Note: user may want to pickle this. 
        return df


    ##################################################################################
    # SplitTrain
    # Split the train data into X and Y and in the proper shape
    # 
    ##################################################################################
    def SplitTrain(self, train, verbose = True):
        #Split the train data into X,Y and in the proper shape
        #Everyone should be able to use this

        if verbose:
            print("Begining the split of Train with all features")
        
        coord_cols = train.columns

        ## need to figure out how to split these...
        ## we could use Keras but not everyone will be using that...
        temp_df = train[coord_cols].copy().dropna(axis = 'index', how = 'any')
        temp_df.image = temp_df.image.map(lambda x: np.array(x).reshape(self.__reshape_image,self.__reshape_image,1))
        
        #Similar to what Joanie did
        X = []
        for i, r in temp_df.iterrows():
            X.append(r.image)
        X = np.array(X)

        Y = temp_df.drop(columns=['image']).values
        
        for i in range(Y.shape[1]):
            new_arr = []
            current_arr = Y[:,i]
            for j in range(len(current_arr)):
                new_arr.append((current_arr[j] - 48.)/48.)    
            Y[:,i] = new_arr

        return X, Y
        
    ##################################################################################
    # SplitTest
    # Split the Test data into X and subset and in the proper shape
    # 
    ##################################################################################
    def SplitTest(self, test, ids, verbose = False):
        
        if verbose:
            print("Begining the split of Test")

        id_lookup = ids
    
        #Get the unique image id's. 
        unique_ids = id_lookup.image_id.unique()
        
        #Get the subset of images in test
        test_subset = test[(test.image_id.isin(unique_ids))]

        #reshape the images
        test_subset.image = test_subset.image.map(lambda x: np.array(x).reshape(96,96,1))     
        
        X = []
        for i, r in test_subset.iterrows():
            X.append(r.image)
        X = np.array(X)
        
        test_subset.drop(columns=['image'], inplace = True)

        if verbose:
            print("End with the split of Test")
        return X, test_subset




