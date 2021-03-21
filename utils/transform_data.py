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

class TransformData(object):

    #Let's make this easy for everyone to use since we all have different paths for our files
    #pickles_path - file where all the yummy pickle files are
    def __init__(self, scale = 255.0, reshape = 96, verbose = False):
        
        # validate that the constructor parameters were provided by caller
        #I don't think we need the pickle path in here...but keep this code because 
        #I'll want to reuse this elsewhere. 
        #if (not pickle_file_path):
        #    raise RuntimeError('Please provide a path to the pickle files.')
        
        #Get a clean path...didn't work for me on first try because of spaces. 
        #pickle_file_path = str(pickle_file_path).replace('\\', '/').strip()
        #if (not pickle_file_path.endswith('/')): pickle_file_path = ''.join((pickle_file_path, '/'))

        #Let's make sure all is good in the world and we can find the path
        #if (not os.path.isdir(pickle_file_path)):
        #    raise RuntimeError("Path of pickle files '%s' is invalid. Please resolve and try again." % pickle_file_path)

        #self.__pickle_file_path = pickle_file_path #Do we need this? not sure but for now let's keep it. 
        self.__scale_image_by = scale
        self.__reshape_image = reshape 


    #Rakesh to add rotate, etc. 

    ##Private
    def __get_coordinate_columns(df, x = True, y = True):

        if x & y:
            coordinates = [c for c in df.columns if c.endswith('_x') | c.endswith('_y')]
        elif x:
            coordinates = [c for c in df.columns if c.endswith('_x')]
        else:
            coordinates = [c for c in df.columns if c.endswith('_y')]
        return coordinates

    #it might be easier to work wtih a dictionary as these are pairs??
    def __get_coordinate_dict():
        coord = {
                'left_eye_center_x': 'left_eye_center_y',  
                'right_eye_center_x': 'right_eye_center_y', 
                'left_eye_inner_corner_x': 'left_eye_inner_corner_y', 
                'left_eye_outer_corner_x': 'left_eye_outer_corner_y', 
                'right_eye_inner_corner_x': 'right_eye_inner_corner_y', 
                'right_eye_outer_corner_x': 'right_eye_outer_corner_y', 
                'left_eyebrow_inner_end_x': 'left_eyebrow_inner_end_y', 
                'left_eyebrow_outer_end_x': 'left_eyebrow_outer_end_y', 
                'right_eyebrow_inner_end_x': 'right_eyebrow_inner_end_y', 
                'right_eyebrow_outer_end_x': 'right_eyebrow_outer_end_y', 
                'nose_tip_x': 'nose_tip_y', 
                'mouth_left_corner_x': 'mouth_left_corner_y', 
                'mouth_right_corner_x': 'mouth_right_corner_y', 
                'mouth_center_top_lip_x': 'mouth_center_top_lip_y', 
                'mouth_center_bottom_lip_x': 'mouth_center_bottom_lip_y'}

        return coord

    #it might be easier to work wtih a dictionary as these are pairs??
    def __get_coordinate_dict_flipped():
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

    ##Public

    #NOT TESTED
    def FlipHorizontal(self, train, verbose = False):
        #Flip the iages horizontaly and adjust the labels. 
        
        adj_train = train.copy()
        
        
        #this will only do all the keypoints. Not sure about this..
        adj_train = adj_train[(adj_train.isnull().sum(axis = 1) == 0)]

        # horizontally flip the images
        adj_train.image = adj_train.image.map(lambda x: np.flip(x.reshape(self.__scale_image_by,self.__scale_image_by), axis=1).ravel())

        cols = __get_coordinate_columns(adj_train, True, False)
        
        # shift all 'x' values by linear mirroring
        for c in cols:
            mod = adj_train[c].values
            mod = np.clip(np.float(0 - 95) - mod, np.float(0), np.float(95))
            adj_train[c] = mod

        
        cols = adj_train.columns
        if verbose:
            print(cols)

        adj_train.rename(columns=__get_coordinate_dict_flipped(), inplace=True)
        
        # change the column order back to original
        adj_train = adj_train[cols]

        if verbose: 
            print(f"Horizontal df with size of {adj_train.shape}")
        return adj_train


    def ScaleImages(self, df, verbose = False):
        #For most CNN we will need to scale the images by 255. 

        if verbose: 
            print("Scaling %d images..." % df.shape[0])
        if 'image' in df.columns:
            
            image_scaled = df.image.values
            #print("Before:",image_scaled)
            image_scaled = image_scaled / self.__scale_image_by
            df.image = image_scaled
            #print("after:",image_scaled)

            if verbose: 
                print("Scaling of %d observations complete." % df.shape[0])
        else:
            print("Scaling not complete. Column 'image' missing.")
        
        #Return the df. Note: user should pickle after this. 
        return df



    def SplitTrain(self, train, normalize = True, verbose = True):
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
        #Y = temp_df[temp_df.columns[1:-1]].values

        #This has already been normalized but this will give us -1,1? 
        if normalize:
            #Since the data has already been normalized not sure we need to do this
            #this is to test it out. Yes we do need to do this!!
            for i in range(Y.shape[1]):
                new_arr = []
                current_arr = Y[:,i]
                for j in range(len(current_arr)):
                    new_arr.append((current_arr[j] - 48.)/48.)    
                Y[:,i] = new_arr
        return X, Y
        

    def SplitTest(self, test, ids, verbose = False):
        
        if verbose:
            print("Begining the split of Test")

        
        id_lookup = ids
    

        #Get the unique image id's. 
        unique_ids = id_lookup.image_id.unique()
        print("got unique ids")

        #Get the subset of images in test
        test_subset = test[(test.image_id.isin(unique_ids))]
        print("test subset shape:", test_subset.shape)

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




