#%%
#######################################
# Load Imports
#######################################

import pandas as pd
import numpy as np
import transform_data

import argparse
import pickle
import os
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, Add, ZeroPadding2D, GlobalMaxPooling2D, DepthwiseConv2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.initializers import glorot_uniform, Constant, lecun_uniform
from keras import backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')

for pd_dev in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)

#%%
def get_GPU_count():
    GPU_count = len(tf.config.list_physical_devices('GPU'))
    return GPU_count

#######################################
# Load Models
#######################################
def load_model_from_file(model_name, model_file, model_json, verbose = False):
        
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

#######################################
# Get Models
#######################################
def get_model_lenet5(model_path, X, Y, l_batch_size, l_epochs, l_validation_split = .01, x_val = None, y_val = None, l_shuffle = True, verbose = True):

    model_name= "LeNet5"
    full_path = "".join([model_path,"Jackie_Lenet5/"])

    #Check the path, if it doesn't exist create it so we can save the model there later.
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    model_file_name = "".join([full_path, "Lenet5.h5"])
    model_json_file = "".join([full_path, "Lenet5.json"])
    
    if verbose: 
        print("Looking for model LeNet5")

    # Create or load the model
    if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
        if verbose: 
            print("LeNet5 model file not found. Model creation begnining")

        GPU_count = len(tf.config.list_physical_devices('GPU'))

        #create a model and return it? or save it? 
        #act = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.1, epsilon = 1e-8)
        act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        lss = 'mean_squared_error'
        mtrc = ['mae','mse']

        stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
        es = EarlyStopping(patience = stop_at, verbose = verbose)
        cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
            mode = 'min', monitor = 'val_mae')

        if GPU_count > 1: 
            dev = "/cpu:0"
        else: 
            dev = "/gpu:0"
        with tf.device(dev):

            model = Sequential()

            #Add layers
            model.add(Convolution2D(filters = 6, kernel_size = (3, 3), input_shape = (96, 96, 1)))
            model.add(ReLU())
            model.add(AveragePooling2D())
            #model.add(Dropout(0.2))

            model.add(Convolution2D(filters = 16, kernel_size = (3, 3)))
            model.add(ReLU())
            model.add(AveragePooling2D())
            #model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(ReLU())
            
            model.add(Dense(256))
            model.add(ReLU())
            
            #30 features or #8 features. not sure which works better yet
            model.add(Dense(30))
            #model.add(Dense(8))
            

        if verbose: 
            print(model.summary())

        #Check the GPU situation since the team is using different systems
        if GPU_count > 1:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                #keras multi_gpu_model let's use it if we have it
                #https://rdrr.io/cran/keras/man/multi_gpu_model.html
                compliled_model = multi_gpu_model(model, gpus = GPU_count)
        else:
            #ug we don't have any GPUs..that's okay. 
            compliled_model = model

        #play with these numbers.....
        #https://keras.io/api/models/model_training_apis/
        compliled_model.compile(optimizer = act, loss = lss, metrics = mtrc)


        
        #https://keras.io/api/models/model_training_apis/
        #Validation data will override validation split so okay to include both
        #This return a history object:
        # A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        # Might be interesting to plot this....TBD
        history = compliled_model.fit(X, Y, validation_split = l_validation_split, validation_data = (x_val, y_val), batch_size = l_batch_size * GPU_count, 
                epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)

        #Save the model and we can version these ... might want to make it so I can modify the names for different versions and configs??
        model_json = compliled_model.to_json()
        with open(model_json_file, "w") as json_file:
            json_file.write(model_json)
    
        if verbose:
            print(f"{model_name} model created and file saved for future use.")
    else:
        #We already have a model file, so retrieve and return it. 
        model = load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)


    return model


# %%
#######################################
# Train Models
#######################################
def train_lenet5(model_path):

    #do the split here and pass in parameters

    model = get_model_lenet5(model_path, X, Y)



# %%
#######################################
# Main - if needed
#######################################
if __name__ == "__main__":
        
    
    print("hello")
    print(get_GPU_count())
# %%

