#######################################
# Load Imports
#######################################

import pandas as pd
import numpy as np
from utils import transform_data
import argparse
import pickle
import os
from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, Add, ZeroPadding2D, GlobalMaxPooling2D, DepthwiseConv2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')

for pd_dev in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)


def get_GPU_count():
    GPU_count = len(tf.config.list_physical_devices('GPU'))
    return GPU_count

def plot_history(hist):
    #Return value hist from the model fit can be used to plot
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title("Loss vs epoch number")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.ylim(1e-3, 1e-2)
    #plt.yscale('log')
    plt.show()

class LoadTrainModels(object):
    ##### PRIVATE
    def __init__(self, model_dir, pickle_path, verbose = False):
        
        # validate that the constructor parameters were provided by caller
        #I don't think we need the pickle path in here...but keep this code because 
        #I'll want to reuse this elsewhere. 
        if (not pickle_path):
            raise RuntimeError('Please provide a path to the pickle files.')
        
        #Get a clean path...didn't work for me on first try because of spaces. 
        pickle_path = str(pickle_path).replace('\\', '/').strip()
        if (not pickle_path.endswith('/')): 
            pickle_path = ''.join((pickle_path, '/'))

        #Let's make sure all is good in the world and we can find the path
        if (not os.path.isdir(pickle_path)):
            raise RuntimeError("Path of pickle files '%s' is invalid. Please resolve and try again." % pickle_path)

        if (not model_dir):
            raise RuntimeError('Please provide a path to the model files.')
        
        #Get a clean path...didn't work for me on first try because of spaces. 
        model_dir = str(model_dir).replace('\\', '/').strip()
        if (not model_dir.endswith('/')): 
            model_dir = ''.join((model_dir, '/'))

        #Let's make sure all is good in the world and we can find the path
        if (not os.path.isdir(model_dir)):
            os.makedirs(model_dir)
        
        if (not os.path.isdir(model_dir)):
            raise RuntimeError("Path of model files '%s' is invalid. Please resolve and try again." % model_dir)

        self.__pickle_file_path = pickle_path #Do we need this? not sure but for now let's keep it. 
        self.__model_dir = model_dir

    
    #######################################
    # Load Models
    #######################################
    def __load_model_from_file(self,model_name, model_file, model_json, verbose = False):

        #model_filename = "".join((self.__model_dir, model_file))
        #model_json = "".join((self.__model_dir, model_json))

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
    def __get_model_lenet5(self,model_name, X, Y, l_batch_size, l_epochs, l_validation_split = .01, x_val = None, y_val = None, l_shuffle = True, verbose = True):
        
        model_file_name = "".join([self.__model_dir, model_name,".h5"])
        model_json_file = "".join([self.__model_dir, model_name,".json"])
        
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

            stop_at = np.max([int(0.1 * l_epochs), 10])
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = model_file_name, verbose = verbose, save_best_only = True, 
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
          
            compliled_model = model

            #play with these numbers.....
            #https://keras.io/api/models/model_training_apis/
            compliled_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if verbose: print("Done compiling")
            
            #https://keras.io/api/models/model_training_apis/
            #Validation data will override validation split so okay to include both
            #This return a history object:
            # A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
            # Might be interesting to plot this....TBD
            history = compliled_model.fit(X, Y, validation_split = l_validation_split, batch_size = l_batch_size * GPU_count, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)
            if verbose: print("Done fitting")
            #This one does not like the none values for x_val and y_val...
            #history = compliled_model.fit(X, Y, validation_split = l_validation_split, validation_data = (x_val, y_val), batch_size = l_batch_size * GPU_count, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)

            #Save the model and we can version these ... might want to make it so I can modify the names for different versions and configs??
            model_json = compliled_model.to_json()
            with open(model_json_file, "w") as json_file:
                json_file.write(model_json)
                       
            #history_param_file = "".join([full_path, model_name,"_hparam.csv"])
            #history_params = pd.DataFrame(history.params)
            #history_params.to_csv(history_param_file)

            #history_file = "".join([full_path, model_name,"_hist.csv"])
            #hist = pd.DataFrame(history.history)
            #hist.to_csv(history_file)

            if verbose:
                print(f"{model_name} model created and file saved for future use.")
        else:
            #We already have a model file, so retrieve and return it. 
            model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)
            #TODO need to add history file here. 
        return model, history
    
    
    
    def __get_model_jcw(self,model_name, X, Y, l_batch_size, l_epochs, l_validation_split = .2, x_val = None, y_val = None, l_shuffle = True, verbose = True):
        
        model_file_name = "".join([self.__model_dir, model_name,".h5"])
        model_json_file = "".join([self.__model_dir, model_name,".json"])
        
        if verbose: 
            print("Looking for model JW")

        # Create or load the model
        if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
            if verbose: 
                print("JW model file not found. Model creation beginning")

                GPU_count = len(tf.config.list_physical_devices('GPU'))

            #create a model and return it? or save it? 
            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            sgd = SGD(lr=0.005, momentum=0.9, nesterov=True) 
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']
            
            #Used for the fitting step
            stop_at = np.max([int(0.1 * l_epochs), 10])
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if GPU_count > 1: 
                dev = "/cpu:0"
            else: 
                dev = "/gpu:0"
            with tf.device(dev):

                model = Sequential()

                #Add layers
                model.add(Convolution2D(32, 3, 3, input_shape=(96, 96,1), data_format='channels_last', use_bias=False))
                model.add(Activation('relu'))
                # we apply batch normalization, which applies a transformation that maintains
                # the mean output close to 0 and the output standard deviation close to 1
                model.add(BatchNormalization()) 
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Convolution2D(64, 2, 2, use_bias=False))
                model.add(Activation('relu'))
                model.add(BatchNormalization()) 
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Convolution2D(128, 2, 2, use_bias=False))
                model.add(Activation('relu'))
                model.add(BatchNormalization()) 
                model.add(MaxPooling2D(pool_size=(2, 2)))

                #Flatten transforms the fully connected layer so it can be read
                model.add(Flatten())
                model.add(Dense(512))
                model.add(Activation('relu'))
                model.add(Dense(512))
                model.add(Activation('relu'))
                model.add(Dense(30))
                

            if verbose: 
                print(model.summary())
          
            compiled_model = model

            #play with these numbers.....
            #https://keras.io/api/models/model_training_apis/
            compiled_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if verbose: print("Done compiling")
            
            #https://keras.io/api/models/model_training_apis/
            #Validation data will override validation split so okay to include both
            #This return a history object:

            history = compiled_model.fit(X, Y, validation_split = l_validation_split, batch_size = l_batch_size * GPU_count, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)
            if verbose: print("Done fitting")
            

            #Save the model and we can version these ... might want to make it so I can modify the names for different versions and configs??
            model_json = compiled_model.to_json()
            with open(model_json_file, "w") as json_file:
                json_file.write(model_json)
        
            #Plotting history
            plot_history(history)

            if verbose:
                print(f"{model_name} model created and file saved for future use.")
        else:
            #We already have a model file, so retrieve and return it. 
            model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)
            #TODO need to add history file here. 
        return model, history

    ###### PUBLIC
    
    #######################################
    # PRINT PATHS
    #######################################
    def print_paths(self):
        print("Model dir:", self.__model_dir)
        print("Pickle dir:", self.__pickle_file_path)

    

    #######################################
    # Train Models
    #######################################
    def train_lenet5(self, model_name, train, split=True, X=None, Y=None, verbose = True):

        data_transform = transform_data.TransformData(verbose=True)
        #Scale train
        train_scaled = data_transform.ScaleImages(train, verbose = True)
        
        #Split train and scale accordingly
        # #do the split here and pass in parameters
        if(split):
            X, Y = data_transform.SplitTrain(train_scaled)
        elif X is None | Y is None:
            raise RuntimeError(f"When Split is set to False, X and Y must be supplied." )
        
        #Get and compile the model. 
        model, history = self.__get_model_lenet5(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True)
        
        return model, history
    
    def train_jcw(self, model_name, train, split=True, X=None, Y=None, verbose = True):

        data_transform = transform_data.TransformData(verbose=True)
        #Scale train
        train_scaled = data_transform.ScaleImages(train, verbose = True)
        
        #Split train and scale accordingly
        # #do the split here and pass in parameters
        if(split):
            X, Y = data_transform.SplitTrain(train_scaled)
        elif X is None | Y is None:
            raise RuntimeError(f"When Split is set to False, X and Y must be supplied." )
        
        #Get and compile the model. 
        model, history = self.__get_model_jcw(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True)
        
        return model, history

