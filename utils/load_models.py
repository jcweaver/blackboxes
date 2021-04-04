#######################################
# Load Imports
#######################################

import pandas as pd
import numpy as np
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
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import transform_data


tf.random.set_seed(42)

tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')

for pd_dev in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)


def get_GPU_count():
    GPU_count = len(tf.config.list_physical_devices('GPU'))
    return GPU_count

#TODO remove this as it was added to the class
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
        if (not model_dir):
            raise RuntimeError('Please provide a path to the model files.')

        #Get a clean path...didn't work for me on first try because of spaces.
        model_dir = str(model_dir).replace('\\', '/').strip()
        if (not model_dir.endswith('/')):
            model_dir = ''.join((model_dir, '/'))

        #Let's make sure all is good in the world and we can find the path
        #If not let's make the path
        if (not os.path.isdir(model_dir)):
            os.makedirs(model_dir)

        if (not os.path.isdir(model_dir)):
            raise RuntimeError(f"Model file {model_dir} is invalid. Please try again.")

        self.__model_dir = model_dir


    #######################################
    # Load Models
    # Load a model from a specified json file
    #
    #
    #######################################
    def __load_model_from_file(self,model_name, model_file, model_json, verbose = False):
        #Check to see if we have the files
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
        #Load from the json file and return.
        model = model_from_json(model_json_data)
        model.load_weights(model_file)

        return model

    #######################################
    # Save Model History Information
    #
    # Save a models history parameters, history and plot.
    #
    #######################################
    def __save_history_info(self, history, model_name, plot_name, metric = "mse", verbose = False):
            #Inspired by conversation with Cris B.
        if verbose:
            print("Saving the history paramters file")
        history_param_file = "".join([self.__model_dir, model_name,"_param.csv"])
        dct = {k:[v] for k,v in history.params.items()}
        history_params = pd.DataFrame(dct)
        history_params.to_csv(history_param_file)

        if verbose:
            print("Saving the history paramters file")
        history_file = "".join([self.__model_dir, model_name,"_hist.csv"])
        hist = pd.DataFrame(history.history)
        hist.to_csv(history_file)

        if verbose:
            print("Creating plots")

        fig = plt.figure(figsize=(15,8),dpi=100)
        fig.suptitle(model_name)
        ax = fig.add_subplot(1,2,1)

        #https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        #https://matplotlib.org/stable/tutorials/intermediate/constrainedlayout_guide.html#sphx-glr-tutorials-intermediate-constrainedlayout-guide-py
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html

        #Plot Loss vs Epoch
        ax.plot(history.history['loss'][1:], label = 'Train',marker = 'o', markersize = 3, alpha = 0.9)
        ax.plot(history.history["".join(["val_loss"])][1:], label = 'Validation',marker = 'o', markersize = 3, alpha = 0.9)
        ax.set_title("Loss vs. Epoch", fontsize = 15, fontweight = 'bold')
        ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0)
        ax.set_ylabel("Loss", fontsize = 12, horizontalalignment='right', y = 1.0)
        plt.legend(loc = 'upper right')

        #Add subplot for next plot
        ax = fig.add_subplot(1,2,2)

        #Plot MSE vs Epoch
        ax.plot(history.history[metric][1:], label = 'Train',
            marker = 'o', markersize = 4, alpha = 0.9)
        ax.plot(history.history["".join(["val_",metric])][1:],  label = 'Validation',
            marker = 'o', markersize = 4, alpha = 0.9)
        ax.set_title("MSE vs Epoch", fontsize = 15, fontweight = 'bold')
        ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0)
        ax.set_ylabel(metric, fontsize = 12, horizontalalignment='right', y = 1.0)
        plt.legend(loc = 'upper left')

        plt.tight_layout()
        plt.savefig(plot_name, dpi=300)
        if verbose:
            print("Plot saved")
        plt.close()

    #######################################
    # Get Models
    #
    # get_model_jn - Model written by Jackie based on Lenet5
    #
    #######################################
    def __get_model_jn(self,model_name, X, Y, l_batch_size, l_epochs, l_validation_split = .01, l_shuffle = True, layers = 7, verbose = True):
        #Inspired by https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086

        model_file_name = "".join([self.__model_dir, model_name,".h5"])
        model_json_file = "".join([self.__model_dir, model_name,".json"])
        model_plot_name = "".join([self.__model_dir, model_name,"_plot.png"])
        model_layer_plot = "".join([self.__model_dir, model_name,"_layerplot.png"])

        if verbose:
            print("Looking for model JN")

        # Create or load the model
        if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
            if verbose:
                print("JN model file not found. Model creation begnining")

            GPU_count = len(tf.config.list_physical_devices('GPU'))

            #Try different values but use Adam
            #act = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.1, epsilon = 1e-8)
            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * l_epochs), 10])

            #Make sure to set early stopping so we aren't going on forever
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = model_file_name, verbose = verbose, save_best_only = True,
                mode = 'min', monitor = 'val_mae')

            model = Sequential()

            #Add layers
            model.add(Convolution2D(filters = 6, kernel_size = (3, 3), input_shape = (96, 96, 1)))
            model.add(ReLU())
            model.add(AveragePooling2D())

            if layers > 1:
                model.add(Convolution2D(filters = 16, kernel_size = (3, 3)))
                model.add(ReLU())
                model.add(AveragePooling2D())
                model.add(Flatten())
            if layers > 2:
                model.add(Dense(512))
                model.add(ReLU())
            if layers > 3:
                model.add(Dense(256))
                model.add(ReLU())
            if layers > 4:
                model.add(Dense(128))
                model.add(ReLU())

            model.add(Dense(30))

            if verbose:
                print(model.summary())

            compliled_model = model

            plot_model(model, to_file=model_layer_plot, show_shapes=True, show_layer_names=True)
            #play with these numbers.....
            #https://keras.io/api/models/model_training_apis/
            compliled_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if verbose:
                print("Compiling complete")

            #https://keras.io/api/models/model_training_apis/
            #Validation data will override validation split so okay to include both
            #This return a history object:
            # A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
            # Might be interesting to plot this....

            batch_size = l_batch_size * GPU_count
            print("Batch size:", batch_size)
            
            history = compliled_model.fit(X, Y, validation_split = l_validation_split, batch_size = l_batch_size * GPU_count, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)

            if verbose:
                print("Fitting complete")

            #Save all of this history information so we can inlude in our final report
            self.__save_history_info(history, model_name, model_plot_name, verbose = False)

            #Save the model and we can version these ... might want to make it so I can modify the names for different versions and configs??
            model_json = compliled_model.to_json()
            with open(model_json_file, "w") as json_file:
                json_file.write(model_json)

            if verbose:
                print(f"{model_name} model created and file saved for future use.")
        else:
            #We already have a model file, so retrieve and return it.
            #I would like to use this for future use. I can always make predictions.
            history_file = "".join([self.__model_dir, model_name,"_hist.csv"])
            history = pd.read_csv(history_file)
            model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)

        return model, history



    def __get_model_jcw(self,model_name, X, Y, l_batch_size, l_epochs, l_validation_split = .2, x_val = None, y_val = None, l_shuffle = True, verbose = True, separate = False):


        model_file_name = "".join([self.__model_dir, model_name,".h5"])
        model_json_file = "".join([self.__model_dir, model_name,".json"])
        model_plot_name = "".join([self.__model_dir, model_name,"_plot.png"])
        model_layer_plot = "".join([self.__model_dir, model_name,"_layerplot.png"])

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
            
            plot_model(model, to_file=model_layer_plot, show_shapes=True, show_layer_names=True)
        
            #play with these numbers.....
            #https://keras.io/api/models/model_training_apis/
            compiled_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if verbose:
                print("Compiling complete")

            #https://keras.io/api/models/model_training_apis/
            #Validation data will override validation split so okay to include both
            #This return a history object:

            history = compiled_model.fit(X, Y, validation_split = l_validation_split, batch_size = l_batch_size * GPU_count, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)

            if verbose:
                print("Fitting complete")

            #self.__save_history_info(history, model_name, model_plot_name, verbose = False)


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
            history_file = "".join([self.__model_dir, model_name,"_hist.csv"])
            history = pd.read_csv(history_file)
            model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)

        return model, history

    def __get_model_sp(self,model_name, X, Y, l_batch_size, l_epochs, l_validation_split = .2, x_val = None, y_val = None, l_shuffle = True, verbose = True, separate = False):
        #Number of features/outputs

        model_file_name = "".join([self.__model_dir, model_name,".h5"])
        model_json_file = "".join([self.__model_dir, model_name,".json"])
        model_plot_name = "".join([self.__model_dir, model_name,"_plot.png"])
        model_layer_plot = "".join([self.__model_dir, model_name,"_layerplot.png"])

        if verbose:
            print("Looking for model SP")

        # Create or load the model
        if (not os.path.isfile(model_file_name)) or (not os.path.isfile(model_json_file)):
            if verbose:
                print("SP model file not found. Model creation beginning")


            #create a model and return it? or save it?
            #act = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']

            #Used for the fitting step
            stop_at = np.max([int(0.1 * l_epochs), 10])
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = model_file_name, verbose = verbose, save_best_only = True,
                mode = 'min', monitor = 'val_mae')

            # Need to reshape X for my model to work
            X_reshape = X.reshape(-1, 96, 96, 1)

            model = Sequential()
            model.add(Conv2D(32, (3,3), input_shape=(96,96,1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Conv2D(64, (2,2)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size =(2,2)))

            model.add(Conv2D(64, (2,2)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size =(2,2)))

            model.add(Flatten())

            model.add(Dense(500))
            model.add(Activation('relu'))

            model.add(Dense(500))
            model.add(Activation('relu'))

            model.add(Dense(500))
            model.add(Activation('relu'))

            model.add(Dense(30))

            if verbose:
                print(model.summary())

            compiled_model = model
            
            plot_model(model, to_file=model_layer_plot, show_shapes=True, show_layer_names=True)

            compiled_model.compile(optimizer = sgd, loss = lss, metrics = mtrc)

            if verbose:
                print("Compiling complete")

            history = compiled_model.fit(X_reshape, Y, validation_split = l_validation_split, batch_size = l_batch_size, epochs = l_epochs, shuffle = l_shuffle, callbacks = [es, cp], verbose = verbose)
            if verbose:
                print("Fitting complete")

            self.__save_history_info(history, model_name, model_plot_name, verbose = False)

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
            history_file = "".join([self.__model_dir, model_name,"_hist.csv"])
            history = pd.read_csv(history_file)
            model = self.__load_model_from_file(model_name, model_file_name, model_json_file, verbose = True)


        return model, history


    ###### PUBLIC

    #######################################
    # PRINT PATHS
    #
    #
    #######################################
    def print_paths(self):
        print("Model dir:", self.__model_dir)




    #######################################
    # Train Models
    #
    #
    #######################################
    def train_model(self, model_name, train, split=True, X=None, Y=None,hoizontal_flip = False, dim = 0.3, brightness = 1.4,layers = 7, verbose = True):

        data_transform = transform_data.TransformData(verbose=True)

        #train_copy = train.copy()

        #Scale train
        train_scaled = data_transform.ScaleImages(train, verbose = True)

        #Flip the image if True
        if hoizontal_flip:
            train_scaled = data_transform.FlipHorizontal(train_scaled, verbose = True)


        #Bright_Dim(self, train, level_of_brightness = 1.4, level_to_dim = 0.3, verbose = False)
        train_scaled = data_transform.Bright_Dim(train_scaled, level_of_brightness = brightness, level_to_dim = dim,verbose = verbose)

        #Split train and scale accordingly
        # #do the split here and pass in parameters
        if(split):
            X, Y = data_transform.SplitTrain(train_scaled)
        elif X is None | Y is None:
            raise RuntimeError(f"When Split is set to False, X and Y must be supplied." )

        if "jn" in model_name:
            #Get and compile the model.
            model, history = self.__get_model_jn(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True,layers=layers)
        elif "jcw" in model_name:
            model, history = self.__get_model_jcw(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True)
        elif "sp" in model_name:
            model, history = self.__get_model_sp(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True)
        else:
            raise RuntimeError("Incorrect model name. Please verify and try again." )
        return model, history


    def train_jcw(self, model_name, train, split=True, X=None, Y=None, verbose = True, separate = False):
        if separate :
            #Only use some of the train
            train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                          'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
                          'image']
            train = train[train_cols].copy()


        data_transform = transform_data.TransformData(verbose=True)
        #Scale train
        train_scaled = data_transform.ScaleImages(train, verbose = True)
        print(train_scaled.columns)

        #Split train and scale accordingly
        # #do the split here and pass in parameters
        if(split):
            X, Y = data_transform.SplitTrain(train_scaled)
        elif X is None | Y is None:
            raise RuntimeError(f"When Split is set to False, X and Y must be supplied." )

        #Get and compile the model.
        model, history = self.__get_model_jcw(model_name, X = X, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True, separate = separate)

        return model, history

    def train_sp(self, model_name, train, split=True, X=None, Y=None, verbose=True, separate = False):
        if separate :
            #Only use some of the train
            train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                          'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
                          'image']
            train = train[train_cols].copy()

        data_transform = transform_data.TransformData(verbose=True)
        #Scale train
        train_scaled = data_transform.ScaleImages(train, verbose = True)

        #Split train and scale accordingly
        # #do the split here and pass in parameters
        if(split):
            X, Y = data_transform.SplitTrain(train_scaled)
        elif X is None | Y is None:
            raise RuntimeError(f"When Split is set to False, X and Y must be supplied." )

        # Need to reshape X for my model to work
        X_reshape = X.reshape(-1, 96, 96, 1)

        #Get and compile the model.
        model, history = self.__get_model_sp(model_name, X = X_reshape, Y = Y, l_batch_size = 128, l_epochs = 300, l_shuffle = True, separate = separate)

        return model, history
