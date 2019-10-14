
# to suppress future warning from tf + np 1.17 combination.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
#runtimewarning is from powertransformer
warnings.filterwarnings('ignore',category=RuntimeWarning)

import sys
import resource

epsilon = 1e-5

# libraries for read in data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations
import matplotlib.pyplot as plt
# %matplotlib inline

# libraries needed for machine learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.preprocessing import PowerTransformer as pt, StandardScaler as ss, MinMaxScaler as mms, RobustScaler as rs, FunctionTransformer as ft, power_transform

from sklearn.compose import ColumnTransformer as ct

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Sequential, load_model
#from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM, Flatten, Bidirectional
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import pickle

import multiprocessing as mp

FLOAT_TYPE = 'float64'
K.set_floatx(FLOAT_TYPE)

# all features
feature_names = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
# we select all features
selected_features = feature_names

# Functions for reading in data from .json files
def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        # line is a long string with data type `str`
        if not match:
            # if the line is full of white space, get out of this func
            return
        # pos will be the position for the first non-white-space character in the `line`.
        pos = match.start()
        try:
            # JSONDecoder().raw_decode(line,pos) would return a tuple (obj, pos)
            # obj is a dict, and pos is an int
            # not sure how the pos is counted in this case, but it is not used anyway.
            obj, pos = decoder.raw_decode(line, pos)
            # obj = {'id': 1, 'classNum': 1, 'values',feature_dic}
            # here feature_dic is a dict with all features.
            # its key is feature name as a str
            # its value is a dict {"0": float, ..., "59": float}
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
            # read about difference between yield and return
            # with `yield`, obj won't be assigned until it is used
            # Once it is used, it is forgotten.
        yield obj

def get_obj_with_last_n_val(line, n):
    # since decode_obj(line) is a generator
    # next(generator) would execute this generator and returns its content
    obj = next(decode_obj(line))  # type:dict
    id = obj['id']
    class_label = obj['classNum']
    data = pd.DataFrame.from_dict(obj['values'])  # type:pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    last_n_indices = np.arange(0, 60)[-n:]
    data = data.loc[last_n_indices]
    return {'id': id, 'classType': class_label, 'values': data}

def convert_json_data_to_nparray(data_dir: str, file_name: str, features):
    """
    Generates a dataframe by concatenating the last values of each
    multi-variate time series. This method is designed as an example
    to show how a json object can be converted into a csv file.
    :param data_dir: the path to the data directory.
    :param file_name: name of the file to be read, with the extension.
    :return: the generated dataframe.
    """
    fname = os.path.join(data_dir, file_name)
    all_df, labels, ids = [], [], []
    with open(fname, 'r') as infile: # Open the file for reading
        for line in infile:  # Each 'line' is one MVTS with its single label (0 or 1).
            obj = get_obj_with_last_n_val(line, 60) # obj is a dictionary
            # if the classType in the sample is NaN, we do not read in this sample
            if np.isnan(obj['classType']):
                pass
            else:
                # a pd.DataFrame with shape = time_steps x number of features
                # here time_steps = 60, and # of features are the length of the list `features`.
                df_selected_features = obj['values'][features]
                # a list of np.array, each has shape=time_steps x number of features
                # I use DataFrame here so that the feature name is contained, which we need later for
                # scaling features.
                all_df.append(np.array(df_selected_features))
                labels.append(obj['classType']) # list of integers, each integer is either 1 or 0
                ids.append(obj['id']) # list of integers
    return all_df, labels, ids

print('Files contained in the ../input directiory include:')
print(os.listdir("./input"))

path_to_data = "./input"
file_name = "fold3Training.json"
#file_name_test = "testSet.json"


def diff_func(x):
    # x is 2d array
    return np.diff(x, axis=0)

def timeseries_detrending(X):
    # X is np.array, 3D
    X_2D = [*zip(X[i] for i in range(X.shape[0]))]
    with mp.Pool() as pool:
        X_new = pool.starmap(diff_func, X_2D)
    return np.asarray(X_new)

#rescaling to range 0-1
def minmax_func(x):
    # x is 2d array
    return (x - x.min(axis=0))/(x.max(axis=0) - x.min(axis=0) + epsilon)

def timeseries_normalization(X):
    # X is np.array, 3D
    X_2D = [*zip(X[i] for i in range(X.shape[0]))]
    with mp.Pool() as pool:
        X_new = pool.starmap(minmax_func, X_2D)
    return np.asarray(X_new)

def impute_func(x):
    # x is 2d array
    return IterativeImputer().fit_transform(x)

def timeseries_imputation(X):
    # X is np.array, 3D
    X_2D = [*zip(X[i] for i in range(X.shape[0]))]
    with mp.Pool() as pool:
        X_new = pool.starmap(impute_func, X_2D)
    return np.asarray(X_new)

def powertransform_func(x):
    # x is 2d array
    return pt().fit_transform(x)

def timeseries_powertransformation(X):
    # X is np.array, 3D
    X_2D = [*zip(X[i] for i in range(X.shape[0]))]
    with mp.Pool() as pool:
        X_new = pool.starmap(powertransform_func, X_2D)
    return np.asarray(X_new)

imputer_per_sample = make_pipeline(ft(timeseries_imputation, validate=False))

preprocessor_per_sample = make_pipeline(imputer_per_sample, ft(timeseries_powertransformation, validate=False), ft(timeseries_detrending, validate=False), ft(timeseries_normalization, validate=False))

preprocessor_per_timestep = make_pipeline(pt(), mms())

"""
## Run this commented part only once, so you are able to save the pickled files. Then comment it out.

# Read in all data in a single file
all_input, labels, ids = convert_json_data_to_nparray(path_to_data, file_name, selected_features)

all_input_test, labels_test, ids_test = convert_json_data_to_nparray(path_to_data, file_name_test, selected_features)


# Change X and y to numpy.array in the correct shape.
X = np.array(all_input)
y = np.array([labels]).T
print("The shape of X is (sample_size x time_steps x feature_num) = {}.".format(X.shape))
print("the shape of y is (sample_size x 1) = {}, because it is a binary classification.".format(y.shape))

#X_test = np.array(all_input_test)
#y_test = np.array([labels_test]).T
#print("The shape of X_test is (sample_size x time_steps x feature_num) = {}.".format(X_test.shape))
#print("the shape of y_test is (sample_size x 1) = {}, because it is a binary classification.".format(y_test.shape))


X = preprocessor_per_sample.fit_transform(X)
#X_test = preprocessor_per_sample.fit_transform(X_test)

# write to pickle
pickle.dump(X, open( "trainingset3_modified.pkl", "wb" ) )
#pickle.dump(X_test, open( "testset_modified.pkl", "wb" ) )
pickle.dump(y, open( "trainingset3_output_modified.pkl", "wb" ) )
#pickle.dump(y_test, open( "testset_output_modified.pkl", "wb" ) )


"""

# read from pickle
X = pickle.load( open( "trainingset3_modified.pkl", "rb" ) )
#X_test = pickle.load( open( "testset_modified.pkl", "rb" ) )
y = pickle.load( open( "trainingset3_output_modified.pkl", "rb" ) )
#y_test = pickle.load( open( "testset_output_modified.pkl", "rb" ) )
labels = y.copy()


# Define metric, which does not depend on imbalance of positive and negative classes in validation/test set
# Defining sensitivity = true_positive/(total real positive) = tp/(tp+fn)
# sensitivity is the same as recall
def sensitivity(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred)) 
    # K.clip(x,a,b) x is a tensor, a and b are numbers, clip converts any element of x falling
    # below the range [a,b] to a, and any element of x falling above the range [a,b] to b.
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # K.epsilon >0 & <<< 1, in order to avoid division by zero.
    sen = recall = true_positives / (possible_positives + K.epsilon())
    return sen

# Specificity = true_negative/(total real negative) = tn/(tn+fp)
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    spec = true_negatives / (possible_negatives + K.epsilon())
    return spec

# Precision = true_positives/predicted_positives = tp/(tp+fp)
def precision(y_true, y_pred):
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred)) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec

# Informedness = sensitivity + specificity - 1
def informedness(y_true, y_pred):
    return sensitivity(y_true, y_pred)+specificity(y_true, y_pred)-1

# f1 = 2/((1/precision) + (1/recall))
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    sen = sensitivity(y_true, y_pred)
    f1 = 2*((prec*sen)/(prec + sen + K.epsilon()))
    return f1


"""
yt = np.array([[[1,0],[1,0],[0,1]]])
yp = np.array([[[.2,.8],[.7,.3],[.5,.5]]])

k_yt = K.variable(value=yt)#, dtype='float64')
k_yp = K.variable(value=yp)#, dtype='float64')

specificity(k_yt,k_yp)
precision(k_yt,k_yp)
f1_score(k_yt,k_yp)
"""


alpha = 0.5
gamma = 2
beta_cb = 0.99
# alpha, gamma, and beta_cb are to be set by CV
unique_targets, unique_targets_cnts = np.unique(labels, return_counts=True)
#y_cnts is basically normalized_unique_targets_cnts
y_cnts = unique_targets_cnts/len(labels)
alpha_cb = (1-beta_cb)/(1-beta_cb**y_cnts)
alpha_cb_norm_fac = len(unique_targets)/np.sum(np.unique(alpha_cb))
alpha_cb *= alpha_cb_norm_fac
alpha_cb = np.array(alpha_cb, dtype=FLOAT_TYPE)



# check NaN in y, X #, X_scaled
print('There are {} NaN in y.'.format(np.isnan(y).sum()))
print('There are {} NaN in X.'.format(np.isnan(X).sum()))
#print('There are {} NaN in X_scaled.'.format(np.isnan(X_scaled).sum()))


# one-hot encode y
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y = np.asarray(onehot_encoder.fit_transform(y), dtype=FLOAT_TYPE)
y_dim = np.shape(y)[1] # y=0 if no flare, y=1 if flare



def wrapped_loss(alpha_cb, alpha=alpha, gamma=gamma):
    def weighted_crossentropy(targets, inputs):
        # we use a modulating factor which down-weights the loss assigned to well-classified examples to prevent numerous easy examples from overwhelming the classifier.
        y_class = K.argmax(inputs, axis=1)
        w = tf.gather(alpha_cb, y_class)
        bce = K.categorical_crossentropy(targets, inputs)
        bce_exp = K.exp(-bce)
        # focal loss
        fl = K.mean(w * K.pow((1-bce_exp), gamma) * bce)
        return fl
    return weighted_crossentropy



def mish(x):
    return x*K.tanh(K.softplus(x))

# Build LSTM networks using keras
num_folds = 5
num_epochs = 50



# Set some hyperparameters
n_sample = len(y)
time_steps = X.shape[1]#60
batch_size = 256
feature_num = len(selected_features) # 25 features per time step
hidden_size = feature_num
use_dropout = True
use_callback = False # to be added later




def classifier(alpha_cb, optimizer='adam',dropout=0.5):
    model = Sequential()
    model.add(LSTM(units=hidden_size, input_shape=(time_steps,feature_num), return_sequences=True))
    model.add(LSTM(units=hidden_size, return_sequences=True))
    #model.add(Bidirectional(LSTM(units=hidden_size, return_sequences=True), merge_mode = 'ave'))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(int(hidden_size/2), activation=mish)))
    model.add(Flatten())
    model.add(Dense(y_dim)) # Dense layer has y_dim=1 or 2 neuron.
    model.add(Activation('softmax'))
    #model.compile(loss=wrapped_loss(alpha_cb), optimizer=optimizer, metrics=[f1_score])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[f1_score])
    return model


# Split X, y into training and validation sets
# define k-fold cross validation test harness
seed = 10
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
cvscores = []


for train, val in kfold.split(np.asarray(labels), np.asarray(labels)):
    X_train = X[train]
    X_val = X[val]
    y_train = y[train]
    y_val = y[val]
    #"""
    for i in range(time_steps):
        _ = preprocessor_per_timestep.fit(X_train[:,i])
        X_train[:,i] = preprocessor_per_timestep.transform(X_train[:,i])
        X_val[:,i] = preprocessor_per_timestep.transform(X_val[:,i])
    """
    #'Class Balanced Loss Based on Effective Number of Samples
    labels_kfold = np.argmax(y_train, axis=1)
    unique_targets, unique_targets_cnts = np.unique(labels_kfold, return_counts=True)
    #y_cnts is basically normalized_unique_targets_cnts
    y_cnts = unique_targets_cnts/len(labels_kfold)
    alpha_cb = (1-beta_cb)/(1-beta_cb**y_cnts)
    alpha_cb_norm_fac = len(unique_targets)/np.sum(np.unique(alpha_cb))
    alpha_cb *= alpha_cb_norm_fac
    alpha_cb = np.array(alpha_cb, dtype=FLOAT_TYPE)
    """
    clf = KerasClassifier(classifier, alpha_cb=alpha_cb, optimizer='adam', epochs=num_epochs, batch_size=batch_size, verbose=1, validation_data=(X_val,y_val))
    history = clf.fit(X_train, y_train)
    cvscores.append(history.history['val_f1_score'][-1] * 100)

