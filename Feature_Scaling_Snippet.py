# import packages
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations


# I delaborately list all hyperparameters to remind myself of the architecture and arguments for each layer.

# Hyperparameters
num_classes = 1
num_features = 14 # number of 'color' channels for the input 1D 'image'
time_steps = 60 # number of pixels for the input 1D 'image'

# all features
feature_names = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ',
                 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
                 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
# relevent features by Fischer ranking score and manually looking at the histograms in positive and negative classes.
# I choose those features with reletively high Fischer ranking score & those whose histograms look different in
# positive and negative classes.
# I further drop features according to their physical definitions. When some features' definitions are related to each
# other, I first double check their correlation by looking at their scatter plot. If their correlation is confirmed visually,
# I drop n number features from the correlated features if there are n confirmed correlations.
relevant_features_0 = ['TOTUSJH','TOTBSQ','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP','USFLUX','TOTFZ',
                     'EPSZ','MEANSHR','SHRGT45','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','R_VALUE']

# By observing the histograms of relevant features, their histograms can be grouped into four categories.
# right skewed with extreme outliers, right skewed without extreme outliers, left skewed with extreme outliers, non skewed
right_skewed_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'EPSZ', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_with_ol = ['TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'SAVNCPP', 'USFLUX', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_without_ol = ['TOTUSJH', 'ABSNJZH', 'EPSZ']
left_skewed_features_with_ol = ['TOTFZ']
non_skewed_features = ['MEANGBT', 'R_VALUE']

# I further decide that TOTFZ is correlated with EPSZ and TOTBSQ. Furthermore, TOTFZ cannot be well scaled yet, I
# decide to drop it for now. Note that TOTFZ is the only feature in the list `left_skewed_with_ol`. In the end, I select
# 14 features for fitting the data, their names are stored in the list called `selected_features`.
selected_features = right_skewed_features + non_skewed_features
print('{} are selected for training.'.format(len(selected_features)))
print('selected features include \n',selected_features)

# get the indice for features
indice_right_skewed_with_ol = []
indice_right_skewed_without_ol = []
indice_non_skewed = []
for i in range(0,len(selected_features)):
    if selected_features[i] in right_skewed_features_with_ol:
        indice_right_skewed_with_ol.append(i)
    elif selected_features[i] in right_skewed_features_without_ol:
        indice_right_skewed_without_ol.append(i)
    elif selected_features[i] in non_skewed_features:
        indice_non_skewed.append(i)


scale_params_right_skewed = pd.read_csv('scale_params_right_skewed.csv')
scale_params_right_skewed.set_index('Unnamed: 0', inplace=True)

scale_params_non_skewed = pd.read_csv('scale_params_non_skewed.csv')
scale_params_non_skewed.set_index('Unnamed: 0', inplace=True)


# all features
feature_names = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', 'MEANPOT', 'EPSZ',
                 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', 'MEANJZD',
                 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
# relevent features by Fischer ranking score and manually looking at the histograms in positive and negative classes.
# I choose those features with reletively high Fischer ranking score & those whose histograms look different in
# positive and negative classes.
# I further drop features according to their physical definitions. When some features' definitions are related to each
# other, I first double check their correlation by looking at their scatter plot. If their correlation is confirmed visually,
# I drop n number features from the correlated features if there are n confirmed correlations.
relevant_features_0 = ['TOTUSJH','TOTBSQ','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP','USFLUX','TOTFZ',
                     'EPSZ','MEANSHR','SHRGT45','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','R_VALUE']

# By observing the histograms of relevant features, their histograms can be grouped into four categories.
# right skewed with extreme outliers, right skewed without extreme outliers, left skewed with extreme outliers, non skewed
right_skewed_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'EPSZ', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_with_ol = ['TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'SAVNCPP', 'USFLUX', 'MEANSHR', 'MEANGAM', 'MEANGBH', 'MEANJZD']
right_skewed_features_without_ol = ['TOTUSJH', 'ABSNJZH', 'EPSZ']
left_skewed_features_with_ol = ['TOTFZ']
non_skewed_features = ['MEANGBT', 'R_VALUE']

# I further decide that TOTFZ is correlated with EPSZ and TOTBSQ. Furthermore, TOTFZ cannot be well scaled yet, I
# decide to drop it for now. Note that TOTFZ is the only feature in the list `left_skewed_with_ol`. In the end, I select
# 14 features for fitting the data, their names are stored in the list called `selected_features`.
selected_features = right_skewed_features + non_skewed_features
print('{} are selected for training.'.format(len(selected_features)))
print('selected features include \n',selected_features)

# get the indice for features
indice_right_skewed_with_ol = []
indice_right_skewed_without_ol = []
indice_non_skewed = []
for i in range(0,len(selected_features)):
    if selected_features[i] in right_skewed_features_with_ol:
        indice_right_skewed_with_ol.append(i)
    elif selected_features[i] in right_skewed_features_without_ol:
        indice_right_skewed_without_ol.append(i)
    elif selected_features[i] in non_skewed_features:
        indice_non_skewed.append(i)


scale_params_right_skewed = pd.read_csv('scale_params_right_skewed.csv')
scale_params_right_skewed.set_index('Unnamed: 0', inplace=True)

scale_params_non_skewed = pd.read_csv('scale_params_non_skewed.csv')
scale_params_non_skewed.set_index('Unnamed: 0', inplace=True)

# define read-in functions
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

#     df = pd.concat(all_df).reset_index(drop=True)
#     df = df.assign(LABEL=pd.Series(labels))
#     df = df.assign(ID=pd.Series(ids))
#     df.set_index([pd.Index(ids)])
    # Uncomment if you want to save this as CSV
    #df.to_csv(file_name + '_last_vals.csv', index=False)
    return all_df, labels, ids


print('Files contained in the ../input directiory include:')
print(os.listdir("../input"))

path_to_data = "../input"
file_name = "fold3Training.json"
fname = os.path.join(path_to_data,file_name)

# Read in all data in a single file
all_input, labels, ids = convert_json_data_to_nparray(path_to_data, file_name, selected_features)

# Change X and y to numpy.array in the correct shape.
X = np.array(all_input)
y = np.array([labels]).T
print("The shape of X is (sample_size x time_steps x feature_num) = {}.".format(X.shape))
print("the shape of y is (sample_size x 1) = {}, because it is a binary classification.".format(y.shape))

# Scale and fillin NaN
def scale_features(X, selected_features, nan_to=0.0):
    X_copy = X.copy() # make a copy of X, must use np.array.copy(), otherwise if use X_copy = X, X_copy would point to the same memory, and once X or X_copy gets changed, both will change.
    for i in range(0,len(selected_features)):
        feature = selected_features[i] # str, feature name
        # right skewed with extreme outliers
        if feature in right_skewed_features_with_ol:
            x_min, y_median, y_IQR = scale_params_right_skewed.loc[['x_min','y_median','y_IQR'],feature]
            x = X[:,:,i] # n_sample x time_steps x 1
#             x = np.nan_to_num(x, nan=nan_to)
            y = np.log(x - x_min + 1.0)
            z = (y - y_median)/y_IQR
            X_copy[:,:,i] = np.nan_to_num(z,nan=nan_to) #,posinf=inf_to,neginf=-inf_to)
        # right skewed without extreme outliers
        elif feature in right_skewed_features_without_ol:
            x_min, y_mean, y_std = scale_params_right_skewed.loc[['x_min','y_mean','y_std'],feature]
            x = X[:,:,i]
#             x = np.nan_to_num(x, nan=nan_to)
            y = np.log(x-x_min+1.0)
            z = (y - y_mean)/y_std
            X_copy[:,:,i] = np.nan_to_num(z,nan=nan_to) #,posinf=inf_to,neginf=-inf_to)
        # non_skewed features, they do not have extreme outliers
        elif feature in non_skewed_features:
            x_mean, x_std = scale_params_non_skewed.loc[['x_mean','x_std'],feature]
            x = X[:,:,i]
#             x = np.nan_to_num(x,nan=nan_to)
            X_copy[:,:,i] = np.nan_to_num((x - x_mean)/x_std,nan=nan_to) #,posinf=inf_to,neginf=-inf_to)
        else:
            print(feature+' is not found, and thus not scaled.')

    return X_copy

# Scale X
X_scaled = scale_features(X, selected_features)

# Save X_scaled and y
# np.save('../input/X_scaled.npy', X_scaled)
# np.save('../input/y.npy',y)

# Split train and validation
val_fraction = 0.33
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=val_fraction, random_state=0)
