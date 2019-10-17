# import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations

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

    return all_df, labels, ids

for i in range(1,4):
    fold_index = str(i)
    path_to_data = "../input"
    file_name = "fold"+fold_index+"Training.json"
    fname = os.path.join(path_to_data,file_name)

    # Read in all data in a single file
    selected_features = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ', \
    'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'TOTFY', \
    'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'R_VALUE', 'XR_MAX']
    all_input, labels, ids = convert_json_data_to_nparray(path_to_data, file_name, selected_features)

    # Change X and y to numpy.array in the correct shape.
    X = np.array(all_input)
    y = np.array([labels]).T
    print("The shape of X is (sample_size x time_steps x feature_num) = {}.".format(X.shape))
    print("the shape of y is (sample_size x 1) = {}, because it is a binary classification.".format(y.shape))

    test_frac = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=0, stratify=y)

    np.save('../input/X_train_'+fold_index+'.npy', X_train)
    np.save('../input/y_train_'+fold_index+'.npy', y_train)
    np.save('../input/X_test_'+fold_index+'.npy', X_test)
    np.save('../input/y_test_'+fold_index+'.npy', y_test)
