#https://arxiv.org/abs/1908.08681 - state of the art activation function, gives 
better results than relu across the board.

#https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras -
#example of how to implement a custom activation function

# Creating a model
from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'custom_activation': Activation(mish)})

"""
# Usage
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation(mish, name='mish'))
print(model.summary())
"""

""" 
#Loading a model with custom objects

# Required, as usual
from keras.models import load_model

# Recommended method; requires knowledge of the underlying architecture of the model
from keras_contrib.layers import PELU
from keras_contrib.layers import GroupNormalization

# Load our model
custom_objects = {'PELU': PELU, 'GroupNormalization': GroupNormalization}
model = load_model('example.h5', custom_objects)
"""

