import tensorflow as tf
from tensorflow.keras.layers import *

class relaxed_softmax(Layer):
    def __init__(self, units=1):
        super(relaxed_softmax, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(self.units,),
                             initializer='ones',
                             trainable=True)
    def call(self, inputs):
        return tf.math.multiply(inputs, self.alpha)

