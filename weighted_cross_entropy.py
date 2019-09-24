import keras
import keras.backend as K

# https://arxiv.org/pdf/1708.02002.pdf

alpha = 0.5
gamma = 2
# alpha and gamma are to be set by CV

def weighted_cross_entropy(targets, inputs, alpha=alpha, gamma=gamma):    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    wce = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)
    return wce

