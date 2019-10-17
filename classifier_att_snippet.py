import keras
from keras.layers import *
from keras_self_attention import SeqSelfAttention
# https://pypi.org/project/keras-self-attention/
# pip install keras-self-attention

def classifier_att(alpha_cb, optimizer='adam',dropout=0.5):
    model = keras.models.Sequential()
    #model.add(keras.layers.LSTM(units=hidden_size, input_shape=(time_steps,feature_num), return_sequences=True))
    #model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size,return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size,return_sequences=True),input_shape=(time_steps,feature_num)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(keras.layers.LSTM(units=hidden_size,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(int(hidden_size/2), activation=mish)))
    model.add(Flatten())
    model.add(Dense(y_dim)) # Dense layer has y_dim=1 or 2 neuron.
    model.add(Activation('softmax'))

    #model.compile(loss=wrapped_loss(alpha_cb), optimizer=optimizer, metrics=[f1_score])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[f1_score])
    return model
