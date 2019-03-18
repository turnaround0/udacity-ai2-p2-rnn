import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# Transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i: i + window_size] for i in range(0, len(series) - window_size)]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0: 2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# An RNN to perform regression on our time series input/output data
# layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
# layer 2 uses a fully connected module with one unit
def build_part1_RNN(window_size):
    model = Sequential()
    hidden_units = 5
    model.add(LSTM(hidden_units, input_shape=(window_size, 1)))
    model.add(Dense(1, activation='linear'))
    return model


# Return the text input with only ascii lowercase and the given punctuation
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    text = text.lower()
    chars = set(text)

    for char in chars:
        # do not allow extended ascii characters
        if not char.encode('utf-8').isalpha() and char not in punctuation:
            text = text.replace(char, ' ')

    return text

# Transform the input text
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i: i + window_size])
        outputs.append(text[i + window_size])

    return inputs, outputs

# 3 layer RNN model
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
# layer 1 should be an LSTM module with 200 hidden units
#  --> note this should have input_shape = (window_size, len(chars))
# layer 2 should be a linear module, fully connected, with len(chars) hidden units
# layer 3 should be a softmax activation (solving a multiclass classification)
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    hidden_units = 200
    model.add(LSTM(hidden_units, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))
    return model
