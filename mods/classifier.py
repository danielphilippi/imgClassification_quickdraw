from keras.models import Sequential

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Dense, Reshape, Embedding, Dropout, Activation, BatchNormalization, ZeroPadding2D, \
    MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU


def cnn_1(input_shape, num_cat):
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(num_cat, activation='tanh'))

    return cnn_model
