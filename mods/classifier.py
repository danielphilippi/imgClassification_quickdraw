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


def cnn_test_dp(input_shape, n_classes):
    # Define model
    model = Sequential()
    model.add(Conv2D(16, (3, 3),
                     padding='same',
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def cnn_test_dm():
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), activation='relu',
               input_shape=self.input_shape))
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
    cnn_model.add(Dense(self.num_cat, activation='softmax'))

    # model.summary()
    return cnn_model