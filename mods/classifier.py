from keras.models import Sequential

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Dense, Reshape, Embedding, Dropout, Activation, BatchNormalization, ZeroPadding2D, \
    MaxPooling2D, Flatten, AveragePooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from keras import losses


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
    """
    https://stackoverflow.com/questions/46714075/keras-conv1d-printing-plotting-information-of-kernel-size-on-summary-or-us
    You should decide the kernel size based on the size of the patterns.
    A kernel size of 1 is nothing more than joining channels together,
    but without looking for lengthy patterns.
    It's almost the same as using a dense layer in 3D data. Often the best size is 3,
    and you stack a series of convolutions and maxpoolings to detect larger patterns.
    In convolutoinal layers, the "filters" are compared to neurons

    :param input_shape:
    :param n_classes:
    :return:
    """
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


def cnn_test_dm(input_shape, n_classes):

    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_fs(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (4,4), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (4, 4), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (4, 4),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_avgpool(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (4,4), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(AveragePooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (4, 4), padding='same',activation='relu'))
    cnn_model.add(AveragePooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (4, 4),padding='same',activation='relu'))
    cnn_model.add(AveragePooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_layers1(input_shape, n_classes):
    cnn_model = Sequential()
    cnn_model.add(
        Conv2D(16, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_layers2(input_shape, n_classes):
    cnn_model = Sequential()
    cnn_model.add(
        Conv2D(16, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    cnn_model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_bn(input_shape, n_classes):

    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), activation='relu',
               input_shape=input_shape))
    
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_do(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.2))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_do2(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((2, 2),strides=2))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2),strides=2))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2),strides=2))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.2))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_do3(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.2))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

    