from keras.models import Sequential
import tensorflow as tf
from keras import layers

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

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D



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

#https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8
def AlexNet(input_shape, nb_classes):
    model = Sequential()
    model.add(
        layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=input_shape))
    model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_classes, activation='softmax'))

    return model

# https://www.kaggle.com/kmader/quickdraw-simple-models
def AlexNetAdopted(input_shape, nb_classes):
    thumb_class_model = Sequential()
    thumb_class_model.add(BatchNormalization(input_shape=input_shape))
    thumb_class_model.add(Conv2D(16, (3, 3), padding='same'))
    thumb_class_model.add(Conv2D(16, (3, 3)))
    thumb_class_model.add(MaxPooling2D(2, 2))
    thumb_class_model.add(Conv2D(32, (3, 3), padding='same'))
    thumb_class_model.add(Conv2D(32, (3, 3)))
    thumb_class_model.add(MaxPooling2D(2, 2))
    thumb_class_model.add(Conv2D(64, (3, 3), padding='same'))
    thumb_class_model.add(Flatten())
    thumb_class_model.add(Dropout(0.5))
    thumb_class_model.add(Dense(256))
    thumb_class_model.add(Dense(nb_classes, activation='softmax'))
    return thumb_class_model


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

    cnn_model.add(Dense(512, activation='relu'))
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

def cnn_test_dm_avgpool3(input_shape, n_classes):
    
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (4,4), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(AveragePooling2D((3, 3)))
    cnn_model.add(Conv2D(64, (4, 4), padding='same',activation='relu'))
    cnn_model.add(AveragePooling2D((3, 3)))
    cnn_model.add(Conv2D(128, (4, 4),padding='same',activation='relu'))
    cnn_model.add(AveragePooling2D((3, 3)))
    
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
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Conv2D(256, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_layers2(input_shape, n_classes):
    cnn_model = Sequential()
    cnn_model.add(
        Conv2D(16, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(Conv2D(16,(3,3),padding='same', activation='relu'))
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

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_bn(input_shape, n_classes):

    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), activation='relu',
               input_shape=input_shape))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_test_dm_mp3(input_shape, n_classes):

    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (3, 3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((3, 3)))
    cnn_model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3)))
    cnn_model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3)))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?

    cnn_model.add(Dense(512, activation='relu'))
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
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.5))  # add?

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

def cnn_combo_1(input_shape, n_classes):

    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(32, (5,5), padding='same',activation='relu',
               input_shape=input_shape))
    cnn_model.add(AveragePooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(64, (4, 4),padding='same', activation='relu'))
    cnn_model.add(AveragePooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    cnn_model.add(AveragePooling2D((3, 3)))
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Flatten())

    #cnn_model.add(Dropout(0.5))  # add?
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model

def cnn_combo_2(input_shape,n_classes):
    cnn_model = Sequential()
    cnn_model.add(
        Conv2D(16, (3,3), padding='same', activation='relu',
               input_shape=input_shape))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Conv2D(256, (3, 3),padding='same',activation='relu'))
    cnn_model.add(MaxPooling2D((3, 3),padding='same'))
    cnn_model.add(Dropout(0.2))
    
    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.2))  # add?

    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dense(n_classes, activation='softmax'))

    # model.summary()
    return cnn_model
    


    