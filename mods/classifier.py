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

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D


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


def alexnet(input_shape, nb_classes):
    # https://github.com/duggalrahul/AlexNet-Experiments-Keras/blob/master/Code/alexnet_base.py
    # code adapted from https://github.com/heuritech/convnets-keras

    inputs = Input(shape=input_shape)


    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1', init='he_normal')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Convolution2D(128, 5, 5, activation="relu", init='he_normal', name='conv_2_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3', init='he_normal')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
        Convolution2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
        Convolution2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1', init='he_normal')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2', init='he_normal')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal')(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    alexnet = Model(input=inputs, output=prediction)

    return alexnet


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

    