from keras.models import Sequential

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Dense, Reshape, Embedding, Dropout, Activation, BatchNormalization, ZeroPadding2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU


def dis_basic(input_shape):
    """CNN with no Batch Normalization as it is fed Normalized inputs"""
    kernel_size = 3
    drop_out = 0
    alpha_l_relu = 0.3  # Default value
    strides = 1
    n_hidden_units = 128
    cnn_model = Sequential()

    cnn_model.add(Conv2D(int(n_hidden_units/8), kernel_size=kernel_size, strides=strides, input_shape=input_shape, padding="same"))
    cnn_model.add(LeakyReLU(alpha=alpha_l_relu))
    cnn_model.add(Dropout(drop_out))

    cnn_model.add(Conv2D(int(n_hidden_units/4), kernel_size=kernel_size, strides=strides, padding="same"))

    cnn_model.add(LeakyReLU(alpha=alpha_l_relu))
    cnn_model.add(Dropout(drop_out))

    cnn_model.add(Conv2D(int(n_hidden_units/2), kernel_size=kernel_size, strides=strides, padding="same"))
    cnn_model.add(LeakyReLU(alpha=alpha_l_relu))
    cnn_model.add(Dropout(drop_out))

    cnn_model.add(Conv2D(n_hidden_units, kernel_size=kernel_size, strides=strides, padding="same"))
    cnn_model.add(LeakyReLU(alpha=alpha_l_relu))
    cnn_model.add(Dropout(drop_out))

    return cnn_model


def dis_default(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    return cnn_model


def dis_1(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(16, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(32, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_2(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(16, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(32, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_3(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_4(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_5(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_4_mod1(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=15, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=15, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_4_mod2(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=15, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=15, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=10, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=10, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_4_mod3(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=15, strides=6, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=15, strides=6, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=10, strides=4, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=10, strides=4, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.3))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(ZeroPadding2D())

    cnn_model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.3))

    cnn_model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model


def dis_4_mod3(input_shape):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, kernel_size=15, strides=2, input_shape=input_shape, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=15, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(64, kernel_size=10, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=10, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Dense(128, activation="relu"))
    cnn_model.add(Dense(128, activation="relu"))

    return cnn_model
