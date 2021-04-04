from keras.models import Sequential

from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Dense, Reshape, Dropout, Activation, BatchNormalization, ZeroPadding2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU


def gen_basic(latent_dim, n_channels):
    """CNN with no Batch Normalization as it is fed Normal Noise"""
    n_hidden_units = 128
    shape_latent = 7
    kernel_size = 3

    cnn_model = Sequential()

    cnn_model.add(Dense(n_hidden_units * shape_latent * shape_latent, activation="relu", input_dim=latent_dim))

    cnn_model.add(Reshape((shape_latent, shape_latent, n_hidden_units)))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(n_hidden_units, kernel_size=kernel_size, padding="same"))
    cnn_model.add(Activation("relu"))

    cnn_model.add(UpSampling2D())
    cnn_model.add(Conv2D(int(n_hidden_units/2), kernel_size=kernel_size, padding="same"))

    cnn_model.add(Conv2D(n_channels, kernel_size=kernel_size, padding='same'))
    cnn_model.add(Activation("tanh"))
    return cnn_model


def gen_default(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=3, padding="same"))
    cnn_model.add(Activation("relu"))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(64, kernel_size=3, padding="same"))
    cnn_model.add(Activation("relu"))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(n_channels, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    return cnn_model


def gen_1(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=3, padding="same"))
    cnn_model.add(Activation("relu"))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(64, kernel_size=3, padding="same"))
    cnn_model.add(Activation("relu"))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model


def gen_2(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(64, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model


def gen_3(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))
    #cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    #cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(64, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model


def gen_4(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(256, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(64, kernel_size=3, padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.2))
    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model


def gen_5(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(256, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=5, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(64, kernel_size=3, padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model


def gen_6(latent_dim, n_channels):
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    cnn_model.add(Reshape((7, 7, 128)))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(256, kernel_size=15, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=15, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(UpSampling2D())

    cnn_model.add(Conv2D(128, kernel_size=10, padding="same"))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=10, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(BatchNormalization(momentum=0.8))

    cnn_model.add(Conv2D(64, kernel_size=5, padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(DepthwiseConv2D(kernel_size=5, padding="same"))
    cnn_model.add(BatchNormalization(momentum=0.8))
    cnn_model.add(LeakyReLU(alpha=0.2))

    cnn_model.add(Dropout(0.25))

    cnn_model.add(Conv2D(32, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    cnn_model.add(Dense(32, activation="relu"))
    cnn_model.add(Dense(n_channels, activation="relu"))

    return cnn_model
