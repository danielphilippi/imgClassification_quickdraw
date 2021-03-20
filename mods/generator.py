


def gen_1():
    cnn_model = Sequential()

    cnn_model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
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
    cnn_model.add(Conv2D(self.n_channels, kernel_size=3, padding='same'))
    cnn_model.add(Activation("tanh"))

    return cnn_model