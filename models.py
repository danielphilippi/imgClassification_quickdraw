import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, multiply, Dropout, Activation, BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU


class Generator():
    def __init__(self, input_shape, num_cat, latent_dim, optimizer, n_channels=1):
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.n_channels = n_channels

        self.model = self.build_generator()

        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy'
        )

    def build_generator(self):

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


        # this is the z space commonly referred to in GAN papers
        latent_space = Input(shape=(self.latent_dim,))
        # this will be our label
        input_cat = Input(shape=(1,), dtype='int32')

        # Categories of the dataset
        category_embedding = Flatten()(Embedding(self.num_cat, self.latent_dim)(input_cat))

        # Element wise product between z-space and a class conditional embedding
        total_input = multiply([latent_space, category_embedding])

        complete_generator = cnn_model(total_input)

        return Model(inputs=[latent_space, input_cat], outputs=complete_generator)


class Discriminator():
    def __init__(self, input_shape, num_cat, optimizer):
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.optimizer = optimizer

        self.model = self.build_discriminator()

        # First loss for fake-real classification. Second loss for categorical input classification
        self.model.compile(
            optimizer=self.optimizer,
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

    def build_discriminator(self):
        cnn_model = Sequential()

        cnn_model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same"))
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

        cnn_model.add(Flatten())

        d_input = Input(shape=self.input_shape)

        features = cnn_model(d_input)

        # Outputs whether if it's classified as fake or real input
        fake_real = Dense(1, activation='sigmoid', name='generated')(features)

        # Outputs the predicted category of the input
        target_label = Dense(self.num_cat, activation='softmax', name='category')(features)

        return Model(d_input, [fake_real, target_label])


class ACGAN():
    def __init__(self, input_shape=(28, 28, 1), num_cat=400, latent_dim=100):
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.latent_dim = latent_dim

        # Adam parameters suggested in https://arxiv.org/abs/1511.06434
        lr = 0.0002
        beta_1 = 0.5
        self.optimizer = Adam(lr, beta_1)

        self.discriminator = Discriminator(self.input_shape, self.num_cat, self.optimizer)
        self.generator = Generator(self.input_shape, self.num_cat, self.latent_dim, self.optimizer)

        self.discriminator.model.trainable = False

        noise_space = Input(shape=(self.latent_dim,))
        input_cat = Input(shape=(1,))

        d_input = self.generator.model([noise_space, input_cat])

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        fake_real, target_label = self.discriminator.model(d_input)

        self.model = Model([noise_space, input_cat], [fake_real, target_label])
        self.model.compile(
            optimizer=self.optimizer,
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

    def train(self, x_train, y_train, n_epochs, batch_size, sample_interval=25):
        # Real-fake arrays of labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(n_epochs):
            ######################
            # Train discriminator#
            ######################

            # Vector of random noises
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Vector of labels of the random noises
            noise_cat_labels = np.random.randint(0, self.num_cat, (batch_size, 1))
            # Generates fake input from noises to feed the discriminator
            fake_input = self.generator.model.predict([noise, noise_cat_labels])

            # Takes a random batch of real inputs
            batch_index = np.random.randint(0, x_train.shape[0], batch_size)
            real_input, real_cat_labels = x_train[batch_index], y_train[batch_index]

            # Predicts real_fake with discriminator & calculates discriminator loss function to back propagate
            d_loss_fake = self.discriminator.model.train_on_batch(fake_input, [fake_labels, noise_cat_labels])
            d_loss_real = self.discriminator.model.train_on_batch(real_input, [real_labels, real_cat_labels])
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            ###################
            # Train Generator #
            ###################

            # Train the generator
            g_loss = self.model.train_on_batch([noise, noise_cat_labels], [real_labels, real_cat_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            self.report(epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0])
            # Saves model and generated images every `sample_interval` epochs
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.model.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator.model, "generator")
        save(self.discriminator.model, "discriminator")

    # TODO
    def report(self):
        return

class Classifier():
    def __init__(self):
        pass
    def build_model(self):
        pass
    def train(self):
        pass
    def report(self):
        pass