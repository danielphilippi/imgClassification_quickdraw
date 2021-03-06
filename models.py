import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, multiply, Dropout, Activation, BatchNormalization, \
    ZeroPadding2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

from mods.classifier import *

import json
from time import time
from definitions import *
import pandas as pd
import getpass
from datetime import datetime
import pickle
from contextlib import redirect_stdout

from utils import plots
from utils.helper import pformat, printlr


class ModelClass:
    def __init__(self):
        self.history = None
        self.report = None
        self.confusion_matrix = None
        self.model = None

    # TODO: picklyze this method
    def save_model(self):
        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.model, "classifier")

    def plot_model(self):
        acc = self.history['acc']
        val_acc = self.history['val_acc']
        loss = self.history['loss']
        val_loss = self.history['val_loss']

        epochs = range(1, len(acc) + 1)

        fig, axs = plt.subplots(2)

        axs[0].plot(epochs, acc, 'bo', label='Training acc')
        axs[0].plot(epochs, val_acc, 'b', label='Validation acc')
        axs[0].set_title('Training and validation accuracy')
        axs[0].legend()

        axs[1].plot(epochs, loss, 'bo', label='Training loss')
        axs[1].plot(epochs, val_loss, 'b', label='Validation loss')
        axs[1].set_title('Training and validation loss')
        axs[1].legend()

        fig.show()
        return fig

    def evaluate(self, test_set):
        # if you have the last version of tensorflow, the predict_generator is deprecated.
        # you should use the predict method.
        # if you do not have the last version, you must use predict_generator
        y_pred = self.model.predict(test_set, 63)  # ceil(num_of_test_samples / batch_size)
        y_pred = (y_pred > 0.5)
        """print('Confusion Matrix')
        self.confusion_matrix = confusion_matrix(test_set.classes, y_pred)
        print(self.confusion_matrix)"""
        print('Classification Report')
        # TODO: check if this work
        target_names = list(set(test_set.classes.keys))  # I'm pretty sure this is not the way
        self.report = classification_report(test_set.classes, y_pred, target_names=target_names)
        print(self.report)
        return self.report


class Generator(ModelClass):
    def __init__(self, input_shape, num_cat, latent_dim, optimizer, cnn, n_channels=1):
        super().__init__()
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.n_channels = n_channels
        self.cnn = cnn(self.latent_dim, self.n_channels)
        self.model = self.build_generator()

        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy'
        )

    def build_generator(self):
        cnn_model = self.cnn

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

    # TODO check if parent evaluate function is valid


class Discriminator(ModelClass):
    def __init__(self, input_shape, num_cat, optimizer, cnn):
        super().__init__()
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.optimizer = optimizer
        self.cnn = cnn(self.input_shape)
        self.model = self.build_discriminator()

        # First loss for fake-real classification. Second loss for categorical input classification
        self.model.compile(
            optimizer=self.optimizer,
            loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
        )

    def build_discriminator(self):
        cnn_model = self.cnn

        cnn_model.add(Flatten())

        d_input = Input(shape=self.input_shape)

        features = cnn_model(d_input)

        # Outputs whether if it's classified as fake or real input
        fake_real = Dense(1, activation='sigmoid', name='generated')(features)

        # Outputs the predicted category of the input
        target_label = Dense(self.num_cat, activation='softmax', name='category')(features)

        return Model(d_input, [fake_real, target_label])

    # TODO check if parent evaluate function is valid


class ACGAN(ModelClass):
    def __init__(self, cnn_gen, cnn_dis, input_shape=(28, 28, 1), num_cat=400, latent_dim=100):
        super().__init__()
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.latent_dim = latent_dim

        # Adam parameters suggested in https://arxiv.org/abs/1511.06434
        lr = 0.0002
        beta_1 = 0.5
        self.optimizer = Adam(lr, beta_1)

        self.discriminator = Discriminator(self.input_shape, self.num_cat, self.optimizer, cnn_dis)
        self.generator = Generator(self.input_shape, self.num_cat, self.latent_dim, self.optimizer, cnn_gen)

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
    def evaluate(self):
        pass

from keras.optimizers import Adam, SGD, RMSprop

class Classifier(ModelClass):
    def __init__(self, img_gen_config, model_config, input_shape=(28, 28, 1)):
        self.img_gen_config = img_gen_config
        self.model_config = model_config
        self.train_config = None

        self.input_shape = input_shape
        self.num_cat = len(img_gen_config['classes'])
        self.num_img = None
        self.class_names = None

        self.history = None
        self.model = globals()[model_config['classifier']](self.input_shape, self.num_cat)

        # Todo: research on metrics
        self.optimizer = self.build_optimizer()
        self.model.compile(optimizer=self.optimizer, **model_config['compiler'])
        self.mode_summary = self.model.summary()

        self.history_plot = None
        self.conf_mat = None
        self.report = None
        self.train_duration = None
        self.scores = None

        self.train_acc = None
        self.test_acc = None
        self.vali_acc = None

    # Todo move definition to mods.classifier and pass to the class
    def build_classifier(self, classifier):
        cnn_model = classifier()
        return cnn_model

    def build_optimizer(self):
        optimizer_n = list(self.model_config['optimizer'].keys())[0].lower()
        params = self.model_config['optimizer'][optimizer_n]
        if optimizer_n == 'adam':
            return Adam(**params)
        elif optimizer_n == 'sgd':
            return SGD(**params)
        elif optimizer_n == 'rmsprop':
            return RMSprop(**params)
        else:
            raise NotImplementedError('Selected optimizer not available. Please choose from [Adam, SGD, RMSprop]')

    # Todo: make args part of self
    def _train(self, train_set, test_set):
        self.num_img = train_set.shape[0]
        self.history = self.model.fit(
            train_set,
            steps_per_epoch=100,
            epochs=30,
            validation_data=test_set,
            validation_steps=50
        )
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

    # es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', min_delta=1, verbose=1, patience=3)

    def _generate_callbacks(self):
        callbacks = [printlr]
        for cb_name, params in self.train_config['callbacks'].items():
            if cb_name == 'early_stopping':
                es = EarlyStopping(**params)
                callbacks.append(es)
            elif cb_name == 'tensor_board':
                tb = TensorBoard(**params)
                callbacks.append(tb)
            elif cb_name == 'learning_rate_scheduler':
                lrs = LearningRateScheduler(**params)
                callbacks.append(lrs)
            elif cb_name == 'reduce_lr_plateau':
                rlrp = ReduceLROnPlateau(cooldown=1, **params)
                callbacks.append(rlrp)

        return callbacks

    def _train_from_generator(self, train_generator, validation_generator, train_config):
        self.train_config = train_config

        batch_size = self.img_gen_config['batch_size']

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // batch_size,
            epochs=self.train_config['n_epochs'],
            callbacks=self._generate_callbacks()
        )

    def _train_from_array(self, train_generator, validation_generator, train_config):
        self.train_config = train_config

        batch_size = self.img_gen_config['batch_size']

        self.history = self.model.fit(
            x=train_generator.x,
            y=train_generator.y,
            batch_size=batch_size,
            steps_per_epoch=train_generator.n // batch_size,
            validation_data=(validation_generator.x, validation_generator.y),
            validation_steps=validation_generator.n // batch_size,
            epochs=self.train_config['n_epochs'],
            callbacks=self._generate_callbacks()
        )

    def train(self, train_generator, validation_generator, train_config):
        rand_config = self.img_gen_config['train_img_randomization']
        if len(rand_config) == 0:
            _train = self._train_from_array
        else:
            _train = self._train_from_generator

        start = time()
        _train(train_generator, validation_generator, train_config)
        end = time()
        self.train_duration = np.round(end - start, 2)

    def _manage_model_overview(self):
        overview = pd.read_csv(MODEL_OVERVIEW_FILE_ABS, sep=';')

        if len(overview.run_id) == 0:
            run_id = 1
        else:
            run_id = overview.run_id.max() + 1

        run_name = f'run_{str(run_id).zfill(4)}'
        model_path_rel = f'run_{str(run_id).zfill(4)}/'
        model_path_abs = os.path.join(MODELS_PATH, model_path_rel)
        if not os.path.exists(model_path_abs):
            os.mkdir(model_path_abs)
        else:
            raise Exception('path to save model already exists!')

        overview_new = pd.DataFrame({
            'run_id': [run_id],
            'path_rel': [model_path_rel],
            'train_acc': round(self.train_acc, 4),
            'vali_acc': round(self.vali_acc, 4),
            'test_acc': round(self.test_acc, 4),
            'duration': [str(self.train_duration).replace('.', ',')],
            'date': [datetime.now().strftime("%Y-%m-%d")],
            'time': [datetime.now().strftime("%H:%M:%S")],
            'user': [getpass.getuser()],
            'compare': [None]
        })

        pd.concat([overview, overview_new]).to_csv(MODEL_OVERVIEW_FILE_ABS, index=False, sep=';')

        return run_name, model_path_abs

    def save(self):

        run_name, model_path_abs = self._manage_model_overview()

        # save config
        config = {
            'run': run_name.replace('run_', ''),
            'version': 0.2,
            'img_gen_config': self.img_gen_config,
            'model_config': self.model_config,
            'train_config': self.train_config,
            'class_names': self.class_names
        }
        with open(os.path.join(model_path_abs, CONFIG_FILE_REL), 'w') as fp:
            json.dump(config, fp, indent=4)

        # save model
        self.model.save(os.path.join(model_path_abs, MODEL_FILE_REL))

        # plot model
        # plot model as graph
        """
        plot_model(
            self.model,
            to_file=os.path.join(model_path_abs, f'architecture_{run_name}.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=300
        )
        """
        # plot model as summary
        with open(os.path.join(model_path_abs, SUMM_TEXT_FILE_REL), 'w') as f:
            with redirect_stdout(f):
                print(f'Summary {run_name}')
                self.model.summary()

        # save history
        with open(os.path.join(model_path_abs, HISTORY_FILE_REL), 'wb') as f:
            pickle.dump(self.history.history, f)

        # save report
        report = {
            'run': run_name.replace('run_', ''),
            'keras_scores': self.scores,
            'test_metrics': self.report
        }

        with open(os.path.join(model_path_abs, REPORT_FILE_REL), 'w') as fp:
            json.dump(report, fp, indent=4)

        # save history plot
        hist_plot = plots.plot_history(self.history.history, self.model_config, title=f'History - {run_name}')
        hist_plot.savefig(os.path.join(model_path_abs, HIST_PLOT_FILE_REL))

        # save confusion matrix
        pd.DataFrame(self.conf_mat).to_csv(os.path.join(model_path_abs, CM_FILE_REL), sep=';')

        cm_plot = plots.plot_confusion_matrix(
            cm=self.conf_mat,
            cmap=plt.cm.Greys,
            classes=self.class_names.values(),
            title=f'CM - {run_name}',
            normalize=False
        )
        cm_plot.savefig(os.path.join(model_path_abs, CM_PLOT_FILE_REL))

        # optionally save learning rate
        # not used; instead add lr to history plot
        if '_lr' in self.history.history.keys():
            lr_plot = plots.plot_lr(self.history.history, title=f'Learning rate - {run_name}')
            lr_plot.savefig(os.path.join(model_path_abs, LR_PLOT_FILE_REL))

    def evaluate(self, test_set):
        # predict
        y_pred = self.model.predict(test_set) # ceil(num_of_test_samples / batch_size)
        y_pred_classes = y_pred.argmax(axis=-1)

        # cm
        print('Confusion Matrix')
        y_true = test_set.y.argmax(axis=1)
        self.conf_mat = confusion_matrix(y_true, y_pred_classes)

        # report
        print('Classification Report')

        for output_dict in [False, True]:
            report = classification_report(
                y_true,
                y_pred_classes,
                labels=list(self.class_names.keys()),
                target_names=list(self.class_names.values()),
                output_dict=output_dict,
                digits=2)

            if output_dict:
                self.report = report
            else:
                print(report)

        # scores
        scores = {}
        for set_name in ['test_set']:
            data_set = test_set
            set_name = set_name.replace('_set', '')
            scores[set_name] = dict(zip(
                [f'{set_name}_loss', f'{set_name}_acc'],
                np.round(self.model.evaluate(data_set), 2)
            ))
        metric = self.model_config['compiler']['metrics'][0]

        scores['train'] = {
            'train_loss_best': (
                np.array(self.history.history['loss']).min(),
                np.array(self.history.history['loss']).argmin()),
            'train_loss_last': self.history.history['loss'][-1],
            'train_acc_best': (
                np.array(self.history.history[metric]).max(),
                np.array(self.history.history[metric]).argmax()),
            'train_acc_last': self.history.history[metric][-1]
        }
        scores['vali'] = {
            'vali_loss_best': (
                np.array(self.history.history['val_loss']).min(),
                np.array(self.history.history['val_loss']).argmin()),
            'vali_loss_last': self.history.history['val_loss'][-1],
            'vali_acc_best': (
                np.array(self.history.history[f'val_{metric}']).max(),
                np.array(self.history.history[f'val_{metric}']).argmax()),
            'vali_acc_last': self.history.history[f'val_{metric}'][-1]
        }

        self.scores = pformat(scores)
        self.train_acc = scores['train']['train_acc_last']
        self.vali_acc = scores['vali']['vali_acc_last']
        self.test_acc = scores['test']['test_acc']
        hist_df = pd.DataFrame(self.history.history)


