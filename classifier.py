import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, multiply, Dropout, Activation, BatchNormalization,\
    ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
from data_manager import load_dataset


# Todo move to models.py
class Classifier:
    def __init__(self, img_gen_config, classifier, optimizer, input_shape=(28, 28, 1)):
        self.img_gen_config = img_gen_config
        self.input_shape = input_shape
        self.num_cat = len(img_gen_config['classes'])
        self.num_img = None
        self.class_names = None

        self.optimizer = optimizer
        self.history = None
        self.model = classifier(self.input_shape, self.num_cat)
        self.train_config = None

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['categorical_accuracy']
        )
        self.mode_summary = self.model.summary()

        self.conf_mat = None
        self.report = None

    # Todo move definition to mods.classifier and pass to the class
    def build_classifier(self, classifier):
        cnn_model = classifier()
        return cnn_model

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

    def train(self, train_generator, validation_generator, train_config):
        self.train_config = train_config
        callbacks = []
        if 'early_stopping' in train_config['callbacks'].keys():
            es = EarlyStopping(**train_config['callbacks']['early_stopping'])
            callbacks.append(es)

        batch_size = self.img_gen_config['batch_size']

        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // batch_size,
            epochs=self.train_config['n_epochs'],
            callbacks=callbacks
        )

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

    # Todo: move to helper fcts module
    def plot_model(self):

        acc = self.history['acc']
        val_acc = self.history['val_acc']
        loss = self.history['loss']
        val_loss = self.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Todo: generate instance fig and return fig
        fig = plt.figure()

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.plot(epochs, loss, 'bo', label = 'Training loss')
        plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    def evaluate(self, test_set):
        # if you have the last version of tensorflow, the predict_generator is deprecated.
        # you should use the predict method.
        # if you do not have the last version, you must use predict_generator
        y_pred = self.model.predict(test_set, 63) # ceil(num_of_test_samples / batch_size)
        y_pred = (y_pred>0.5)
        print('Confusion Matrix')
        self.conf_mat = confusion_matrix(test_set.classes, y_pred)
        print(self.conf_mat)
        print('Classification Report')
        target_names = ['Cats', 'Dogs']
        self.report = classification_report(test_set.classes, y_pred, target_names=target_names)
        print(self.report)



