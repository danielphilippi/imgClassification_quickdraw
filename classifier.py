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
from sklearn.metrics import classification_report, confusion_matrix
from data_manager import load_dataset


# Todo move to models.py
class Classifier:
    def __init__(self, classifier, optimizer, input_shape=(28, 28, 1), num_cat=11):
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.num_img = None
        self.optimizer = optimizer
        self.history = None
        self.model = self.build_classifier(classifier)

        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy'
        )

        self.conf_mat = None
        self.report = None

    # Todo move definition to mods.classifier and pass to the class
    def build_classifier(self, classifier):
        cnn_model = Sequential()

        cnn_model.add(
            Conv2D(32, (3, 3), activation='relu',
                   input_shape=self.input_shape))
        cnn_model.add(MaxPooling2D((2, 2)))
        cnn_model.add(Conv2D(64, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2, 2)))
        cnn_model.add(Conv2D(128, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2, 2)))
        cnn_model.add(Conv2D(128, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2, 2)))
        
        cnn_model.add(Flatten())
        
        cnn_model.add(Dropout(0.5)) # add?

        cnn_model.add(Dense(512, activation='relu'))
        cnn_model.add(Dense(self.num_cat, activation='softmax'))

        # model.summary()     

        return cnn_model

    @staticmethod
    # Todo: move to data_manager module
    def preprocess(self, train_dir, validation_dir):
        # rescale all images by 1/255
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Todo(DP): limit number of imgs passed from the dir to the generator
        # Todo (DP): train / test split inside ImageDataGenerator
        # Todo (DP): 
        train_set = train_datagen.flow_from_directory(
                                train_dir, # target directory
                                target_size = (28, 28),
                                batch_size = 20,
                                class_mode='categorical')
    
        test_set = test_datagen.flow_from_directory(
                                validation_dir, 
                                target_size = (28, 28),
                                batch_size = 20,
                                class_mode='categorical')

        train_set.class_indices

        return train_set, test_set

    # Todo: make args part of self
    def train(self, train_set, test_set):
        self.num_img = train_set.shape[0]
        self.history = self.model.fit(
            train_set,
            steps_per_epoch=100,
            epochs=30,
            validation_data=test_set,
            validation_steps=50
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

        plt.plot(epochs, acc, 'bo', label = 'Training acc')
        plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
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



