import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, multiply, Dropout, Activation, BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator


class Classifier():
    def __init__(self, input_shape, num_cat, optimizer):
        self.input_shape = input_shape
        self.num_cat = num_cat
        self.optimizer = optimizer

        self.model = self.build_classifier(self)

        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy'
        )

    def build_classifier(self):
        cnn_model = Sequential()

        cnn_model.add(Conv2D(32, (3,3), activation = 'relu',
                       input_shape=self.input_shape))
        cnn_model.add(MaxPooling2D((2,2)))
        cnn_model.add(Conv2D(64, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2,2)))
        cnn_model.add(Conv2D(128, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2,2)))
        cnn_model.add(Conv2D(128, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D((2,2)))
        
        cnn_model.add(Flatten())
        
        cnn_model.add(Dropout(0.5)) # add?

        cnn_model.add(Dense(512, activation='relu'))
        cnn_model.add(Dense(self.num_cat, activation='softmax'))

        # model.summary()     

        return cnn_model

    def train(self, train_dir, validation_dir):
        # rescale all images by 1/255
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                                train_dir, # target directory
                                target_size = (150,150), #resizes all images by 150 x 150
                                batch_size = 20,
                                class_mode='binary') # because you use binary_crossentropy, need binary labels
    
        validation_generator = train_datagen.flow_from_directory(
                                validation_dir, 
                                target_size = (150,150), 
                                batch_size = 20,
                                class_mode='binary')
        
        history = cnn_model.fit(train_generator, steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

        return history

    def save(model, model_name):

        model_path = "saved_model/%s.json" % model_name
        weights_path = "saved_model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                    "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    def metrics(self):

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label = 'Training acc')
        plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label = 'Training loss')
        plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    # TODO
    def report(self):
        pass