import numbers
import collections
from tensorflow.keras import backend as k
import tensorflow as tf



def formatfloat(x):
    return float(x)


def pformat(dictionary, function=formatfloat):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, pformat(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, collections.Container):
        return type(dictionary)(pformat(value, function) for value in dictionary)
    if isinstance(dictionary, numbers.Number):
        return function(dictionary)
    return dictionary


# Define the Required Callback Function
# https://stackoverflow.com/questions/59680252/add-learning-rate-to-history-object-of-fit-generator-with-tensorflow
class printlearningrate(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = k.eval(optimizer.lr)
        Epoch_count = epoch + 1
        print('\n', "Epoch:", Epoch_count, ', LR: {:.4f}'.format(lr))


printlr = printlearningrate()
