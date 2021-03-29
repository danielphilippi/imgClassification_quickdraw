from definitions import *

import quickdraw as qd
import numpy as np
from PIL import Image
import PIL.ImageOps
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import os
import numpy as np
import urllib.request

from random import sample, seed
from shutil import rmtree
from PIL import Image

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def _download(class_name, base_path=IMG_BASE_URL, target_path=NPY_PATH):
    # https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e
    cls_url = class_name.replace('_', '%20')
    path = base_path + cls_url + '.npy'
    print(path)
    path, _ = urllib.request.urlretrieve(path, os.path.join(target_path, f'{class_name}.npy'))
    return path


def save_pngs(a, path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        rmtree(path)
        os.mkdir(path)

    for idx, aa in enumerate(a):
        f_img = Image.fromarray(aa.reshape(28, 28))
        f_img.save(os.path.join(path, f'{idx}.png'))


def reshape_1d(a_1d, class_name):
    x_train_4d = a_1d.reshape(a_1d.shape[0], 28, 28, 1)

    labels = np.full(a_1d.shape[0], class_name)

    # x_train_4d = x_train.reshape(x_train.shape[0], 28,28,1)
    # x_train_comb = np.concatenate((x_train_comb, x_train_4d), axis=0)

    # labels = np.full(x_train.shape[0], cl)
    # y_train = np.append(y, labels)

    return x_train_4d, labels


def to_img(array_4d):
    fft_p = array_4d.reshape(28, 28)
    new_p = Image.fromarray(fft_p)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    return new_p


def _prepare_img_for_generator(classes, test_ratio, max_imgs_per_class, mode='array'):
    seed(999)
    train_size = None

    if mode not in ['array', 'dir']:
        raise ValueError

    x_train = np.empty([0, 28, 28, 1])
    y_train = np.empty([0])
    x_test = np.empty([0, 28, 28, 1])
    y_test = np.empty([0])
    class_names = {}

    for cl_idx, cl in enumerate(classes):
        print('*' * 100)
        print(f'processing class **{cl}**')
        file_name = f'{cl}.npy'
        source_path = NPY_PATH

        if file_name not in os.listdir(source_path):
            print(f'download dara to {source_path}')
            tmp_path = _download(cl, target_path=source_path)
        else:
            print(f'loading data from buffer: {source_path}')
            tmp_path = os.path.join(source_path, file_name)
        x = np.load(tmp_path)
        print('Shape of raw data: ', x.shape)

        if max_imgs_per_class is not None:
            rand_idx = sample([i for i in range(x.shape[0])], max_imgs_per_class)
            x = x[rand_idx, :]

        # stratify: memorize abs train size of first class
        if train_size is None:
            train_size = int(x.shape[0] * (1 - test_ratio))

        # split array into train and test
        x_train_1d = x[0:train_size, :]
        x_test_1d = x[train_size:x.shape[0], :]
        print(f'Shape after sampling, shuffle and first split: train: {x_train_1d.shape} | test: {x_test_1d.shape}')

        if mode == 'dir':
            # save pngs to disk
            save_pngs(x_train_1d, os.path.join(TRAIN_IMG_PATH, cl))
            save_pngs(x_test_1d, os.path.join(TEST_IMG_PATH, cl))

            return 'ok'

        elif mode == 'array':

            x_train_tmp, y_train_tmp = reshape_1d(x_train_1d, cl_idx)
            x_train = np.concatenate((x_train, x_train_tmp), axis=0)
            y_train = np.append(y_train, y_train_tmp)

            x_test_tmp, y_test_tmp = reshape_1d(x_test_1d, cl_idx)
            x_test = np.concatenate((x_test, x_test_tmp), axis=0)
            y_test = np.append(y_test, y_test_tmp)

            class_names[cl_idx] = cl

    if mode == 'dir':
        return 'ok'
    elif mode == 'array':
        seed(22)
        rand_idx = sample([i for i in range(x_train.shape[0])], x_train.shape[0])
        x_train, y_train = x_train[rand_idx], y_train[rand_idx]

        num_classes = len(classes)
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return x_train, y_train, x_test, y_test, class_names


def build_set_generators(classes, max_imgs_per_class=10000, vali_ratio=.2, test_ratio=.2, batch_size=32):
    x_train, y_train, x_test, y_test, class_names = _prepare_img_for_generator(
        classes=classes,
        test_ratio=test_ratio,
        max_imgs_per_class=max_imgs_per_class
    )

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=vali_ratio)  # set validation split

    train_generator = train_datagen.flow(
        x=x_train,
        y=y_train,
        shuffle=False,
        batch_size=batch_size,
        subset='training'
    )  # set as training data

    validation_generator = train_datagen.flow(
        x=x_train,
        y=y_train,
        shuffle=False,
        batch_size=batch_size,
        subset='validation')  # set as validation data

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.0)  # set validation split

    test_generator = test_datagen.flow(
        x=x_test,
        y=y_test,
        shuffle=False,
        batch_size=batch_size
    )  # set as validation data

    return train_generator, validation_generator, test_generator, class_names


def load_dataset(path):
    #qd_data = qd.QuickDrawData()
    big_mamals = ["camel", "cow", "elephant", "giraffe", "horse", \
                  "kangoroo", "lion", "panda", "rhinoceros", "tiger", "zebra"]
    img_list = []
    for category in big_mamals: #qd_data.drawing_names[:num_cat]:
        img_list.append((load_category(path, category), category))

    # TODO
    # split in train, val, test

    return img_list

def load_category(path, category_name):
    return np.load(path+category_name+".npy")

def print_head_category(category):
    width = 250
    height = 250
    zoom = 1
    count = 0
    for item in category:
        clear_output(wait=True)
        count += 1
        a = item.reshape(28, 28)

        f_img = Image.fromarray(a)
        img = PIL.ImageOps.invert(f_img)
        plt.imshow(img, Image.BICUBIC, reducing_gap=3)
        plt.show()

        time.sleep(.5)

        if count == 5:
            break
