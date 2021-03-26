import os

DATA_PATH = '../data/'
NPY_PATH = os.path.join(DATA_PATH, 'npy/')
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train/')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test/')

for d in [DATA_PATH, NPY_PATH, TRAIN_IMG_PATH, TEST_IMG_PATH]:
    if not os.path.exists(d):
        os.mkdir(d)

IMG_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

