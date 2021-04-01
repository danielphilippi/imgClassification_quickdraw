import os
import pandas as pd
joinpath = os.path.join

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(BASE_PATH)

print(f'set base dir to {BASE_PATH}')

DATA_PATH = joinpath(BASE_PATH, 'data/')
NPY_PATH = os.path.join(DATA_PATH, 'npy/')
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train/')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test/')

for d in [DATA_PATH, NPY_PATH, TRAIN_IMG_PATH, TEST_IMG_PATH]:
    if not os.path.exists(d):
        os.mkdir(d)

IMG_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

# saving models

MODELS_PATH = joinpath(DATA_PATH, 'models')

if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)

MODEL_OVERVIEW_FILE_REL = 'overview.csv'
MODEL_OVERVIEW_FILE_ABS = joinpath(MODELS_PATH, MODEL_OVERVIEW_FILE_REL)
if not os.path.isfile(MODEL_OVERVIEW_FILE_ABS):
    pd.DataFrame({
        'run_id': [],
        'path_rel': [],
        'accuracy': [],
        'duration': [],
        'date': [],
        'time': [],
        'user': [],
        'compare': []
    }).to_csv(MODEL_OVERVIEW_FILE_ABS, index=False, sep=';')

CONFIG_FILE_REL = 'config.json'
MODEL_FILE_REL = 'model.h5'
HISTORY_FILE_REL = 'history.pickle'
REPORT_FILE_REL = 'report.json'
CM_FILE_REL = 'cm.csv'
CM_PLOT_FILE_REL = 'cm.pdf'


