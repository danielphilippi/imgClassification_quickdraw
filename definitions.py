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
        'train_acc': [],
        'vali_acc': [],
        'test_acc': [],
        'duration': [],
        'date': [],
        'time': [],
        'user': [],
        'compare': []
    }).to_csv(MODEL_OVERVIEW_FILE_ABS, index=False, sep=';')

CONFIG_FILE_REL = '01_config.json'
MODEL_FILE_REL = '90_model.h5'
HISTORY_FILE_REL = '91_history.pickle'
REPORT_FILE_REL = '04_report.json'
CM_FILE_REL = '92_cm.csv'
CM_PLOT_FILE_REL = '05_cm.pdf'
HIST_PLOT_FILE_REL = '03_history.pdf'
SUMM_TEXT_FILE_REL = '02_modelsummary.txt'


