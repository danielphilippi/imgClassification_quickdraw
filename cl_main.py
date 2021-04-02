from data_manager import build_set_generators
from models import Classifier
from mods.classifier import cnn_test_dp
import json
import os
from definitions import *
import warnings

if __name__ == '__main__':
    # define classes if imgs to load

    config_manual = True
    config_path_rel = 'data/models/run_010/config.json'
    config_path = os.path.join(BASE_PATH, config_path_rel)

    if config_manual:
        big_mamals = [
            "camel", "cow", "elephant", "giraffe", "horse",
            "kangaroo", "lion", "panda", "rhinoceros", "tiger", "zebra"
        ]
        n_classes = 3
        class_list = big_mamals[0:n_classes]

        img_gen_config = {
            'classes': class_list,
            'max_imgs_per_class': 100,
            'vali_ratio': .2,
            'test_ratio': .2,
            'batch_size': 32,
            'data_augmentation': None, # include GAN imgs
            'train_img_randomization': {} # args passed to ImageDataGenerator
        }

        model_config = {
            'classifier': 'cnn_test_dp',
            'compiler': {
                'loss': 'categorical_crossentropy',
                'optimizer': 'adam',
                'metrics': ['categorical_accuracy']
            }
        }

        train_config = {
            'n_epochs': 50,
            'callbacks': {
                'early_stopping': {
                    'monitor': 'val_loss',
                    'mode': 'min',
                    'verbose': 1,
                    'patience': 5
                }

            }
        }
    else:
        # load
        with open(config_path, 'r') as f:
            config = json.load(f)
        # split
        img_gen_config = config['img_gen_config']
        model_config = config['model_config']
        train_config = config['train_config']
        # warn
        warn_message = f'Using config from {config_path_rel}'
        warnings.warn(warn_message)


    # build img data generators
    train_generator, validation_generator, test_generator, class_names = build_set_generators(**img_gen_config)

    # define model
    cl = Classifier(
        img_gen_config=img_gen_config,
        model_config=model_config,
        input_shape=train_generator.x.shape[1:]
    )
    cl.class_names = class_names


    # cl.train(train_generator, validation_generator, train_config)
    cl.train(train_generator, validation_generator, train_config)

    # eval
    cl.evaluate(test_generator)

    # save
    cl.save()


