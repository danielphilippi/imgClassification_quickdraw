from mods import models
from utils.data_manager import _prepare_img_for_generator

if __name__ == '__main__':
    # Load the dataset
    big_mamals = [
        "camel", "cow", "elephant", "giraffe", "horse",
        "kangaroo", "lion", "panda", "rhinoceros", "tiger", "zebra"
    ]
    n_classes = 11
    class_list = big_mamals[0:n_classes]

    # build img data generators
    img_gen_config = {
        'classes': class_list,
        'max_imgs_per_class': 6000,
        'vali_ratio': 0,
        'test_ratio': .2,
    }

    model_config = {
        'discriminator': 'dis_4_mod3',
        'dis_compiler': {
            'loss': ['binary_crossentropy', 'sparse_categorical_crossentropy'],
            'optimizer': 'adam',
        },
        'generator': 'gen_5',
        'gen_compiler': {
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['categorical_accuracy']
        }
    }

    train_config = {
        'n_epochs': 2001,
        'batch_size': 100,
    }

    x_train, y_train, x_val, y_val, x_test, y_test, class_names = _prepare_img_for_generator(**img_gen_config)

    acgan = models.ACGAN(num_cat=n_classes,
                         img_gen_config=img_gen_config,
                         model_config=model_config,
                         class_names=class_names)

    acgan.train(x_train, y_train, train_config)

    acgan.evaluate(x_test, y_test)

    acgan.save()

    acgan.sample_arrays(1000)
