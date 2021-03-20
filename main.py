import models
from data_manager import load_dataset
import os

from mods.generator import gen_1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the dataset
    path = "" # CHANGE PATH IF YOU ARE NOT MARIO ;)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(os.path(path))

    n_epochs = 10000    # Until convergence or infinity
    batch_size = 50

    acgan = models.ACGAN(generator=gen_1)
    acgan.train(x_train, y_train, n_epochs, batch_size)

    """
    new_img = acgan.generator.model.predict()
    
    classifier = Classifier()
    classifier.train()
    
    acgan.discriminator.model.predict()
    classifier.model.predict()
    
    """