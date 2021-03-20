import models
from data_manager import load_dataset
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load the dataset
    path = "D:/mario/quickdraw/" # CHANGE PATH IF YOU ARE NOT MARIO ;)
    x_train, y_train, x_test, y_test = load_dataset(os.path(path))

    n_epochs = 10000    # Until convergence or infinity
    batch_size = 50

    acgan = models.ACGAN()
    acgan.train(x_train, y_train, n_epochs, batch_size)
