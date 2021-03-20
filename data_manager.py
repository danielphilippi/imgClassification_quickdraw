import quickdraw as qd
import numpy as np
from PIL import Image
import PIL.ImageOps
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

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