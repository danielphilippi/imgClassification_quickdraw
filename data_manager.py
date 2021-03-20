import quickdraw as qd
import numpy as np
from PIL import Image
import PIL.ImageOps
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def load_dataset(path, num_cat=345):
    qd_data = qd.QuickDrawData()
    img_list = []
    for category in qd_data.drawing_names[:num_cat]:
        img_list.append((load_category(path, category), category))
    
    return img_list

def load_category(path, category_name):
    return np.load(path+category_name+".npy")

def print_head_category(category):
    width = 120
    height = 120
    zoom = 1
    count = 0
    for item in category:
        clear_output(wait=True)
        count += 1
        a = item.reshape(28, 28)

        f_img = Image.fromarray(a)
        img = PIL.ImageOps.invert(f_img)
        plt.imshow(img.resize((zoom * width, zoom * height), Image.BICUBIC, reducing_gap=3))
        plt.show()

        time.sleep(.5)

        if count == 5:
            break