from definitions import *
import tensorflow as tf
import keras
import numpy as np
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import os


def get_last_conv_layer_name(model):
    last_conv_layer_name = ''
    for l in model.layers:
        if 'conv' in l.name:
            last_conv_layer_name=l.name
    return last_conv_layer_name


def make_gradcam_heatmap(img_array, model, pred_index=None):
    last_conv_layer_name = get_last_conv_layer_name(model)
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = img.copy()
    img *= 255 / img.max()

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap).copy()

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))

    return superimposed_img


if __name__ == '__main__':
    pass

"""
    heatmap_test_path = 'data/test_heatmap/'
    if not os.path.exists(heatmap_test_path):
        os.mkdir(heatmap_test_path)

    img_npy_file = os.path.join(heatmap_test_path, 'test.npy')
    with open(img_npy_file, 'rb') as f:
        img = np.load(f)

    model_file = os.path.join(heatmap_test_path, 'model.h5')
    model = keras.models.load_model(model_file)

    heatmap = make_gradcam_heatmap(img, model)
    # plt.imshow(heatmap)

    cam = save_and_display_gradcam(img.reshape(28, 28, 1), heatmap, alpha=.9)
    plt.imshow()

    r = 3
    c = 10
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            heatmap = make_gradcam_heatmap(img, model)
            cam = save_and_display_gradcam(img.reshape(28, 28, 1), heatmap, alpha=.9)

            axs[i, j].imshow(np.array(cam))
            axs[i, j].axis('off')
            cnt += 1
    fig.show()
    fig.savefig("t.pdf")

    cam_path = os.path.join(heatmap_test_path, 'cam.png')
    cam.save(cam_path)
"""
