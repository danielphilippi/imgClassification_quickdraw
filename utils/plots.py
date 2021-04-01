from matplotlib import pyplot as plt
import numpy as np


# Todo: move to helper fcts module
def plot_history(history, model_config, title):
    metric = model_config['compiler']['metrics'][0]
    acc = history[f'{metric}']
    val_acc = history[f'val_{metric}']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = [i for i in range(len(acc))]

    # generate instance fig and return fig
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.suptitle(title)

    ax1.plot(epochs, acc, c='dimgrey', marker='o', label='Training acc')
    ax1.plot(epochs, val_acc, c='grey', marker='x', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.set_ylim(0., 1.)
    ax1.legend()

    ax2.plot(epochs, loss, c='dimgrey', marker='o', label='Training loss')
    ax2.plot(epochs, val_loss, c='grey', marker='x', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()

    plt.xlabel('epoch')
    fig.show()
    return fig


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    https://stackoverflow.com/questions/57763363/validation-accuracy-metrics-reported-by-keras-model-fit-log-and-sklearn-metrics
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # fig.show()

    return fig
