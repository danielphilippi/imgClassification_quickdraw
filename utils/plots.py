from matplotlib import pyplot as plt
import numpy as np
from utils.cam import make_gradcam_heatmap, save_and_display_gradcam
from skimage.transform import resize


# Todo: move to helper fcts module
def plot_history(history, model_config, title):
    metric = model_config['compiler']['metrics'][0]
    acc = history[f'{metric}']
    val_acc = history[f'val_{metric}']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = [i for i in range(len(acc))]

    # generate instance fig and return fig
    if 'lr' in history.keys():
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(6.4, 4.8/2*3))
    else:
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

    if 'lr' in history.keys():
        lr = history['lr']
        ax3.plot(epochs, lr, c='grey', marker='o', label='')
        ax3.set_title('Learning rate')

    plt.xlabel('epoch')
    fig.show()
    return fig


def plot_lr(history, title):
    lr = history['lr']
    epochs = [i for i in range(len(lr))]

    fig, ax1 = plt.subplots(nrows=1)
    fig.suptitle(title)

    ax1.plot(epochs, lr, c='grey', marker='o', label='')
    ax1.set_title('')
    #ax1.set_ylim(0., 1.)
    #ax1.legend()
    plt.xlabel('epoch')
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


def plot_cam(model, num_cat, y_true, y_pred_classes, test_set, class_names, title):

    idx_correct = []
    idx_incorrect = []

    for idx in range(y_true.shape[0]):
        if y_true[idx] == y_pred_classes[idx]:
            idx_correct.append(idx)
        else:
            idx_incorrect.append(idx)

    tuple_correct = list(zip(idx_correct, y_true[idx_correct]))  # (idx, cat)
    tuple_incorrect = list(zip(idx_incorrect, y_true[idx_incorrect]))  # (idx, cat)

    idx_per_class_correct = [[] for i in range(num_cat)]
    for idx, cat in tuple_correct:
        idx_per_class_correct[cat].append(idx)

    idx_per_class_incorrect = [[] for i in range(num_cat)]
    for idx, cat in tuple_incorrect:
        idx_per_class_incorrect[cat].append(idx)

    r = 2
    c = num_cat
    #fig, axs = plt.subplots(nrows=c, ncols=r, sharey=True, figsize=(6.4 / 3 * 2, 4.8 * 2))

    cnt = 0
    zoom = 4

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10/2, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    fig.suptitle(title)

    idx_per_class_comb = [idx_per_class_correct, idx_per_class_incorrect]
    outer_titles = ['Correct\nClassification', 'Wrong\nClassification']

    for o in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(c, r,
                                                 subplot_spec=outer[o], wspace=0.1, hspace=0.1)
        idx_per_class = idx_per_class_comb[o]

        # set outer titles
        ax = plt.Subplot(fig, outer[o])
        ax.set_title(outer_titles[o])
        ax.axis('off')
        fig.add_subplot(ax)
        for i in range(c):
            for j in range(r):
                axs = plt.Subplot(fig, inner[i, j])

                # j = j if j < len(idx_per_class_correct[i]) else 0
                try:
                    #if i == 0: axs.set_title('dd')

                    img_idx = idx_per_class[i][j]
                    img = test_set.x[img_idx].copy()

                    heatmap = make_gradcam_heatmap(img.reshape(1, img.shape[0], img.shape[0], 1), model)
                    cam = save_and_display_gradcam(img.reshape(img.shape[0], img.shape[0], 1), heatmap, alpha=.9)

                    cam_array = np.array(cam).copy()
                    cam_array_resized = resize(cam_array, (28 * zoom, 28 * zoom))

                    axs.imshow(cam_array_resized, interpolation='nearest')
                    if (j != 0) | (o == 1):
                        axs.axis('off')
                    axs.xaxis.set_visible(False)  # Hide only x axis
                    axs.set_ylabel(class_names[i], rotation='horizontal', ha='right', va='center')
                    #axs[i, j].set_yticklabels([])
                    axs.tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        left=False,  # ticks along the bottom edge are off
                        right=False,  # ticks along the top edge are off
                        labelleft=False)  # labels along the bottom edge are off

                    cnt += 1
                    fig.add_subplot(axs)

                except IndexError:
                    axs.axis('off')
                    axs.set_ylabel(class_names[i], rotation='horizontal', ha='right', va='center')

    #fig.subplots_adjust(wspace=0.00)
    # fig.show()
    #fig.savefig('test.pdf')
    fig.tight_layout()

    return fig

