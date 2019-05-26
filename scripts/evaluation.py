
# standard imports
import itertools

# third party imports
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_and_accuracy(history, display=True):
    fontsize = 14
    fig = plt.figure(figsize=(12, 9))

    ax = fig.subplots(2, 1)
    ax[0].set_ylabel("Loss", fontsize=fontsize, weight="bold")
    ax[0].set_xlabel("Epochs (#)", fontsize=fontsize, weight="bold")
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss")
    ax[0].legend(loc='best')

    ax[1].set_ylabel("Accuracy", fontsize=fontsize, weight="bold")
    ax[1].set_xlabel("Epochs (#)", fontsize=fontsize, weight="bold")
    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best')

    if display:
        plt.tight_layout()
        plt.show()
    return fig


def plot_confusion_matrix(matrix, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.cm.Blues
    title = 'Confusion matrix'
    classes = range(10)

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return fig
