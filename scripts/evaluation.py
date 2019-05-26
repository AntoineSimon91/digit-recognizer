
# third party imports
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
