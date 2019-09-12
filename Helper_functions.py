import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import Constant
from sklearn.metrics import confusion_matrix


def cls():
    """

    :return:
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def countdown(introduction_screen, t=5):
    """
    
    :param introduction_screen: 
    :param t: 
    :return: 
    """
    while t:
        min, secs = divmod(t, 60)
        time_format = '{:02d}:{:02d}'.format(min, secs)
        introduction_screen.set_status_text("Pause! " + time_format)
        time.sleep(1)
        t -= 1


def wait(time_in_sec):
    """

    :param time_in_sec:
    :return:
    """
    dif = 0
    start = time.time()
    while dif <= time_in_sec:
        end = time.time()
        dif = end - start


# def divide_chunks(l, n):
#     tmp = []
#     for i in range(0, len(l), n):
#         tmp.append(l[i:i + n])
#     return tmp


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
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
    return fig


def visualization(history, y_true, y_predict, show_figures=False,
                  labels=Constant.label_display_without_rest, config="", save_path=""):
    """

    :param history:
    :param y_true:
    :param y_predict:
    :param show_figures:
    :param labels:
    :param config:
    :param save_path:
    :return:
    """
    if not history == 0:
        visualization_history(history, save_path=save_path, config=config, show=show_figures)

    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    matrix = plot_confusion_matrix(y_true, y_predict, classes=labels,
                                   title='Confusion matrix, without normalization')
    matrix.savefig(save_path + config + "_confusion_matrix.svg")

    # Plot normalized confusion matrix
    norm_matrix = plot_confusion_matrix(y_true, y_predict, classes=labels, normalize=True,
                                        title='Normalized confusion matrix')
    norm_matrix.savefig(save_path + config + "_norm_confusion_matrix.svg")

    if show_figures:
        plt.show()

    print("Accuracy score", sklearn.metrics.accuracy_score(y_true, y_predict),
          "\nClassification report", sklearn.metrics.classification_report(y_true, y_predict),
          "\nMean absolute error", sklearn.metrics.mean_absolute_error(y_true, y_predict))

    f = open(save_path + config + "Overview" + config + ".txt", 'a', newline='')
    with f:
        for txt in ["Accuracy score " + str(sklearn.metrics.accuracy_score(y_true, y_predict)),
                    "Classification report " + str(sklearn.metrics.classification_report(y_true, y_predict)),
                    "Mean absolute error " + str(sklearn.metrics.mean_absolute_error(y_true, y_predict))]:
            f.writelines(str(txt))
    f.close()
    print("Visualization finish")
    return sklearn.metrics.accuracy_score(y_true, y_predict)


def visualization_history(history, save_path="", config="", show=False):
    """

    :param history:
    :param save_path:
    :param config:
    :param show:
    :return:
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(save_path + config + "_accuracy.svg")
    if show:
        plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(save_path + config + "_loss.svg")
    if show:
        plt.show()


def flat_users_data(dict_data):
    """

    :param dict_data:
    :return:
    """
    x, y = [], []
    for user in dict_data:
        for n in range(len(user['data'])):
            x.append(user['data'][n])
            y.append(user['label'][n])
    return x, y
