import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import Constant
from sklearn.metrics import confusion_matrix


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def countdown(introduction_screen, t=5, ):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        introduction_screen.set_status_text("Pause! " + timeformat)
        time.sleep(1)
        t -= 1


# UNUSED!
# def list_list_to_matrix(list_list):
#     array_list = []
#     for i in range(len(list_list)):
#         tmp = numpy.asarray(list_list[i])
#         array_list.append(numpy.asarray(list_list[i]))
#     m = numpy.asmatrix(array_list)
#     return numpy.asmatrix(numpy.asarray(item) for item in list_list[:-1])
#
#     matrix1 = numpy.asmatrix(tmp)
#     tmp2 = numpy.array(tmp1)
#     tmp3 = tmp2.shape(50, 8)
#     for item in list_list[:-1]:
#         tmp.extend(item)
#
#     print("")


def wait(time_in_sec):
    dif = 0
    start = time.time()
    while dif <= time_in_sec:
        end = time.time()
        dif = end - start


def divide_chunks(l, n):
    tmp = []
    for i in range(0, len(l), n):
        tmp.append(l[i:i + n])
    return tmp


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
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
    return ax


def visualization(history, y_test, y_predict, skip_confusion=False):
    if not history == 0:
        visualization_history(history)

    np.set_printoptions(precision=2)

    if not skip_confusion:
        # Plot non-normalized confusion matrix
        matrix = plot_confusion_matrix(y_test, y_predict, classes=Constant.save_label,
                                       title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        norm_matrix = plot_confusion_matrix(y_test, y_predict, classes=Constant.save_label, normalize=True,
                                            title='Normalized confusion matrix')
        plt.show()

    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_predict),
          "\nClassification report", sklearn.metrics.classification_report(y_test, y_predict),
          "\nMean absolute error", sklearn.metrics.mean_absolute_error(y_test, y_predict))


    print("Visualization finish")


def visualization_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
