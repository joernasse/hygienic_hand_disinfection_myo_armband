import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import Constant
from sklearn.metrics import confusion_matrix


def cls():
    """
    Clear Cmmandline window
    :return: No returns
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def wait(time_in_sec):
    """
    Wait a specific time, like busy waiting
    :param time_in_sec: int
            The time to wait
    :return: No returns
    """
    dif = 0
    start = time.time()
    while dif <= time_in_sec:
        end = time.time()
        dif = end - start


def plot_confusion_matrix(y_true, y_predict, classes, norm=False, title=None, cmap=plt.cm.Blues):
    """
    From https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param y_true: list
            The correct labels for a set of data(samples)
    :param y_predict: list
            The predicted list for the same set of data, where the y_true came from
    :param classes: list
            List with classes(categories)
    :param norm: boolean
            If True, confusion matrix will normalizized
            If False, confusion matrix will not normalizised
    :param title: string, default=None
            Describes the titel of the plot
    :param cmap:
    :return:
    """
    if not title:
        if norm:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_predict)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if norm:
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
    fmt = '.2f' if norm else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def result_visualization(y_true, y_predict, show_figures=False, labels=Constant.label_display_without_rest,
                         config="", save_path="./"):
    """
    Visualization of the prediction results
    :param y_true: list
            The correct labels for a set of data(samples)
    :param y_predict: list
            The predicted list for the same set of data, where the y_true came from
    :param show_figures: boolean, default False
            If True diagram is displayed
            If False diagram is not displayed
    :param labels: list, default Constant.label_display_without_rest
            List of labels to display
    :param config: string, default ""
            Current configuration
    :param save_path:string, default "./"
            Specifies the path to save the file
    :return:
    """
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    matrix = plot_confusion_matrix(y_true, y_predict, classes=labels,
                                   title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    norm_matrix = plot_confusion_matrix(y_true, y_predict, classes=labels, norm=True,
                                        title='Normalized confusion matrix')

    norm_matrix.savefig(save_path + config + "_norm_confusion_matrix.svg")
    matrix.savefig(save_path + config + "_confusion_matrix.svg")

    if show_figures:
        plt.show()

    acc_score = sklearn.metrics.accuracy_score(y_true, y_predict)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_true, y_predict),
          "\nClassification report", sklearn.metrics.classification_report(y_true, y_predict),
          "\nMean absolute error", sklearn.metrics.mean_absolute_error(y_true, y_predict))

    f = open(save_path + config + "Results.txt", 'a', newline='')
    with f:
        for txt in ["Accuracy score " + str(acc_score),
                    "Classification report " + str(sklearn.metrics.classification_report(y_true, y_predict)),
                    "Mean absolute error " + str(sklearn.metrics.mean_absolute_error(y_true, y_predict))]:
            f.writelines(str(txt))
    f.close()
    print("Visualization finish")
    return acc_score


def history_visualization(history, save_path="./", config="", show_results=False):
    """
    Visualization of the training history. The diagrams can optional be saved and hide
    :param history:
    :param save_path: sting
            Path for saving the file (folder)
    :param config: string 
            The configuration of data processing
    :param show_results: boolean, default False
            If True, visualization will show
            If False, visualization will not show
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
    if show_results:
        plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(save_path + config + "_loss.svg")
    if show_results:
        plt.show()


def flat_users_data(dict_data):
    """
    Flat the date structure from a given directory to an array
    :param dict_data:dict{'data':list,'label':list}
            Represent the data structure as directory. Contains data and labels
    :return: array,array
            x: The flatted array for data
            y: The flatted array for labels
    """
    x, y = [], []
    for user in dict_data:
        for n in range(len(user['data'])):
            x.append(user['data'][n])
            y.append(user['label'][n])
    return x, y


def normalize_by_rest_gesture(data, sensor, mode='rest_mean'):
    """
    TODO
    :param data:
    :param sensor:
    :param mode:
    :return:
    """
    print("Normalization by Rest gesture - Start")
    if sensor == Constant.IMU:
        element = 10
    else:
        element = 9
    rest_data = data['Rest']
    channel, mean = [], []
    if mode == 'rest_mean':
        for ch in range(1, element):
            for i in range(len(rest_data)):
                channel.append(rest_data[i][ch])
            mean.append(np.mean(channel))  # Mean of base REST over all channels (1-8)

        try:
            for d in data:
                if d == 'Rest':
                    continue
                for ch in range(1, element):  # Channel/ IMU entries
                    for i in range(len(data[d])):  # Elements
                        data[d][i][ch] = data[d][i][ch] / mean[ch - 1]
        except:
            print("Not expected exception in normalization by rest function")
            raise

    if mode == 'max_value_channel':
        for d in data:
            items = []
            for ch in range(1, element):
                for i in range(len(data[d])):
                    items.append(data[d][i][ch])
                max_val = max(items)
                items = items / max_val
    print("Normalization by Rest gesture - Done")
    return data


def countdown(t):
    """
    Calculate the countdown from given value t
    :param t: int
            countdown duration
    :return:
    """
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat)
        time.sleep(1)
        t -= 1
