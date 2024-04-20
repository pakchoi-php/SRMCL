
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def confusion_matrix_excel(y_true, y_pred,num_classes):
    #5类/3类
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    data = pd.DataFrame(cm)
    writer = pd.ExcelWriter('confusion_matrix.xlsx')
    data.to_excel(writer, 'cm', float_format='%.5f')
    writer.save()
    writer.close()
    return cm


def plot_Matrix(dataset,cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='15')

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    print(cm.shape[1])
    print(cm.shape[0])
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), title=title, ylabel='True label', xlabel='Predicted label')

    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
#Negative Positive Surprise
    if classes==3:
        ax.set_xticklabels(['Negative', 'Positive', 'Surprise'], fontsize='small')
        ax.set_yticklabels(['Negative', 'Positive', 'Surprise'], fontsize='small')
    else:
        if dataset=="casme":
            ax.set_xticklabels(['happiness', 'disgust', 'repression', 'surprise', 'others'], fontsize='small')
            ax.set_yticklabels(['happiness', 'disgust', 'repression', 'surprise', 'others'], fontsize='small')
        else:
            ax.set_xticklabels(['happiness', 'anger', 'contempt', 'surprise', 'others'], fontsize='small')
            ax.set_yticklabels(['happiness', 'anger', 'contempt', 'surprise', 'others'], fontsize='small')



    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'g'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(round(cm[i, j], 2), fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg')
    plt.show()
