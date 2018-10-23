import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

def evaluate(Y, Y_pred):
    # calculate accuracy
    accuracy = metrics.accuracy_score(Y, Y_pred)
    print 'Accuracy:'
    print '\t' + accuracy

    # calculate weighted F1 Scores
    f1_scores = metrics.f1_score(Y, Y_pred, labels=[0, 1, 2, 3], average='weighted')
    print 'Weighted F1 Scores:'
    print '\tMSA: ' + f1_scores[0]
    print '\tEGY: ' + f1_scores[1]
    print '\tEGY-MSA: ' + f1_scores[2]
    print '\tNon-Arabic: ' + f1_scores[3]


def confusion_matrix(Y, Y_pred):
    cm = metrics.confusion_matrix(Y, Y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = [0, 1, 2, 3]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()