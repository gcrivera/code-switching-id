import itertools
# import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

def get_predictions(scores):
    Y = []
    Y_pred = []

    msa_msa_probs = scores['m'][0]
    msa_egy_probs = scores['m'][1]
    for i in range(len(msa_msa_probs)):
        Y.append(0)
        msa_prob = msa_msa_probs[i]
        egy_prob = msa_egy_probs[i]
        if msa_prob >= egy_prob:
            Y_pred.append(0)
        else:
            Y_pred.append(1)

    egy_msa_probs = scores['f'][0]
    egy_egy_probs = scores['f'][1]
    for i in range(len(egy_msa_probs)):
        Y.append(1)
        msa_prob = egy_msa_probs[i]
        egy_prob = egy_egy_probs[i]
        if msa_prob > egy_prob:
            Y_pred.append(0)
        else:
            Y_pred.append(1)

    return Y,Y_pred


def evaluate(Y, Y_pred):
    # calculate accuracy
    accuracy = metrics.accuracy_score(Y, Y_pred)
    print 'Accuracy:'
    print '\t' + str(accuracy)

    # calculate weighted F1 Scores
    f1_score = metrics.f1_score(Y, Y_pred, labels=[0, 1, 2, 3], average='weighted')
    print 'Weighted F1 Score:'
    print '\t' + str(f1_score)


def confusion_matrix(Y, Y_pred):
    cm = metrics.confusion_matrix(Y, Y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    # classes = [0, 1]
    #
    # plt.figure()
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.show()