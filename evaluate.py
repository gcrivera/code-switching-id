import numpy as np
import sklearn.metrics as metrics
# Metrics to create
# 1) Accuracy
# 2) Weighted F1
# 4) Confusion Matrix

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