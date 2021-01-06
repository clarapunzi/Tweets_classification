from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def eval_classifier(y_test, y_pred):

    # Confusion matrix
    print('Confusion matrix: \n', confusion_matrix(y_test, y_pred))

    # Checking performance our model with classification report.
    report = classification_report(y_test, y_pred, digits=4, target_names=['non-rumours', 'rumours'], output_dict=True)
    accuracy = report.pop('accuracy')
    print('Classification report:\n', pd.DataFrame.from_dict(report, orient='index'))

    # Model Accuracy: how often is the classifier correct?
    print("Main measures:\nAccuracy:", accuracy)
    
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", report['rumours']['precision'])

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", report['rumours']['recall'])

    # Model specificity
    print('Specificity: ', report['non-rumours']['recall'])


def eval_threshold_roc(y_test, y_pred_prob, clf, multiple_attributes, dataset):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    # get the best threshold
    J = tpr - fpr   # Youden’s J statistic
    ix = np.argmax(J)
    best_thresh = thresholds[ix] if thresholds[ix] != 1 else thresholds[ix + 1]  # to avoid 1 as threshold (ix +1 because the vector of thresholds is decreasing)
    print('Best threshold for the current model: ', best_thresh)

    # uncomment the following line to get a plot for each model

    # plt.plot(fpr, tpr, marker='.', label=clf)
    # plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.rcParams['font.size'] = 12
    # plt.title('ROC curve for {clf} classifier'.format(clf=clf))
    # plt.xlabel('False Positive Rate (1 - Specificity)')
    # plt.ylabel('True Positive Rate (Sensitivity)')
    # plt.legend()
    # plt.grid(True)
    # filename = 'plots/'+dataset+'ROC_'+clf+'.png' if multiple_attributes else 'plots/'+dataset+'ROC_'+clf+'_text_only.png'
    # plt.savefig(filename)
    # plt.show()

    # Checking performance our model with ROC Score.
    print('AUC score: ', metrics.roc_auc_score(y_test, y_pred_prob))

    return best_thresh, fpr, tpr, ix


def multiplot_roc(fpr_knn, tpr_knn, ix_knn, fpr_svm, tpr_svm, ix_svm, fpr_rf, tpr_rf, ix_rf, fpr_nb, tpr_nb, ix_nb, multiple_attributes):
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline', color='limegreen')
    plt.plot(fpr_knn, tpr_knn, label='k-NN', zorder=1, color='dimgrey')
    plt.plot(fpr_svm, tpr_svm, label='SVM', zorder=1, color='royalblue')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest', zorder=1, color='orange')
    plt.plot(fpr_nb, tpr_nb, label='Multinomial NB', zorder=1, color='mediumvioletred')
    plt.scatter(fpr_knn[ix_knn], tpr_knn[ix_knn], marker='o', color='dimgrey', zorder=2)
    plt.scatter(fpr_svm[ix_svm], tpr_svm[ix_svm], marker='o', color='royalblue', zorder=2)
    plt.scatter(fpr_rf[ix_rf], tpr_rf[ix_rf], marker='o', color='orange', zorder=2)
    plt.scatter(fpr_nb[ix_nb], tpr_nb[ix_nb], marker='o', color='mediumvioletred', zorder=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.grid(True)
    filename = 'plots/ROC_multiple_attributes.png' if multiple_attributes else 'plots/ROC.png'
    plt.savefig(filename)
    return


def eval_threshold_precision_recall(y_test, y_pred_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)

    # since I want to obtain a good precision, I put the lower bound of 0.7 on the value of this measure
    indexes, high_precision = [], []
    for i, j in enumerate(precision):
        if j >= 0.7:
            high_precision.append(j)
            indexes.append(i)
    precision_tail = np.array(high_precision)
    recall_tail = np.array([recall[i] for i in indexes])

    # convert to f score to get the best threshold
    fscore = (2 * precision_tail * recall_tail) / (precision_tail + recall_tail)
    # locate the index of the largest f score
    ix = indexes[np.argmax(fscore)]
    if ix < len(thresholds):
        best_thresh = thresholds[ix]
    elif thresholds[ix - 1] != 1:
        best_thresh = thresholds[ix-1]  # (ix +1 because the vector of thresholds is increasing)
    else:
        best_thresh = thresholds[ix-2]
    print('Best threshold for the current model: ', best_thresh)

    # Checking performance our model with AUC PR score
    # This function actually computes the average precision (AP) from the predictions probabilities
    print('AUC-PR score: ', metrics.average_precision_score(y_test, y_pred_prob))

    return best_thresh, precision, recall, ix


def multiplot_precision_recall(p_knn, r_knn, ix_knn, p_svm, r_svm, ix_svm, p_rf, r_rf, ix_rf, p_nb, r_nb, ix_nb, multiple_attributes, baseline):
    plt.hlines(baseline, xmin=0, xmax=1, linestyle='--', label='Baseline', color='limegreen')
    plt.plot(r_knn, p_knn, label='k-NN', zorder=1, color='dimgrey')
    plt.plot(r_svm, p_svm, label='SVM', zorder=1, color='royalblue')
    plt.plot(r_rf, p_rf, label='Random Forest', zorder=1, color='orange')
    plt.plot(r_nb, p_nb, label='Multinomial NB', zorder=1, color='mediumvioletred')
    plt.scatter(r_knn[ix_knn], p_knn[ix_knn], marker='o', color='dimgrey', zorder=2)
    plt.scatter(r_svm[ix_svm], p_svm[ix_svm], marker='o', color='royalblue', zorder=2)
    plt.scatter(r_rf[ix_rf], p_rf[ix_rf], marker='o', color='orange', zorder=2)
    plt.scatter(r_nb[ix_nb], p_nb[ix_nb], marker='o', color='mediumvioletred', zorder=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    filename = 'plots/Precision_Recall_multiple_attributes.png' if multiple_attributes else 'plots/Precision_Recall.png'
    plt.savefig(filename)
    return
