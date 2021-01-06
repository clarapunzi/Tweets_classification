from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score

from modules.evaluation import eval_classifier, eval_threshold_roc, eval_threshold_precision_recall
import modules.constants as cst


def svm_classification(x_train, y_train, x_test, y_test, multiple_attributes, dataset, tuning, threshold):

    if tuning:
        svm_clf = svm.SVC()
        svm_opt = svm_finetunig(svm_clf, x_train, y_train)
    else:
        # without hyperparameter optimization (using the previously stored optimal values)
        c = cst.SVM_C_2[dataset] if multiple_attributes else cst.SVM_C[dataset]
        svm_opt = svm.SVC(C=c, class_weight=cst.SVM_CLASS_WEIGHT, gamma=cst.SVM_GAMMA, kernel=cst.SVM_KERNEL, probability=cst.SVM_PROBABILITY)
        svm_opt.fit(x_train, y_train)

    # predict probabilities of each tweet of being in each class
    y_pred_prob = svm_opt.predict_proba(x_test)

    if threshold:
        if dataset == 'charlieHebdo':
            # Check the Precision-Recall curve to find a good threshold that maximize precision
            thr, precision, recall, ix = eval_threshold_precision_recall(y_test, y_pred_prob[:, 1])
            results = [precision, recall, ix]
        else:
            # Check the ROC curve to find a good threshold
            thr, fpr, tpr, ix = eval_threshold_roc(y_test, y_pred_prob[:, 1], 'SVM', multiple_attributes, dataset)
            results = [fpr, tpr, ix]
    else:
        thr = cst.THR_SVM_2[dataset] if multiple_attributes else cst.THR_SVM[dataset]
        results = []

    # predict rumours if the predicted probability is greater than the threshold calculated above
    positive_probs = [y[1] for y in y_pred_prob]
    y_pred_class = [1 if y > thr else 0 for y in positive_probs]

    # print some classification measures
    eval_classifier(y_test, y_pred_class)
    results.insert(0, y_pred_class)

    return results


def svm_finetunig(clf, x, y):
    """ hyperparameters:
            C : regularization parameter, positive float (trades off correct classification of training examples against maximization of the decision function’s margin)
            kernel : kernel to  use in the algorithm, e.g. linear for separable dataset, rbf if not
            gamma : kernel coefficient (how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close')
            class_wight : consider class weight for imbalanced problems
    """
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', 'None'], 'probability': [True]},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'kernel': ['rbf'], 'class_weight': ['balanced', 'None'], 'probability': [True]},  # gamma = 0.001 approximates gamma = 'auto' which is equal to 1/768
        ]
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    grid = GridSearchCV(clf, param_grid, scoring=scorers, refit='precision_score', return_train_score=True)
    grid.fit(x, y)

    # print best parameter after tuning
    print(grid.best_params_) 

    return grid
