from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from modules.evaluation import eval_classifier, eval_threshold_roc, eval_threshold_precision_recall
import modules.constants as cst


def knn_classification(x_train, x_test, y_train, y_test, multiple_attributes, dataset, tuning, threshold):

    if tuning:
        knn = KNeighborsClassifier()
        knn_opt = knn_finetuning(knn, x_train, y_train)
    else:
        # without hyperparameter optimization  (using the previously stored optimal values)
        n = cst.KNN_K_2[dataset] if multiple_attributes else cst.KNN_K[dataset]
        knn_opt = KNeighborsClassifier(leaf_size=cst.KNN_LEAFS, p=cst.KNN_P, n_neighbors=n)
        knn_opt.fit(x_train, y_train)

    # Predict test data set.
    y_pred_prob = knn_opt.predict_proba(x_test)

    if threshold:
        if dataset == 'charlieHebdo':
            # Check the Precision-Recall curve to find a good threshold that maximize precision
            thr, precision, recall, ix = eval_threshold_precision_recall(y_test, y_pred_prob[:, 1])
            results = [precision, recall, ix]
        else:
            # Check the ROC curve to find a good threshold
            thr, fpr, tpr, ix = eval_threshold_roc(y_test, y_pred_prob[:, 1], 'KNN', multiple_attributes, dataset)
            results = [fpr, tpr, ix]
    else:
        thr = cst.THR_KNN_2[dataset] if multiple_attributes else cst.THR_KNN[dataset]
        results = []

    # predict rumours if the predicted probability is greater than the threshold calculated above
    positive_probs = [y[1] for y in y_pred_prob]
    y_pred_class = [1 if y > thr else 0 for y in positive_probs]

    # Checking performance our model with classification report.
    eval_classifier(y_test, y_pred_class)
    results.insert(0, y_pred_class)

    return results


def knn_finetuning(knn, x, y):
    """ hyperparameters:
            leaf_size : leaf size passed to the tree used to compute the nearest neighbors
            n_neighbors (k) : number of neighbors to use for query around each point
            p : Power parameter for the Minkowski metric, e.g. p = 1 is the Manhattan distance l1
    """
    leaf_size = list(range(1, 10))
    n_neighbors = list(range(1, 20))
    p = [1, 2]
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    # Use GridSearch for hyperparameter optimization
    clf = GridSearchCV(knn, hyperparameters, cv=10)
    # Fit the model
    best_model = clf.fit(x, y)

    # print best parameter after tuning
    print(best_model.best_params_)

    return best_model
