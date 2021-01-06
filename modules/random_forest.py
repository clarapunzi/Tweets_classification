from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from modules.evaluation import eval_classifier, eval_threshold_roc, eval_threshold_precision_recall
import modules.constants as cst


def rf_classification(x_train, x_test, y_train, y_test, multiple_attributes, dataset, tuning, threshold):

    if tuning:
        rf = RandomForestClassifier()
        rf_opt = rf_finetunig(rf, x_train, y_train)
    else:
        # without hyperparameter optimization (using the previously stored optimal values)
        crit = cst.RF_CRITERION_2[dataset] if multiple_attributes else cst.RF_CRITERION
        max_d = cst.RF_MAX_DEPTH_2[dataset] if multiple_attributes else cst.RF_MAX_DEPTH[dataset]
        max_f = cst.RF_MAX_FEATURES_2[dataset] if multiple_attributes else cst.RF_MAX_FEATURES[dataset]
        n = cst.RF_N_ESTIMATORS_2[dataset] if multiple_attributes else cst.RF_N_ESTIMATORS[dataset]
        rf_opt = RandomForestClassifier(criterion=crit, max_depth=max_d, max_features=max_f, n_estimators=n)
        rf_opt.fit(x_train, y_train)

    y_pred_prob = rf_opt.predict_proba(x_test)

    if threshold:
        if dataset == 'charlieHebdo':
            # Check the Precision-Recall curve to find a good threshold that maximize precision
            thr, precision, recall, ix = eval_threshold_precision_recall(y_test, y_pred_prob[:, 1])
            results = [precision, recall, ix]
        else:
            # Check the ROC curve to find a good threshold
            thr, fpr, tpr, ix = eval_threshold_roc(y_test, y_pred_prob[:, 1], 'RandomForest', multiple_attributes, dataset)
            results = [fpr, tpr, ix]
    else:
        thr = cst.THR_RF_2[dataset] if multiple_attributes else cst.THR_RF[dataset]
        results = []

    positive_probs = [y[1] for y in y_pred_prob]
    y_pred_class = [1 if y > thr else 0 for y in positive_probs]

    # print some statistics
    eval_classifier(y_test, y_pred_class)
    results.insert(0, y_pred_class)

    return results


def rf_finetunig(clf, x_train, y_train):
    """ hyperparameters:
            n_estimators: number of trees to combine
            max_features: number of features to consider when looking for the best split, e.g., "sqrt"=sqrt(n_features), "log2"=log2(n_features)
            max_depth: max depth of the tree
            criterion: function to measure the quality of a split, e.g., “gini” for the Gini impurity and “entropy” for the information gain
    """
    param_grid = { 
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    grid = GridSearchCV(clf, param_grid, refit=True, verbose=1)
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_) 

    return grid
