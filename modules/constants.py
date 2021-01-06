"""
OPTIMIZED HYPERPARAMETERS FOR ALL THE CLASSIFIERS
Those denoted by '*_2' refer to the second classification task, i.e. that on multiple attributes of the tweets

"""

# KNN
KNN_LEAFS = 1
KNN_P = 1
KNN_K = {'sydneySiege': 6, 'ottawaShooting': 4, 'charlieHebdo': 7}
KNN_K_2 = {'sydneySiege': 6, 'ottawaShooting': 3, 'charlieHebdo': 10}

# SVM
SVM_C = {'sydneySiege': 10, 'ottawaShooting': 10, 'charlieHebdo': 100}
SVM_C_2 = {'sydneySiege': 10, 'ottawaShooting': 100, 'charlieHebdo': 10}
SVM_CLASS_WEIGHT = 'balanced'
SVM_GAMMA = 'scale'
SVM_KERNEL = 'rbf'
SVM_PROBABILITY = True

# RANDOM FOREST
RF_CRITERION = 'entropy'
RF_MAX_DEPTH = {'sydneySiege': 7, 'ottawaShooting': 8, 'charlieHebdo': 7}
RF_MAX_FEATURES = {'sydneySiege': 'sqrt', 'ottawaShooting': 'log2', 'charlieHebdo': 'sqrt'}
RF_N_ESTIMATORS = {'sydneySiege': 500, 'ottawaShooting': 100, 'charlieHebdo': 200}

RF_CRITERION_2 = {'sydneySiege': 'entropy', 'ottawaShooting': 'gini', 'charlieHebdo': 'entropy'}
RF_MAX_DEPTH_2 = {'sydneySiege': 7, 'ottawaShooting': 6, 'charlieHebdo': 7}
RF_MAX_FEATURES_2 = {'sydneySiege': 'sqrt', 'ottawaShooting': 'auto', 'charlieHebdo': 'sqrt'}
RF_N_ESTIMATORS_2 = {'sydneySiege': 500, 'ottawaShooting': 100, 'charlieHebdo': 100}

# MULTINOMIAL NAIVE BAYES
NB_ALPHA = {'sydneySiege': 1.5, 'ottawaShooting': 0.5, 'charlieHebdo': 0.5}
NB_ALPHA_2 = 0.5

# THRESHOLDS
THR_KNN = {'sydneySiege': 0.67, 'ottawaShooting': 0.75, 'charlieHebdo': 0.5}
THR_SVM = {'sydneySiege': 0.58, 'ottawaShooting': 0.72, 'charlieHebdo': 0.64}
THR_RF = {'sydneySiege': 0.44, 'ottawaShooting': 0.55, 'charlieHebdo': 0.64}
THR_NB = {'sydneySiege': 0.53, 'ottawaShooting': 0.68, 'charlieHebdo': 0.40}

THR_KNN_2 = {'sydneySiege': 0.67, 'ottawaShooting': 0.67, 'charlieHebdo': 0.9}
THR_SVM_2 = {'sydneySiege': 0.46, 'ottawaShooting': 0.65, 'charlieHebdo': 0.74}
THR_RF_2 = {'sydneySiege': 0.45, 'ottawaShooting': 0.62, 'charlieHebdo': 0.63}
THR_NB_2 = {'sydneySiege': 0.52, 'ottawaShooting': 0.70, 'charlieHebdo': 0.37}
