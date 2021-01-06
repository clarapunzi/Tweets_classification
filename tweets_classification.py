import pandas as pd
import argparse
import re
from sklearn.model_selection import train_test_split
import warnings

# my modules:
from modules.preprocessing import find_near_duplicate, load_tweets, get_balanced_data
from modules.svm import svm_classification
from modules.random_forest import rf_classification
from modules.knn import knn_classification
from modules.multinomialNB import multinomial_nb_classification
from modules.pretraining import bert_contextual_embedding
from modules.evaluation import multiplot_roc, multiplot_precision_recall

# from modules.feature_selection import select_features
warnings.filterwarnings('ignore')


# Software for single dataset
def main(dataset, multiple_attributes, tuning, threshold, get_bert):
    """
    Parameters
    ----------
    dataset : String (charlieHebdo, ottawaShooting or sydneySiege)
        The dataset to from which to classify the tweets (required!)
    multiple_attributes : Boolean
        True if you want to consider multiple tweet attributes for classification purposes, False if you want to consider their text only (default)
    tuning : Boolean
        True if if you want to execute the hyperparameter tuning of the classifiers, False otherwise (default)
    threshold : Boolean
        True if you want to find the optimal threshold for each classifier, False otherwise (default)
    get_bert : Boolean
        True if you want to compute the BERT contextual embeddings, False if you want to load them from memory (default)

    Returns
    -------
    None
    """
    # STEP 1: PREPROCESSING
    print('---------- DATASET : {df} ----------'.format(df=dataset))
    print('---------- STEP 1: DATA PRE-PROCESSING ----------')

    # 1.1 Load the dataset ( format : id | tweet | label) or with more tweet features when multiple_attributes is True
    df_full = load_tweets(dataset, multiple_attributes)
    # 1.2 Removing near duplicate tweets
    df_full = find_near_duplicate(df_full)

    # STEP 2: CLASSIFICATION TASK
    if not multiple_attributes:
        print('---------- STEP 2: CLASSIFICATION (TWEETS TEXT ONLY) ----------')
        df = df_full
    else:
        print('---------- STEP 2: CLASSIFICATION (TWEETS & ADDITIONAL ATTRIBUTES) ----------')
        # for the pre-training step I only consider the text of the tweets, all other additional attributes (already numerical) will be rejoined later
        df = df_full[['tweet', 'label']]

    print('---------- STEP 2.1: BERT CONTEXTUAL EMBEDDING ----------')
    if get_bert:
        train_tweet_text, test_tweet_text, train_labels_text, test_labels_text = train_test_split(df['tweet'],
                                                                                                  df['label'],
                                                                                                  random_state=2018,
                                                                                                  test_size=0.3,
                                                                                                  stratify=df['label'])
        # 2.1 Get the sentence embedding
        print('Getting BERT contextual embedding of the train set')
        train_tweet = bert_contextual_embedding(train_tweet_text)
        # encoded_tweets_train.to_csv(dataset+'_encoded_tweets_train_full.csv')
        print('Getting BERT contextual embedding of the test set')
        test_tweet = bert_contextual_embedding(test_tweet_text)
        # encoded_tweets_test.to_csv(dataset+'_encoded_tweets_test_full.csv')

        train_labels = train_labels_text
        # train_labels_text.to_csv('train_labels.csv')
        test_labels = test_labels_text
        # test_labels_text.to_csv('test_labels.csv')

    else:
        # 2.1 Load BERT sentence embeddings from memory
        print('Getting BERT contextual embedding of the train set')
        train_tweet = pd.read_csv(dataset + '/train.csv').set_index('tweetId')
        # the following line was necessary because I stored all vector values are read by pandas as strings in the format 'tensor(0.1965)'
        # so it is necessary to parse them and extract the correspondent decimal value
        # (this could have been avoided if I had stored all tweets representations in decimal format..)
        train_tweet = train_tweet.applymap(lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0]))

        print('Getting BERT contextual embedding of the test set')
        test_tweet = pd.read_csv(dataset + '/test.csv', index_col='tweetId')
        test_tweet = test_tweet.applymap(lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", x)[0]))

        train_labels = pd.read_csv(dataset + '/train_labels.csv', index_col='tweetId')
        test_labels = pd.read_csv(dataset + '/test_labels.csv', index_col='tweetId')

    if not multiple_attributes:
        print('The classification task will be executed on the text of the tweets only')
    else:
        print('The classification task will be executed on multiple tweets\' attributes')
        # add additional features to the dataframe
        # final df format: id | tweet (768 cols) | favorite_count | source | retweet_count | user_id | user_verified | reliable_source | user_geo_enabled | user_time_zone
        # join train and test dataframes with the new feature dataframes
        train_tweet = train_tweet.join(df_full[['favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified',
                                                'reliable_source', 'user_geo_enabled', 'user_time_zone']], how='left')
        test_tweet = test_tweet.join(df_full[['favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified',
                                              'reliable_source', 'user_geo_enabled', 'user_time_zone']], how='left')
        # select_features(train_tweet,train_labels)

    # Being the dataset 'CharlieHebdo' very unbalanced, I will use the SMOTE algorithm to balance the data by oversampling the minority class
    if dataset == 'charlieHebdo':
        train_tweet, train_labels = get_balanced_data(train_tweet, train_labels)

    print('---------- STEP 2.2: CLASSIFICATION ----------')

    print('\n----- 2.2.1: DUMB classifier -----')
    y_dumb_acc = max(test_labels['label'].mean(), 1 - test_labels['label'].mean())
    print('Accuracy of dumb model, i.e. always predicting the most frequent class: ', y_dumb_acc)

    print('\n----- 2.2.2: KNN classifier -----')
    results_knn = knn_classification(train_tweet, test_tweet, train_labels, test_labels, multiple_attributes, dataset,
                                     tuning, threshold)

    print('\n----- 2.2.3: SVM classifier -----')
    results_svm = svm_classification(train_tweet, train_labels, test_tweet, test_labels, multiple_attributes, dataset,
                                     tuning, threshold)

    print('\n----- 2.2.4: Random Forest classifier -----')
    results_rf = rf_classification(train_tweet, test_tweet, train_labels, test_labels, multiple_attributes, dataset,
                                   tuning, threshold)

    print('\n----- 2.2.5: Multinomial Naive Bayes classifier -----')
    results_nb = multinomial_nb_classification(df_full, multiple_attributes, dataset, tuning, threshold)

    if threshold:
        # save a plot with all ROC curves
        if dataset != 'charlieHebdo':
            multiplot_roc(results_knn[1], results_knn[2], results_knn[3], results_svm[1], results_svm[2],
                          results_svm[3], results_rf[1], results_rf[2], results_rf[3], results_nb[1], results_nb[2],
                          results_nb[3], multiple_attributes)
        else:
            baseline = train_labels['label'].sum() / train_labels['label'].count()
            multiplot_precision_recall(results_knn[1], results_knn[2], results_knn[3], results_svm[1], results_svm[2],
                                       results_svm[3], results_rf[1], results_rf[2], results_rf[3], results_nb[1],
                                       results_nb[2], results_nb[3], multiple_attributes, baseline)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Software for fake news identification in tweets')
    parser.add_argument('--dataset', required=True,
                        help='the dataset to analyze. To choose among: ottawaShooting, charlieHebdo and SydneySiege')
    parser.add_argument('--multiple_attributes', action='store_true', default=False,
                        help='When provided, the classification task will be executed on multiple attributes of the tweets, otherwise only the text of the tweets will be considered (default)')
    parser.add_argument('--tuning', action='store_true', default=False,
                        help='When provided, the hyperparameter tuning of the all classifiers will be executed, otherwise the optimal values previously found will be used (default)')
    parser.add_argument('--threshold', action='store_true', default=False,
                        help='When provided, the optimal threshold for the binarization of the predictions of each classifier will be evaluated, otherwise the optimal values previously found will be used (default)')
    parser.add_argument('--get_bert', action='store_true', default=False,
                        help='When provided, the BERT contextual embedding of the tweets will be computed, otherwise it will be loaded from the physical memory')
    args = parser.parse_args()
    main(dataset=args.dataset, multiple_attributes=args.multiple_attributes, tuning=args.tuning,
         threshold=args.threshold, get_bert=args.get_bert)
