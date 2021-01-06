import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.model_selection import GridSearchCV

import modules.constants as cst
from modules.evaluation import eval_classifier, eval_threshold_roc, eval_threshold_precision_recall


def multinomial_nb_classification(df, multiple_attributes, dataset, tuning, threshold):
    # data pre-processing
    df = preprocessing(df)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(df['text_final'], df['label'],
                                                                        test_size=0.3,
                                                                        stratify=df['label'],
                                                                        random_state=2018)
    # word vectorization
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(df['text_final'])
    x_train_tfidf = tfidf_vect.transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    if multiple_attributes:
        x_train_tfidf = pd.DataFrame.sparse.from_spmatrix(x_train_tfidf, index=x_train.index)
        x_train_tfidf = x_train_tfidf.join(df[['favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified', 'reliable_source', 'user_geo_enabled', 'user_time_zone']], how='left')
        x_test_tfidf = pd.DataFrame.sparse.from_spmatrix(x_test_tfidf, index=x_test.index)
        x_test_tfidf = x_test_tfidf.join(df[['favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified', 'reliable_source', 'user_geo_enabled', 'user_time_zone']], how='left')

    # classification with Multinomial Naive Bayes
    if tuning:
        nb_clf = naive_bayes.MultinomialNB()
        nb_opt = nb_finetuning(nb_clf, x_train_tfidf, y_train)
    else:
        # without hyperparameter optimization (using the previously stored optimal values)
        a = cst.NB_ALPHA_2 if multiple_attributes else cst.NB_ALPHA[dataset]
        nb_opt = naive_bayes.MultinomialNB(alpha=a)
        nb_opt.fit(x_train_tfidf, y_train)

    # predict the labels on validation dataset
    y_pred_prob = nb_opt.predict_proba(x_test_tfidf)

    if threshold:
        if dataset == 'charlieHebdo':
            # Check the Precision-Recall curve to find a good threshold that maximize precision
            thr, precision, recall, ix = eval_threshold_precision_recall(y_test, y_pred_prob[:, 1])
            results = [precision, recall, ix]
        else:
            # Check the ROC curve to find a good threshold
            thr, fpr, tpr, ix = eval_threshold_roc(y_test, y_pred_prob[:, 1], 'MultinomialNB', multiple_attributes, dataset)
            results = [fpr, tpr, ix]
    else:
        thr = cst.THR_NB_2[dataset] if multiple_attributes else cst.THR_NB[dataset]
        results = []

    # predict class of each tweet
    positive_probs = [y[1] for y in y_pred_prob]
    y_pred_class = [1 if y > thr else 0 for y in positive_probs]

    # print some statistics
    eval_classifier(y_test, y_pred_class)
    results.insert(0, y_pred_class)

    return results


def preprocessing(df):
    # Step 1: Remove blank rows if any.
    df['tweet'].dropna(inplace=True)
    # Step 2: Change all the text to lower case
    df['tweet'] = [entry.lower() for entry in df['tweet']]
    # Step 3: Tokenization
    df['tweet'] = [word_tokenize(entry) for entry in df['tweet']]
    # Step 4: Remove stop words, non-numeric terms and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    text_final_list = []
    for index, row in df.iterrows():
        final_words = []
        word_lemmatized = WordNetLemmatizer()
        entry = row['tweet']
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Ignore stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        text_final_list.append(str(final_words))
    df['text_final'] = text_final_list
    return df


def nb_finetuning(clf, x, y):
    """ hyperparameters:
        alpha : Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    """
    param_grid = {
        'alpha': np.linspace(0.5, 1.5, 6)
    }
    grid = GridSearchCV(clf, param_grid)
    grid.fit(x, y)

    # print best parameter after tuning
    print(grid.best_params_)

    return grid
