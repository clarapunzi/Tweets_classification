import json
import os
from pathlib import Path
from snapy import MinHash, LSH
import re
import pandas as pd 
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler

SEED = 3

# path to the folder with the original file of the tweets
path = 'C:/Users/clara/Documents/DMT_Project/datasets__Tweets_Classification_project__DMT_2019_2020'
newspaper_list = pd.read_csv('reliable_newspapers.csv', header=None)[0].tolist()
datasets_names = {'charlieHebdo': 'charliehebdo-all-rnr-threads',
                  'ottawaShooting': 'ottawashooting-all-rnr-threads',
                  'sydneySiege': 'sydneysiege-all-rnr-threads'}
tweets_classification = ['rumours', 'non-rumours']
min_jaccard_similarity = 1
shingle_length = 2


def build_df_text_only(dataset):
    """
    Parameters
    ----------
    dataset : nested dictionary
        Nested dictionary with two key: rumours and non-rumours, each having the dict of tweets as value in the form:
        {'rumours': {filename1.json: filename2.text}, filename1.json: filename2.text, ...},
         'non-rumours': {filename1.json: filename2.text}, filename1.json: filename2.text, ...}}

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe of the corresponding dataset, having the tweet identifier as index and two columns:
            one for the text of the tweet, the second for the label (1 = rumours, 0 = non-rumours)
    """
    df_r = pd.DataFrame(dataset['rumours'].values(), columns=['tweet'], index=dataset['rumours'].keys())
    df_r['label'] = 1
    df_nr = pd.DataFrame(dataset['non-rumours'].values(), columns=['tweet'], index=dataset['non-rumours'].keys())
    df_nr['label'] = 0
    df = pd.concat([df_r, df_nr], ignore_index=False)
    df.index.name = 'tweetId'
    df.index = [int(i) for i in df.index]
    return df


def build_df(dataset):
    """
    Parameters
    ----------
    dataset : nested dictionary
        Nested dictionary with two key: rumours and non-rumours, each having the dict of tweets as value in the form:
        {'rumours': {filename1.json: filename2.text}, filename1.json: filename2.text, ...},
         'non-rumours': {filename1.json: filename2.text}, filename1.json: filename2.text, ...}}

    Returns
    -------
    df : pd.DataFrame
        Pandas dataframe of the corresponding dataset, having the tweet identifier as index and multiple columns:
        tweet (text), favorite_count, source,retweet_count, user_id, user_verified, user_description, user_geo_enabled, user_time_zone
        and label (1 = rumours, 0 = non-rumours)
    """
    df_r = pd.DataFrame(dataset['rumours'].values(), columns=['tweet', 'favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified', 'user_description', 'user_geo_enabled', 'user_time_zone'], index=dataset['rumours'].keys())
    df_r['label'] = 1
    df_nr = pd.DataFrame(dataset['non-rumours'].values(), columns=['tweet', 'favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified', 'user_description', 'user_geo_enabled', 'user_time_zone'], index=dataset['non-rumours'].keys())
    df_nr['label'] = 0
    df = pd.concat([df_r, df_nr], ignore_index=False)
    df.index.name = 'tweetId'
    df.index = [int(i) for i in df.index]

    # feature pre-processing
    # fill null values
    df = df.fillna({"user_description": ""})
    df = df.fillna({"user_time_zone": df.user_time_zone.mode()})
    # label encoding
    df["user_time_zone"] = df["user_time_zone"].astype('category')
    df["user_time_zone"] = df["user_time_zone"].cat.codes
    df["source"] = df["source"].astype('category')
    df["source"] = df["source"].cat.codes
    # convert boolean column to int
    df['user_verified'] = df['user_verified'].astype(int)
    df['user_geo_enabled'] = df['user_geo_enabled'].astype(int)
    # instead of keeping the column user_description which would have required more text processing, I decided to replace it with a derived binary column
    # having value 1 if the text of the user's description contains a reference to a reliable source
    # (list from https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources#Sources), 0 otherwise
    df['reliable_source'] = df['user_description'].apply(
        lambda y: 1 if any(desc in y for desc in newspaper_list) else 0)
    df.drop('user_description', axis=1, inplace=True)

    df_num = df[['favorite_count', 'source', 'retweet_count', 'user_id', 'user_verified', 'reliable_source', 'user_geo_enabled', 'user_time_zone']]
    df = df[['tweet', 'label']]
    # normalize numerical features in the range 0-1
    x = df_num.values  # returns a numpy array
    cols = df_num.columns
    ind = [int(i) for i in df_num.index]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_num = pd.DataFrame(x_scaled, columns=cols, index=ind)
    df = df.join(df_num)
    return df


def check_duplicate_id_files(dataset):
    """
    Check whether there are identical file identifiers in the rumours and non-rumours dictionary
    """
    if len(set(dataset['non-rumours'].keys()).intersection(set(dataset['rumours'].keys()))) == 0:
        return True
    else:
        return False


def load_tweets(dataset, multiple_attributes):
    """
    Parameters
    ----------
    dataset : String
        Dataset to analyse, to be chosen among: charlieHebdo, ottawaShooting, sydneySiege
    multiple_attributes: Boolean
        True if multiple attributes needs to be retrieved, False if only the text of tweets is required

    Returns
    -------
    tweet_df : pd.DataFrame
        Pandas dataframe of the corresponding dataset, having the tweet identifier as index and two columns:
            one for the text of the tweet, the second for the label (1 = rumours, 0 = non-rumours)

    """
    print('Loading data from source...')
    tweet_dict = {'rumours': {}, 'non-rumours': {}}

    for tweet_type in tweets_classification:
        for file in Path(os.path.join(path, datasets_names[dataset], tweet_type)).iterdir():
            if not file.name.startswith("."):
                with open(file, 'r') as f:
                    data = json.load(f)
                    tweet = data['text'].strip()
                    # remove all http links to keep only a cleaner text
                    tweet = " ".join(filter(lambda x: x[:4] != 'http', tweet.split()))
                    tweet = tweet.replace('\n', ' ').strip()

                    # when required from the user, the following additional attributes are retrieved and added to the dataset:
                    if multiple_attributes:
                        favorite_count = data['favorite_count']
                        source = data['source']
                        source = source[source.find('>') + 1:source.find('<', source.find(
                            '>'))].strip()  # extract the text outside the tag
                        retweet_count = data['retweet_count']
                        user_id = data['user']['id']
                        user_verified = data['user']['verified']
                        user_description = data['user']['description']
                        user_geo_enabled = data['user']['geo_enabled']
                        user_time_zone = data['user']['time_zone']

                        tweet_dict[tweet_type][re.sub('.json$', '', file.name)] = [tweet, favorite_count, source, retweet_count, user_id, user_verified, user_description, user_geo_enabled, user_time_zone]
                    else:
                        tweet_dict[tweet_type][re.sub('.json$', '', file.name)] = tweet

    # Check for anomalies
    check_id = check_duplicate_id_files(tweet_dict)
    if not check_id:
        print('There are multiple tweets with same identifier')
    if not multiple_attributes:
        tweet_df = build_df_text_only(tweet_dict)
    else:
        tweet_df = build_df(tweet_dict)
    return tweet_df


def remove_near_duplicates(tweets, adjacency_list):
    keys_to_delete = []
    for tweet_id, duplicates in adjacency_list.items():
        if tweet_id not in keys_to_delete and len(duplicates) > 0:
            # get the category of the first tweet
            label_cat = tweets.at[tweet_id, 'label']

            # check the category of each near duplicate
            for dup in duplicates:
                dup_cat = tweets.at[dup, 'label']
                
                if label_cat == dup_cat:
                    # if both duplicates are in the same category, delete only the duplicate and keep the label
                    keys_to_delete.append(dup)
                else:
                    # if the duplicates belong to different categories, delete both
                    keys_to_delete.extend([tweet_id, dup])
    # remove duplicate pairs
    print("Number of near-duplicate key detected: ", len(keys_to_delete))
    tweets = tweets.drop(keys_to_delete)
    return tweets


def find_near_duplicate(tweets):
    """Use a LSH object to find the near duplicate strings.
 
    Args:
        tweets (df): tweetId & text
    """
    print('Looking for near duplicate tweets')
    content = tweets['tweet'].to_list()
    labels = tweets.index.to_list()
    # I used the following only to check the regularity of the input data and remove eventual rows with no text
    for index, row in tweets.iterrows():
        if len(row['tweet']) < shingle_length:
            print(index, row)
    for t in content:
        if len(t.split()) < shingle_length:
            print(t)

    # Create MinHash object.
    minhash = MinHash(content, n_gram=shingle_length, n_gram_type='term', seed=SEED)
 
    # Create LSH model.
    lsh = LSH(minhash, labels)
 
    adjacency_list = lsh.adjacency_list(min_jaccard=1)
    tweets = remove_near_duplicates(tweets, adjacency_list)
    print('Final dataset size: ', tweets.shape[0])
    return tweets


def get_balanced_data(x, y):
    over = SMOTE(sampling_strategy=1)
    # under = RandomUnderSampler(sampling_strategy=0.5)
    x, y = over.fit_resample(x, y)
    return x, y
