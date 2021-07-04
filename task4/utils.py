import datetime
import functools
import itertools
import os
from pathlib import Path
import re
import sqlite3
from timeit import default_timer

import numpy as np
import pandas as pd

def str_to_list(string):
    """Convert a single list-formatted string into a list of strings."""
    string = re.sub(r"[\[\]\']", "", string)
    return list(string.split(','))

def melt_list_column(df, col_name, id_var):
    """Expand a column of lists into a dataframe with duplicated keys for list
    elements."""
    mask = df[col_name].isin(['[]'])
    col = df[col_name][~mask]
    return (col
            # expand items into columns
            .str.replace(r"[\[\]\']", "", regex=True)
            .str.split(',', expand=True)
            # melt wide table into duplicated tweets
            .reset_index()
            .melt(id_vars=[id_var])
            .drop(columns=['variable'])
            .dropna()
           )

def combine_csvs(directory):
    """Join multiple CSVs into a single dataframe."""

    df = pd.DataFrame()

    p = Path(directory)

    for file in p.iterdir():
        temp = pd.read_csv(file,
                         index_col=0,
                         low_memory=False,)
        temp['file'] = str(file.stem)
        temp['campaign'] = (file.name.split('_')[-5]
                        + file.name.split('_')[-4])
        temp['release'] = file.name.split('_')[-4]
        temp['government'] = file.name.split('_')[-5]
        df = df.append(temp)

    return df


class UsersData(object):
    """Class for generating dataframe from csv dataset of users."""
    def __init__(self, data_path):
        self.df = combine_csvs(data_path)
        self.fields = {'user_display_name':'string',
                        'user_screen_name':'string',
                        'user_reported_location':'string',
                        'user_profile_description':'string',
                        'user_profile_url':'string',
                        'follower_count':'int64',
                        'following_count':'int64',
                        'account_creation_date':'datetime64',
                        'account_language':'string',
                        'campaign':'string',
                        'government':'string',
                        'file':'string'}
        self.df = self.df.astype(self.fields)

    def __str__(self):
        return 'Dataset: {}'.format(name)

    def to_torch(self):
        return #torch.data.DataLoader(self.df)

    def to_tf(self):
        return tf.data.Dataset.from_tensor_slices(dict(self.df))

    def to_np(self):
        return np.to_array(self.df)
        

class TweetsData(object):
    """Class for generating dataframe from csv dataset of tweets."""
    def __init__(self, data_path):
        self.df = combine_csvs(data_path)
        self.fields = {'userid':'string',
                        'user_display_name':'string',
                        'user_screen_name':'string',
                        'user_reported_location':'string',
                        'user_profile_description':'string',
                        'user_profile_url':'string',
                        'account_creation_date':'datetime64',
                        'account_language':'string',
                        'tweet_language':'string',
                        'tweet_text':'string',
                        'tweet_time':'datetime64',
                        'tweet_client_name':'category',
                        'in_reply_to_userid':'string',
                        'retweet_userid':'string',
                        'latitude':'category',
                        'longitude':'category',
                        'campaign':'string',
                        'government':'string',
                        'hashtags':'string',
                        'urls':'string',
                        'user_mentions':'string',
                        'file':'string'}

        self.df = self.df.astype(self.fields)
        self.df['type'] = self.df['is_retweet'].apply(
                            lambda x: 'retweet' if x == True else 'original')
        self.df['has_quote'] = self.df['quoted_tweet_tweetid'].notna()

    def __str__(self):
        return 'Dataset: {}'.format(name)

    def to_torch(self):
        return #torch.data.DataLoader(self.df)

    def to_tf(self):
        return tf.data.Dataset.from_tensor_slices(dict(self.df))

    def to_np(self):
        return np.to_array(self.df)
