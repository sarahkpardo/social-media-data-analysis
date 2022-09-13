"""Utility functions specific to tweet datasets."""

from collections import Counter
import itertools
from pathlib import Path
import re

import pandas as pd

from sm_analysis.utils import *


class UsersData(object):
    """Class for generating dataframe from csv dataset of users."""
    def __init__(self, data_path):
        self.df = combine_csvs(data_path)
        self.fields = {
            'user_display_name': 'string',
            'user_screen_name': 'string',
            'user_reported_location': 'string',
            'user_profile_description': 'string',
            'user_profile_url': 'string',
            'follower_count': 'int64',
            'following_count': 'int64',
            'account_creation_date': 'datetime64',
            'account_language': 'string',
            'campaign': 'string',
            'government': 'string',
            'file': 'string'
        }
        self.df = self.df.astype(self.fields)

    def __str__(self):
        return 'Dataset: {}'.format(self.name)


class TweetsData(object):
    """Class for generating dataframe from csv dataset of tweets."""
    def __init__(self, data_path):
        self.df = combine_csvs(data_path)
        self.fields = {
            'userid': 'string',
            'user_display_name': 'string',
            'user_screen_name': 'string',
            'user_reported_location': 'string',
            'user_profile_description': 'string',
            'user_profile_url': 'string',
            'account_creation_date': 'datetime64',
            'account_language': 'string',
            'tweet_language': 'string',
            'tweet_text': 'string',
            'tweet_time': 'datetime64',
            'tweet_client_name': 'category',
            'in_reply_to_userid': 'string',
            'in_reply_to_tweetid': 'object',
            'quoted_tweet_tweetid': 'object',
            'retweet_userid': 'string',
            'retweet_tweetid': 'object',
            'latitude': 'category',
            'longitude': 'category',
            'campaign': 'string',
            'government': 'string',
            'hashtags': 'string',
            'urls': 'string',
            'user_mentions': 'string',
            'file': 'string'
        }

        self.df = self.df.astype(self.fields)
        self.df['type'] = self.df['is_retweet'].apply(lambda x: 'retweet'
                                                      if x else 'original')
        self.df['is_reply'] = self.df['in_reply_to_tweetid'].notna()
        self.df['has_quote'] = self.df['quoted_tweet_tweetid'].notna()
        self.df['hashtags'] = self.df['hashtags'].dropna().map(str_to_list)
        self.df['urls'] = self.df['urls'].dropna().map(str_to_list)
        self.df['user_mentions'] = self.df['user_mentions'].dropna().map(
            str_to_list)

    def __str__(self):
        return 'Dataset: {}'.format(self.name)


def get_unique_ids(campaign, campaign_users):
    campaign_userids = (campaign_users.groupby('userid').size().index)
    tweet_userids = (campaign.groupby('userid').size().index)
    retweet_userids = (campaign.loc[:]['retweet_userid'].dropna())
    reply_userids = (campaign.loc[:]['in_reply_to_userid'].dropna())
    mention_userids = (
        campaign['user_mentions']
        # expand usernames into columns
        .explode()
        # melt wide table into duplicated tweets
        .reset_index().melt(id_vars=['tweetid'], value_name='mentioned_userid')
        # clean up
        .astype({
            'mentioned_userid': 'string'
        }).drop(columns=['variable']).join(
            campaign['userid'],
            on='tweetid').dropna()).loc[:]['mentioned_userid']

    return Counter(
        list(
            itertools.chain(*[
                campaign_userids,
                tweet_userids,
                retweet_userids,
                reply_userids,
                mention_userids,
            ])))


def top_user(campaign):
    return campaign.groupby('userid').size().idxmax()


def hashtag_list(campaign):
    hashtags = (campaign.loc[:]['hashtags'].fillna(''))
    return [el.strip() for el in long_list(hashtags) if el != '']


def top_days(campaign, limit=1):
    grouped = (campaign.reset_index().set_index('tweet_time').resample('D'))

    return (grouped.size().sort_values(ascending=False)).index[:limit]


def original_tweets(tweets):
    is_retweet = tweets['is_retweet'] is not False
    return tweets.loc[:][is_retweet]


def retweets(tweets):
    is_retweet = tweets['is_retweet'] is True
    return tweets.loc[:][is_retweet]


def replies(tweets):
    is_reply = tweets['in_reply_to_tweetid'].notna()
    return tweets.loc[:][is_reply]
