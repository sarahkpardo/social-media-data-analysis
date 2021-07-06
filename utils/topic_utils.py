import re
import string

import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud


def preprocess_tweet(string):
    """Remove symbols; replace urls, hashtags, and user 
       mentions with a placeholder token.
    """
    # "rt" ("retweet") 
    string = re.sub('rt', '', string.lower())
    
    # @-mentions
    string = re.sub(r'@\w+', '<-@->', string)
    
    # hyperlinks
    string = re.sub(r'http\S+', '<-URL->', string)
    string = re.sub(r'www.[^ ]+', '<-URL->', string)
    
    # hashtags
    string = re.sub(r'#\w+', '<-#->', string)
    
    # digits
    string = re.sub(r'[0-9]+', '', string)
    
    # symbols
    string = re.sub(r'[!"$%&()*+,./:;=?[\]^_`{|}~]', '', string)
    
    return string

def tokenize_string(string):
    tokens = (TweetTokenizer(preserve_case=False, 
                            reduce_len=True, 
                            strip_handles=False)
              .tokenize(string))
    # remove symbol-only tokens
    tokens = [t for t in tokens if not t in string.punctuation]
    return tokens



def preprocess_tweet(df):
    """Remove symbols; replace urls, hashtags, and user 
       mentions with a placeholder token.
    """
    # "rt" ("retweet") 
    df = df.str.lower().replace('rt', '')
    
    # @-mentions
    df = df.str.replace(r'@\w+', '<-@->')
    
    # hyperlinks
    df = df.str.replace('http\S+', '<-URL->')
    df = df.str.replace('www.[^ ]+', '<-URL->')
    
    # hashtags
    df = df.str.replace(r'#\w+', '<-#->')
    
    # digits
    df = df.str.replace(r'[0-9]+', '')
    
    # symbols
    df = df.str.replace(r'[!"$%&()*+,./:;=?[\]^_`{|}~]', '')
    
    return df

def tweet_tokenize(df):
    """
    Convert `in_string` of text to a list of tokens using NLTK's TweetTokenizer
    """
    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=False)
    df = df.map(tokenizer.tokenize)
    return df

def tokenize(df):
    """Apply nltk tokenizing function to a dataframe,
       removing single-character tokens.
    """
    df = df.map(word_tokenize)
    df = df.apply(lambda x: [x for token in x if len(token)>2])
    return df

def remove_stopwords(df):
    """Remove stop words, based on nltk list."""
    
    cache = set(stopwords.words())
    df = df.apply(lambda x: [word for word in x 
                                    if word.lower() not in cache])
    return df

def lemmatize(df):
    """Lemmatize using nltk WordNet method."""
    lemmatizer = WordNetLemmatizer()
    
    df = df.apply(lambda x: [lemmatizer.lemmatize(word)
                                  for word in x
                                  if len(lemmatizer.lemmatize(w))>3])
    return df

def preprocess_words(df, tweet_tokenizer=True, lemmatize=False):
    """Apply word-level preprocessing."""
    
    if tweet_tokenizer:
        token_fn = tweet_tokenize
    else: token_fn = tokenize
        
    if lemmatize:
        df = df.pipe(token_fn).pipe(remove_stopwords).pipe(lemmatize)
        return df
    
    df = df.pipe(token_fn).pipe(remove_stopwords)
    return df

def preprocess_text(self, tweet_tokenize, lemmatize):
    """Pre-process text in a dataframe column.

    Usage:

        df['processed_text'] = preprocess_data(df['not_processed_text'])

    """
    return df.pipe(preprocess_string).pipe(preprocess_words)


def all_text(list_of_strings):
    """
    Concatenate a list of strings into a single string.
    """
    return ' '.join([string for string in list_of_strings])

def all_tokens(list_of_lists):
    """
    Concatenate items from multiple lists into a single list.
    """
    return list(itertools.chain(*list_of_lists))



def tokenize_iter(df_of_lists):
    """
    Usage: df.pipe(tokenize_df)
    """
    # Concatenate into a long string to be tokenized
    long_string = all_text(all_tokens(df_of_lists))
    preprocessed = preprocess_tweet(long_string)
    
    tokens = (TweetTokenizer(preserve_case=False, 
                            reduce_len=True, 
                            strip_handles=False)
              .tokenize(preprocessed))
    # remove symbol-only tokens
    tokens = [t for t in tokens if not t in string.punctuation]
    return tokens

def tokenize_list(list_of_strings):
    """
    Usage: df.map(tokenize_list)
    """
    long_string = all_text(list_of_strings)
    preprocessed = preprocess_tweet(long_string)
    
    tokens = (TweetTokenizer(preserve_case=False, 
                            reduce_len=True, 
                            strip_handles=False)
              .tokenize(preprocessed))
    # remove symbol-only tokens
    tokens = [t for t in tokens if not t in string.punctuation]
    return tokens