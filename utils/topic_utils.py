import collections
import itertools
import re
import string
import timeit

import matplotlib.pyplot as plt
import nltk
#nltk.download('popular')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.api import StringTokenizer
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
from wordcloud import WordCloud

def long_string(list_of_strings):
    """
    Concatenate a list of strings into a single string.
    """
    return ' '.join([string for string in list_of_strings])

def long_list(list_of_lists):
    """
    Concatenate items from multiple lists into a single list.
    """
    return list(itertools.chain(*list_of_lists))

def preprocess_string(string):
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
    string = re.sub(r'[!"$%&()*+,./:;=?[\]^_`{|}~...]+', '', string)
    
    return string

def tokenize_string(input_string, tokenizer):
    """Preprocess and tokenize a raw tweet string.
    
    Args:
        in_string: string to tokenize
        tokenizer: object with a ``tokenize'' method for strings
        
    Return:
        tokens: list of strings (tokens)
    """
    preprocessed = preprocess_string(input_string)
    
    tokens = tokenizer.tokenize(preprocessed)
    
    # remove symbol-only tokens
    tokens = [t for t in tokens if not t in string.punctuation]
    
    return tokens

def remove_stopwords(tokens, stopword_list):
    """Remove unwanted words (``stop words'') from a list of string tokens.
    
    Args:
        tokens: list of tokens to filter
        stopword_list: list of words to remove
        
    Return:
        filtered_tokens: list of tokens with stop words removed
    """    
    cache = set(stopword_list)
    filtered_tokens = [token for token in tokens
                       if token.lower() not in cache]
    
    return filtered_tokens

def lemmatize(tokens, lemmatizer):
    """Lemmatize a set of tokens.
    
    Args:
        tokens: list of tokens to lemmatize
        lemmatizer: object with a ``lemmatize'' method for strings
        
    Return:
        lemmatized: list of lemmatized tokens
    """
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized


