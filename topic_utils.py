import collections
import itertools
import re
import string
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.api import StringTokenizer
from nltk.tokenize import TweetTokenizer

import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import cosine
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel, TFAutoModel

from wordcloud import WordCloud


string_agg = lambda x: list(x)

def get_model(model, task):
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_tokens(['[HTAG]', '[URL]', '[AT]'])
    
    return model, tokenizer
    
def get_labels(task):
    mapping_link = ('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{}/mapping.txt'.format(task))
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    return [row[1] for row in csvreader if len(row) > 1]
    
def predict_sentiment(document):
    # tokenize
    encoded_input = tokenizer(document, return_tensors='tf')
    
    # apply model
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    return labels[np.argmax(scores)]#, np.max(scores)

def _sentiment(document):
    """Usage: df.apply(get_sentiment_apply)"""
    
    task = 'sentiment'
    MODEL = 'cardiffnlp/twitter-roberta-base-{}'.format(task)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.add_tokens(['[HTAG]', '[URL]', '[AT]'])
    
    mapping_link = ('https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{}/mapping.txt'.format(task))
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    
    # tokenize
    encoded_input = tokenizer(document, return_tensors='tf')
    
    # apply model
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    return labels[np.argmax(scores)]#, np.max(scores)


def plot_top_words(model, 
                   feature_names, 
                   n_top_words, 
                   n_components,
                   title):
    fig, axes = plt.subplots(2, n_components//2, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_[:n_components]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def extract_topics(documents,
                   vectorizer=None,
                   n_samples=2000,
                   n_features=1000,
                   n_components=10,
                   n_top_words=20,
                   apply_preprocessing=True,
                   stop_words=None,
                   ):
    
    if stop_words == None:
        stop_words = [*stopwords.words(),
                      '[url]', '[at]', '[htag]',
                      '[sep]', '[unk]', '[cls]']
    if vectorizer == None:
        vectorizer = CountVectorizer(analyzer='word',
                             strip_accents='ascii',
                             stop_words=stop_words,
                             ngram_range=(1,2),
                             preprocessor=preprocess_partial,
                             tokenizer=tokenize_partial,
                            )
    if apply_preprocessing:
        documents_list = (documents
                          .map(long_string)
                          .map(preprocess_string)
                         )

    print('vectorizing...')
    tf = vectorizer.fit_transform(documents_list)
        
    
    print(('LDA:\nn_samples: {}\nn_features: {}\nn_components: {}')
          .format(n_samples, n_features, n_components))

    lda = LatentDirichletAllocation(n_components=n_components, 
                                    max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)
    
    tf_feature_names = vectorizer.get_feature_names()

    plot_top_words(lda, 
                   tf_feature_names, 
                   n_top_words,
                   n_components,
                   'Categories in LDA model')
    plt.tight_layout()


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

def preprocess_string(string, 
                      special_tokens=False,
                      sep=False):
    """Remove symbols; optionally replace urls, hashtags, and user 
       mentions with a special token.
    """
    string = re.sub(r'[\[\]]+', '', string)
    
    # "rt" ("retweet") 
    string = re.sub('rt', '', string.lower())
    
    if special_tokens:
        # @-mentions
        string = re.sub(r'@\w+', '[AT]', string)

        # hyperlinks
        string = re.sub(r'http\S+', '[URL]', string)
        string = re.sub(r'www.[^ ]+', '[URL]', string)
    
        # hashtags
        string = re.sub(r'#\w+', '[HTAG]', string)
        
    else:
        # @-mentions
        string = re.sub(r'@\w+', '', string)

        # hyperlinks
        string = re.sub(r'http\S+', '', string)
        string = re.sub(r'www.[^ ]+', '', string)
    
        # hashtags
        string = re.sub(r'#\w+', '', string)
        
        # symbols
        string = re.sub(r'[!"$%&()*+,./:;=?^_`{|}~]+', '', string)
    
    if sep:
        # symbols        
        string = re.sub(r'[!]{2,}', '!', string)
        string = re.sub(r'[,]{2,}', ',', string)
        string = re.sub(r'[.]{2,}', '.', string)
        string = re.sub(r'[?]{2,}', '?', string)
        
        string = re.sub(r'[!"(),./:;?_{|}]+', ' [SEP] ', string)
        string = re.sub(r'[!"$%&()*+,./:;=?^_`{|}~]+', '', string)
        
    # digits
    string = re.sub(r'[0-9]+', '', string)

    return string

def tokenize_string(input_string, 
                    stop_words=None,
                    tokenizer=TweetTokenizer(preserve_case=False,
                                             reduce_len=True,
                                             strip_handles=False),
                    preprocess=False,
                   ):
    """Preprocess and tokenize a raw tweet string.
    
    Args:
        in_string: string to tokenize
        tokenizer: object with a ``tokenize'' method for strings
        
    Return:
        tokens: list of strings (tokens)
    """
    if preprocess == True:
        input_string = preprocess_string(input_string)
    
    tokens = tokenizer.tokenize(input_string)
    
    # remove stop words
    if stop_words == None:
        stop_words = [*stopwords.words(),
                      '[url]', '[at]', '[htag]',
                      '[sep]', '[unk]', '[cls]']
        
    cache = set(stop_words)
    no_stop_words = [token for token in tokens
                     if token.lower() not in cache
                     and len(token) > 1
                    ]
    
    return no_stop_words

def remove_stopwords(tokens, stop_words=None):

    if stop_words == None:
        stop_words = [*stopwords.words(),
                      '[url]', '[at]', '[htag]',
                      '[sep]', '[unk]', '[cls]']
        
    cache = set(stop_words)
    no_stop_words = [token for token in tokens
                     if token.lower() not in cache
                     and len(token) > 1
                     and token[0] != '#']
    
    return no_stop_words

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

def make_tokens(list_of_strings,
                stop_words=[*stopwords.words(),
                          '[URL]', '[AT]', '[HTAG]']):
    """Apply preprocessing and tokenization to a list of strings.
    Usage: 
            output = make_tokens(series_of_strings)

            output = df.apply(make_tokens)
            
    Return:
        A list of lists of tokens for each string.
    """    
    
    #t1 = default_timer()

    processed_strings = [preprocess_string(string)
                         for string in list_of_strings]

    tokenized_strings = [tokenize_string(string,
                                         stop_words)
                          for string in processed_strings]
                         
    #t2 = default_timer()
    #print('elapsed', t2 - t1)
    
    return tokenized_strings

def word_frequency(list_of_words):
    """
    Return:
        dict of {'word':'frequency'} sorted by frequency (high to low)
    """
    counts = collections.Counter(list_of_words)
    
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

def visualize(data,
              from_frequencies=True,
              limit=100,
              color=(150,50,50)):
    """
    """
    cloud = WordCloud(background_color="white",
                  prefer_horizontal=0.9,
                  max_font_size=40,
                  relative_scaling=.5,
                  color_func=lambda *args,**kwargs:color)
    
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True)[:limit])
        
    if from_frequencies:
        cloud.generate_from_frequencies(data)
    else:
        cloud.generate(data)
    
    fig, ax = plt.subplots()
    ax.imshow(cloud)
    ax.axis('off')
    
    return fig