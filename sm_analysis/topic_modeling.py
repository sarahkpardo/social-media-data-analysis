"""Utility functions for topic modeling."""

import collections
import csv
import functools
import itertools
import re
import urllib

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from wordcloud import WordCloud

from sm_analysis.utils import *

def get_model(model_name):
    """Retrieve a pre-trained model and tokenizer from the transformers
    library."""
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['[HTAG]', '[URL]', '[AT]'])

    return model, tokenizer


def get_labels(csv_link):
    """Retrieve the class labels for a given transformers library model."""
    with urllib.request.urlopen(csv_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    return [row[1] for row in csvreader if len(row) > 1]


def predict_sentiment(document, labels=None, model=None, tokenizer=None):
    """Apply sentiment prediction model to a raw document."""
    # preprocess
    document = preprocess_string(document)

    # tokenize
    encoded_input = tokenizer(document, return_tensors='tf')

    # apply model
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)

    return labels[np.argmax(scores)]


# plotting for LDA
def plot_top_words(model, feature_names, n_top_words, n_components, title):
    """Plotting utility for the results of a fitted sklearn LDA model."""
    fig, axes = plt.subplots(2,
                             n_components // 2,
                             figsize=(30, 15),
                             sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_[:n_components]):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}', fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def extract_topics(
        documents,
        vectorizer=None,
        n_samples=2000,
        n_features=1000,
        n_components=10,
        n_top_words=20,
        apply_preprocessing=True,
        stop_words=None,
):
    """Utility function for applying LDA topic modeling to a list of raw
    documents."""

    if stop_words is None:
        stop_words = [
            *stopwords.words(),
            '[url]',
            '[at]',
            '[htag]',
        ]
    if vectorizer is None:
        tokenize_partial = functools.partial(tokenize_string,
                                             stop_words=stop_words)
        preprocess_partial = functools.partial(preprocess_string,
                                               special_tokens=False)
        vectorizer = CountVectorizer(
            analyzer='word',
            strip_accents='ascii',
            stop_words=stop_words,
            ngram_range=(1, 2),
            preprocessor=preprocess_partial,
            tokenizer=tokenize_partial,
        )
    if apply_preprocessing:
        documents = (documents.map(long_string).map(preprocess_string))

    print('vectorizing...')
    tf = vectorizer.fit_transform(documents)

    print('LDA:\nn_samples: {}\nn_features: {}\nn_components: {}'.format(
        n_samples, n_features, n_components))

    lda = LatentDirichletAllocation(n_components=n_components,
                                    max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)

    tf_feature_names = vectorizer.get_feature_names()

    plot_top_words(lda, tf_feature_names, n_top_words, n_components,
                   'Categories in LDA model')
    plt.tight_layout()


def long_string(list_of_strings):
    """Concatenate a list of strings into a single string."""
    return ' '.join([string for string in list_of_strings])


def long_list(list_of_lists):
    """Concatenate items from multiple lists into a single list."""
    return list(itertools.chain(*list_of_lists))


def preprocess_string(string, special_tokens=False, sep=False):
    """Remove symbols; optionally replace urls, hashtags, and user
       mentions with a special token.
    """
    string = re.sub(r'[\[\]]+', '', string)

    # "rt" ("retweet")
    string = re.sub('rt', '', string.lower())

    # replace & with and
    string = re.sub('&amp;', 'and', string)

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


def tokenize_string(
        input_string,
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
    if preprocess:
        input_string = preprocess_string(input_string)

    tokens = tokenizer.tokenize(input_string)

    # remove stop words
    if stop_words is None:
        stop_words = [
            *stopwords.words(), '[url]', '[at]', '[htag]', '[sep]', '[unk]',
            '[cls]'
        ]

    cache = set(stop_words)
    no_stop_words = [
        token for token in tokens
        if token.lower() not in cache and len(token) > 1
    ]

    return no_stop_words


def remove_stopwords(tokens, stop_words=None):
    if stop_words is None:
        stop_words = [
            *stopwords.words(), '[url]', '[at]', '[htag]', '[sep]', '[unk]',
            '[cls]'
        ]

    cache = set(stop_words)
    no_stop_words = [
        token for token in tokens
        if token.lower() not in cache and len(token) > 1 and token[0] != '#'
    ]

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
                stop_words=[*stopwords.words(), '[URL]', '[AT]', '[HTAG]'],
                timer=False):
    """Apply preprocessing and tokenization to a list of strings.
    Usage:
            output = make_tokens(series_of_strings)

            output = df.apply(make_tokens)

    Return:
        A list of lists of tokens for each string.
    """
    if timer:
        t1 = default_timer()

    processed_strings = [
        preprocess_string(string) for string in list_of_strings
    ]

    tokenized_strings = [
        tokenize_string(string, stop_words) for string in processed_strings
    ]

    if timer:
        t2 = default_timer()
        print('elapsed', t2 - t1)

    return tokenized_strings


def word_frequency(list_of_words):
    """
    Return:
        dict of {'word':'frequency'} sorted by frequency (high to low)
    """
    counts = collections.Counter(list_of_words)

    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def visualize(data, from_frequencies=True, limit=100, color=(150, 50, 50)):
    """Generate word cloud visualization from a set of tokens.
    """
    cloud = WordCloud(background_color="white",
                      prefer_horizontal=0.9,
                      max_font_size=40,
                      relative_scaling=.5,
                      color_func=lambda *args, **kwargs: color)

    data = dict(
        sorted(data.items(), key=lambda item: item[1], reverse=True)[:limit])

    if from_frequencies:
        cloud.generate_from_frequencies(data)
    else:
        cloud.generate(data)

    fig, ax = plt.subplots()
    ax.imshow(cloud)
    ax.axis('off')

    return fig


def vis_from_series(series):
    tokens = make_tokens((series).to_list())
    freq = word_frequency(long_list(tokens))
    fig = visualize(freq)
    return fig
