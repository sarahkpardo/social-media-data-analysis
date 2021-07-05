import itertools
import re
from string import punctuation
import unicodedata

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def tweet_text_preview(tweets, samples=5):
    freq_terms = (get_frequent_terms(map(lambda x: x.all_text, tweets),
                  ngram_range=(2, 3))
                  .head(50))
    terms = list(freq_terms.reset_index()["token"])

    print("-----------------start summary-----------------------------")
    print("\t----sample tweets ----")
    _tweets = [t for t in it.islice(sorted(tweets,
                                           key=lambda x: x.favorite_count,
                                           reverse=True),
                                    samples)]
    for tweet in _tweets:
        print(f"tweet text:\n \t {tweet.all_text} \n favs: \t {tweet.favorite_count}")
        print()

    print("\t----sample terms ----")
    print(', '.join(terms))
    print("----------------- end summary------------------------------")

    
def plot_timeseries(series,
                    ma=False,
                    raw=False,
                    expanding=False,
                    ewma=False,
                    overall=False,
                    median=False,
                    title=None,
                    time_bin="hour",
                    date_markers=None,
                    y_label=None,
                    custom_yaxis=None,
                    custom_ax=None,
                    **kwargs):
    """
    custom plotting function for our time-series dataframes. 
    Args:
        series: pd.Series or pd.Dataframe
        raw: plot the basic values in the frame
        expanding: plot an expanding mean
        ewma: plot an ewma line
        overall: plot an overall mean
        median: plot the overall median
        title: custom title to use
        time_bin: marks the y-axis correctly
        date_markers: plots a dot on the signal where a given date is noted.
        y_label: custom y-axis label
        custom_yaxis: custom axis
        custom_ax: passing a custom Axes here will assign this plot to that
                   axis
   """
    if isinstance(series, pd.DataFrame):
        series = series["count"]

    lw = 0.75
    if custom_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = custom_ax

    if y_label is None:
        period = series.index.to_period().freqstr
        _bin = "day" if period == "D" else "hour"
        _y_label = "tweets per {}".format(_bin)
        plt.ylabel(_y_label)
    else:
        if isinstance(y_label, str):
            plt.ylabel(y_label)

    if date_markers is not None:
        def dateindex_to_str(index, include_hour=True):
            idx = 16 if include_hour else 10
            return [str(date)[0:idx].replace("T", " ")
                    for date in index.values]

        (ax.plot(date_markers, series.loc[date_markers],
                 "o", markersize=4, color='m', label="point"))

    if raw:
        series.plot(label="raw", lw=lw, ax=ax)

    if ma:
        (series.rolling(ma).mean()
         .plot(ax=ax, label="{}{} ma".format(ma, time_bin), lw=lw))

    if ewma:
        if isinstance(ewma, int):
            (series.ewm(span=ewma).mean()
             .plot(ax=ax, label="emwa - span {}".format(ewma), lw=lw))
        else:
            (series.ewm(alpha=0.05).mean()
             .plot(ax=ax, label="emwa, $\alpha = 0.05$", lw=lw))

    if expanding:
        series.expanding().mean().plot(ax=ax, label="expanding_mean", lw=lw)

    if overall:
        (pd.DataFrame(series)
         .assign(global_mean=lambda x: x['count']
                 .mean())["global_mean"]
         .plot(ax=ax, label="global_mean", lw=lw))

    if median:
        (pd.DataFrame(series)
         .assign(global_median=lambda x: x['count'].median())["global_median"]
         .plot(ax=ax, label="global_median"))

    plt.tight_layout()
    plt.xlabel("datetime")

    if custom_yaxis is not None:
        def log_axis(x, pos):
            'The two args are the value and tick position'
            str_ = '$' + "2^{" + str(x) + "}" + '$'
            return str_
        formatter = FuncFormatter(log_axis)
        ax.yaxis.set_major_formatter(formatter)

    if title:
        ax.set_title(title)
    if custom_ax is not None:
        return
    else:
        return ax


def make_normalplot(df, random=True):
    if random:
        plt.plot(df.index.values,
                 np.random.normal(size=df.shape[0]),
                 lw=0.8,
                 alpha=0.75)
        plt.ylim((-5, 5))
        plt.title("Generated normal time series with $\sigma$ bands")
    else:
        plt.plot(df.index.values, df.values, lw=0.8, alpha=0.75)
        plt.ylim((-5, 8))
        plt.title("Dataframe with bands showing up to 3 sigma")

    plt.axhline(y=1, color="red")
    plt.axhline(y=-1, color="red")
    plt.axhline(y=2, color="orange")
    plt.axhline(y=-2, color="orange")
    plt.axhline(y=3, color="yellow")
    plt.axhline(y=-3, color="yellow")

    arrowprops = dict(arrowstyle="-",
                      color="black",
                      lw=2)

    #textprops = dict(rotation="vertical", fontsize=16)
    textprops = dict()

    plt.annotate("1 $\sigma$",
                 xy=(df.index.values[10], 1),
                 xytext=(df.index.values[10], -1.5),
                 arrowprops=arrowprops,
                 **textprops)

    plt.annotate("2 $\sigma$",
                 xy=(df.index.values[750], 2),
                 xytext=(df.index.values[750], -2.5),
                 arrowprops=arrowprops,
                 **textprops
                 )

    plt.annotate("3 $\sigma$",
                 xy=(df.index.values[1500], 3),
                 xytext=(df.index.values[1500], -3.5),
                 arrowprops=arrowprops,
                 **textprops
                 )
    
def get_frequent_terms(text_series, stop_words=None, ngram_range=None):
    if ngram_range is None:
        ngram_range = (1, 3)

    count_vectorizer = CountVectorizer(analyzer="word",
                                       tokenizer=tokenizer,
                                       stop_words=stop_words,
                                       ngram_range=ngram_range)

    term_freq_matrix = count_vectorizer.fit_transform(text_series)
    terms = count_vectorizer.get_feature_names()
    term_frequencies = term_freq_matrix.sum(axis=0).tolist()[0]

    term_freq_df = (pd.DataFrame(list(zip(terms, term_frequencies)),
                                 columns=["token", "count"])
                    .set_index("token")
                    .sort_values("count", ascending=False))
    return term_freq_df