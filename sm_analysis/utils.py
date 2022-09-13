"""General utility functions"""

from collections import Counter
import itertools
from pathlib import Path
import re

import pandas as pd

# Python string handling

def long_string(list_of_strings):
    """Concatenate a list of strings into a single string."""
    return ' '.join([string for string in list_of_strings])

def long_list(list_of_lists):
    """Concatenate items from multiple lists into a single list."""
    return list(itertools.chain(*list_of_lists))

def string_agg(x):
    return list(x)

def sort_dict(dictionary):
    return sorted(dictionary.items(), key=lambda item: item[1], reverse=True)

def str_to_list(string):
    """Convert a single list-formatted string into a list of strings."""
    string = re.sub(r"[\[\]\']", "", string)
    return list(string.split(','))



# pandas dataframe sm-analysis
def mean_normalize(df):
    return (df - df.mean()) / df.std()


def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def fill_mean(df, groups, col):
    return df.groupby(groups)[col].transform(lambda x: x.fillna(x.mean()))


def standardize(df):
    return (df - df.mean()) / df.std()

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


# csv handling

def combine_csvs(directory):
    """Join multiple CSVs into a single dataframe."""

    df = pd.DataFrame()

    p = Path(directory)

    for file in p.iterdir():
        if file.suffix == ".csv":
            temp = pd.read_csv(
            file,
            delimiter=',',
            index_col=0,
            low_memory=False,
            encoding="utf8",
            )
        
            temp['file'] = str(file.stem)
            temp['campaign'] = (file.name.split('_')[-5] +
                                file.name.split('_')[-4])
            temp['release'] = file.name.split('_')[-4]
            temp['government'] = file.name.split('_')[-5]
            df = df.append(temp)

    return df


