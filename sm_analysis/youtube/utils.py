import collections
import functools
import html
import itertools
from pathlib import Path
import random
import re
import string
from timeit import default_timer
import warnings
import os


def extract_id(text):
    # https://webapps.stackexchange.com/questions/54443/format-for-id-of-youtube-video

    # maximally-constrained regex for a video id:
    vid_pattern = re.compile('[0-9A-Za-z_-]{10}[048AEIMQUYcgkosw]')

    # maximally-constrained regex for a channel id:
    chid_pattern = re.compile('[0-9A-Za-z_-]{21}[AQgw]')
    
    vid_match = vid_pattern.findall(text)
    
    chid_match = chid_pattern.findall(text)
    
    return (vid_match,chid_match)