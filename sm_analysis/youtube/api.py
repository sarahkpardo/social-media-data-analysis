import collections
from datetime import datetime
import functools
import html
import itertools
import os
from pathlib import Path
import random
import re
import string
from timeit import default_timer
import warnings
import yaml

warnings.filterwarnings('ignore')

import googleapiclient.discovery
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sm_analysis
from sm_analysis.utils import *
from ..utils import *

with open('auth.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    SERVICE_NAME = config["SERVICE_NAME"]
    VERSION = config["VERSION"]
    DEV_KEY = config["DEV_KEY"]
    
class Collection:
    def __init__(self):
        self.nextPageToken = None
        self.method = None
        self.items = []
        
    def api_request(self,
                    part="snippet",
                    **kwargs):
        pass
        
    def parse(self, response):
        pass
    
    def to_dataframe(self):
        pass
    
    
class Search(Collection):
    def __init__(self, query):
        self.method = None
        self.items = []
        self.nextPageToken = None
        
        self.query = query
        
    def api_request(self,
                    part="snippet",
                    **kwargs):
        """
        optional kwargs to search().list():
            {channelId=None, 
             channelType=None, 
             location=None, 
             locationRadius=None, 
             maxResults=None, 
             order=None, 
             pageToken=None, 
             publishedAfter=None, 
             publishedBefore=None, 
             q=None, 
             regionCode=None, 
             relatedToVideoId=None, 
             relevanceLanguage=None, 
             safeSearch=None, 
             topicId=None, 
             type=None, 
             videoCaption=None, 
             videoCategoryId=None, 
             videoDefinition=None, 
             videoDimension=None, 
             videoDuration=None, 
             videoEmbeddable=None, 
             videoLicense=None, 
             videoSyndicated=None, 
             videoType=None,
         }
         """
    
        youtube = build(SERVICE_NAME, 
                        VERSION, 
                        developerKey=DEV_KEY)

        request = youtube.search().list(
            part=part,
            q=self.query,
            pageToken=self.nextPageToken,
            **kwargs
        )
        try:
            response = request.execute()
        except HttpError as e:
            print('Error response status code : {0}, reason : {1}'
                  .format(e.status_code, e.error_details))

        youtube.close()

        return response
        

    def parse(self, response):
        self.nextPageToken = response["nextPageToken"]
        for item in response["items"]:
            self.items.append(item["id"] | item["snippet"])
        
    def clear(self):
        self.nextPageToken = None
        self.items = []
        
    def to_dataframe(self):
        return pd.DataFrame(self.items)
        
        

class CommentThreads(Collection):
    def __init__(self, video_id):
        self.method = None
        self.items = []
        self.nextPageToken = None
        
        self.video_id = video_id
        
    def api_request(self,
                    part="snippet",
                    **kwargs):
        
        youtube = build(SERVICE_NAME, 
                        VERSION, 
                        developerKey=DEV_KEY)

        request = youtube.commentThreads().list(
            part=part,
            videoId=self.video_id,
            pageToken=self.nextPageToken,
            **kwargs
        )
        try:
            response = request.execute()
        except HttpError as e:
            print('Error response status code : {0}, reason : {1}'
                  .format(e.status_code, e.error_details))

        youtube.close()

        return response
        

    def parse(self, response):
        self.items += response["items"]
        try:
            self.nextPageToken = response["nextPageToken"]
        except KeyError as e:
            return
        
    def clear(self):
        self.nextPageToken = None
        self.items = []
        
    def to_dataframe(self):
        df = (pd.DataFrame(self.items)
             .drop(["kind","etag"],axis=1)
             )
        idx = df["id"]
        df = (pd.DataFrame
             .from_records(df["snippet"])
             .set_index(idx))
        snippet = (pd.DataFrame
                  .from_records(df["topLevelComment"])
                  .drop(["kind","etag"],axis=1)
                  )
        df2 = (pd.DataFrame
              .from_records(snippet["snippet"],index=snippet["id"])
              )
        df = (df
             .drop(["videoId","topLevelComment"],axis=1)
             )
        df = df2.join(df)

        auth_chid = (pd.DataFrame
                     .from_records(df["authorChannelId"])
                     .set_index(df.index)
                     .rename(columns = {'value':'authorchannelId'})
                    )
        df = (df
              .drop(["authorChannelId"],axis=1)
              .join(auth_chid)
             )
        df["parentId"] = df.index

        return df
        
        
class Resource():
    def __init__(self, contents):
        self.method = None
        self.contents = contents

class Comment(Resource):
    
    def __init__(self, 
                 contents,
                 video_id=None
                ):
        self.method = None
        
        try: 
            if contents["topLevelComment"]:
                self.contents = contents["topLevelComment"]
        except KeyError as e:
            self.contents = contents
                
        self.id = {'id':contents["id"]}
        self.snippet = self.contents["snippet"]
        self.video_id = {'videoId':video_id}
        self.text = html.unescape(self.snippet["textDisplay"])
        
        
def follow_thread(parent_id,
                  max_pages=3,
                 ):
    replies = []
    page_token = None

    youtube = build(SERVICE_NAME, 
                    VERSION, 
                    developerKey=DEV_KEY)

    request = youtube.comments().list(
        part="snippet",
        parentId=parent_id
    )
    try:
        response = request.execute()
    except HttpError as e:
        print('Error response status code : {0}, reason : {1}'
              .format(e.status_code, e.error_details))
    
    replies += response["items"]
    try:
        page_token = response["nextPageToken"]
    except KeyError as e:
        youtube.close()
        return replies
    
    pages = 0
    while pages <= max_pages:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            pageToken=page_token
        )
        try:
            response = request.execute()
        except HttpError as e:
            print('Error response status code : {0}, reason : {1}'
                  .format(e.status_code, e.error_details))
            break
        replies += response["items"]
        try:
            page_token = response["nextPageToken"]
        except KeyError as e:
            youtube.close()
            return replies
        pages += 1
        
    youtube.close()

    return replies

def thread_to_dataframe(thread):
    df = pd.DataFrame(thread)
    df = (pd.DataFrame
          .from_records(df["snippet"])
          .set_index(df["id"])
         )
    auth_chid = (pd.DataFrame
                     .from_records(df["authorChannelId"])
                     .set_index(df.index)
                     .rename(columns = {'value':'authorchannelId'})
                    )
    df = (df
          .drop(["authorChannelId"],axis=1)
          .join(auth_chid)
         )
    return df