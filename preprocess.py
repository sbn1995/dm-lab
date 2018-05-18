#  -*- coding: utf-8 -*-
"""
This file is for preparing the data as follows:
-read the data into pandas dataframe
-merge description text into one String (right now it's a list of strings)
-function for retrieving corresponding img data

"""

import pandas as pd
import numpy as np
import re
from langdetect import detect
#from PIL import load_img

#read from single folder
def read_goodreads(file_path):
    """Will return a pandas dataframe with the goodreads data"""
    img_path_full = file_path + "\\full"
    img_path_thumbs_sm = file_path + "\\thumbs\\small"
    img_path_thumbs_bg = file_path + "\\thumbs\\big"
    df = pd.read_json(file_path + ".jl", lines = True)
    return df

"""
This method should remove from the dataset all books not belonging to one of the genres listed
"""
def clean_genres(df, store):
    classes = ['Classics','Sequential Art','Science Fiction','Mystery','Fiction','History','Fantasy','Nonfiction','Historical','Childrens']
    pass

"""
This method should remove from the dataset all books not belonging to one of the ratings listed
"""
def clean_ratings(df, store):
    classes = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
    # round(rating,1) to use just one floating point
    pass


def clean_description(df, store):
    """converts all lists of strings in descriptions to a string"""
    sh = df.shape
    df['lang'] = ''
    for i in range(sh[0]):
        txt = df.loc[i,['description']][0]
        newtxt = txt
        lang = ''
        if type(txt) == list:
            newtxt = ' '.join(txt)
            newtxt = re.sub('[^A-Za-z0-9 ]+', '', newtxt)
            try:
                lang = detect(newtxt)
            except:
                pass
        elif type(txt) == str:
            newtxt = re.sub('[^A-Za-z0-9 ,!?\'\":;$%&(=).]+', '', newtxt)
            try:
                lang = detect(newtxt)
            except:
                pass
        else:
            newtxt = ''
        df.loc[i,['description']] = newtxt
        df.loc[i,['lang']] = lang
    store['df'] = df  # save it

def get_img(filepath):
    #img = load_img(filepath)
    pass


def convert_to_hdf(original_file, cleaned_file):
    store = pd.HDFStore(cleaned_file + '.h5')
    df = read_goodreads(original_file)
    clean_description(df, store)
