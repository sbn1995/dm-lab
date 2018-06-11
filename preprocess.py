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
from os import listdir, walk, path
from tqdm import tqdm
#from PIL import load_img

#read from single folder
def read_goodreads(file_path):
    """Will return a pandas dataframe with the goodreads data"""
    img_path_full = file_path + "\\full"
    img_path_thumbs_sm = file_path + "\\thumbs\\small"
    img_path_thumbs_bg = file_path + "\\thumbs\\big"
    df = pd.read_json(file_path + ".jl", lines = True)
    return df


def clean_description(df, store):
    """converts all lists of strings in descriptions to a string
    and save the resulting df to store"""
    sh = df.shape
    df['lang'] = ''
    for i in tqdm(range(sh[0])):
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
    return df

def get_img(filepath):
    #img = load_img(filepath)
    pass


def percentages(df, col = 'top_genre', filename='bla.txt'):
    """returns how much of each label there is in the specified column (in percent)"""
    df2 = df.groupby(col)['page'].nunique()
    df2 = df2.sort_values(ascending=False)
    s = df2.sum()
    df2 = df2/s * 100
    df2 = df2.apply(truncate)
    df2 = df2.apply(str)
    df2 = df2 + "%"
    print(df2)
    #df2.to_csv(r''+filename, sep= ' ', mode='a')

def get_top_genres(df, n=5):
    """returns the top n genres (hardcoded)"""
    genres = ['Fiction', 'Nonfiction', 'Mystery', 'Classics', 'Fantasy', 'Childrens', 'History',
              'Science Fiction', 'Romance', 'Historical', 'Sequential Art', 'Young Adult',
              'Philosophy', 'Poetry', 'Religion', 'Biography', 'Horror', 'Favorites',
              'Science']
    genres_n = genres[0:n]
    return genres_n


def filter_avg_rating(rating):
    """rounds the rating (float) to an integer"""
    return int(round(rating))

def filter_added(added):
    """categorize the added/popularity (int) number"""
    categories = [100, 1000, 10000, 100000, 1000000, 10000000]
    return min(categories, key=lambda x:abs(x-added))


def filter_genres(df, genres):
    """discards all entries that arent one of the top genres (returns the whole df)"""
    genreinds = df['top_genre'].isin(genres)
    return df[genreinds]


def categorize_vals(df):
    """categorize the numeric values rating, num_votes and added, returns the whole df"""
    df.avg_rating_all_editions = df.avg_rating_all_editions.apply(filter_avg_rating)
    df.avg_rating_this_edition = df.avg_rating_this_edition.apply(filter_avg_rating)
    df.num_votes_all_editions = df.num_votes_all_editions.apply(filter_added)
    df.num_votes_this_edition = df.num_votes_this_edition.apply(filter_added)
    df.added_by_all_editions = df.added_by_all_editions.apply(filter_added)
    df.added_by_this_edition = df.added_by_this_edition.apply(filter_added)
    return df


def filter_lang(df, lang='en'):
    """discards all entries with a language other than english"""
    langinds = df['lang'] == lang
    return df[langinds]

def rating_to_class(num):
    """truncates a rating to one of the classes between 3.0 and 4.5 (0.1 steps)"""
    classes = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
    return min(classes, key=lambda x:abs(x-num))

def filter_ratings(df):
    """returns df, truncates all ratings to categories (3.0 - 4.5)"""
    df['rating_t'] = df['avg_rating_this_edition'].apply(rating_to_class)
    df['rating_a'] = df['avg_rating_all_editions'].apply(rating_to_class)
    return df

def get_image_path(i,df, full = True):
    """returns the image path to the ith entry of dataframe df"""
    img_name= df.loc[df.index[i],['images']][0][0]['path'].split("/")[-1]
    folders = listdir("data")
    folders = [path.join("data", f) for f in folders]
    folders = [f for f in folders if path.isdir(f)]
    for i in folders:
        if(full):
            img_path = path.join(i,"full", img_name)
            if(path.isfile(img_path)):
                return img_path
        else:
            img_path = path.join(i,"thumbs", "big", img_name)
            if(path.isfile(img_path)):
                return img_path
    return img_path

def convert_to_hdf(original_file, cleaned_file = None):
    """converts the original .jl file to a cleaned .h5 file"""
    if(cleaned_file == None):
        cleaned_file = original_file
    store = pd.HDFStore(cleaned_file + '.h5')
    df = read_goodreads(original_file)
    return clean_description(df, store)


def get_df(filename):
    """returns the NaN-safe df stored in filename.h5"""
    #file_path = ".\data\\0_1_1M"
    #file_path_clean = filename+file_path + "_clean"
    store = pd.HDFStore(filename + '.h5')
    df = store['df'] #load clean df
    df = df.loc[~ df['description'].isnull()]
    return df

def get_df_cleaned(filename, first_time = False, filter_genre = False, filter_language = True, filter_rating = True, categorize = False):
    if(first_time):
        df = convert_to_hdf(filename)
    else:
        try:
            df = get_df(filename)
        except:
            print("create df .h5 file first (set first_time = True)")
    if(filter_language):
        df = filter_lang(df)
    if(categorize):
        df = categorize_vals(df)
    if(filter_genre):
        df = filter_genres(df, get_top_genres(df))
    if(filter_rating):
        df = filter_ratings(df)
    return df


def main():
    #filename = '1_100'
    #df = get_df_cleaned('data\\'+ filename, first_time = True)
    #print(get_image_path(1,df))
    df = get_df_cleaned(path.join("data", "all"), first_time = True)

if __name__ == "__main__":
    main()
