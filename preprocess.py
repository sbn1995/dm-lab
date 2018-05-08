#  -*- coding: utf-8 -*-
"""
This file is for preparing the data as follows:
-read the data into pandas dataframe
TODO:
-merge description text into one String (right now it's a list of strings)
-function for retrieving corresponding img data

"""

import pandas as pd
import numpy as np
import re
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
    """converts all lists of strings in descriptions to a string"""
    sh = df.shape
    for i in range(sh[0]):
        txt = df.loc[i,['description']][0]
        newtxt = txt
        if type(txt) == list:
            newtxt = ' '.join(txt)
            newtxt = re.sub('[^A-Za-z0-9 ]+', '', newtxt)
        elif type(txt) == str:
            newtxt = re.sub('[^A-Za-z0-9 ]+', '', newtxt)
        else:
            newtxt = ''
        df.loc[i,['description']] = newtxt
    store['df'] = df  # save it

def get_img(filepath):
    #img = load_img(filepath)
    pass

def mallet_savetxt(df):
    f = open("goodreads_mallet_big.txt", "a")
    for i in range(df.shape[0]):
        labels = get_labels(df.iloc[i])
        description = str(df.iloc[i]['description'])
        print(labels + '\n' + description)
        f.write(str(i)+'\t'+ 'bla ' + labels + '\t' + description + '\n')
    f.close()

def get_labels(row):
    return '_'.join(str(row['top_genre']).split(' '))


def sort_mallet_keys(path):
    df = pd.read_csv(path, delimiter='\t')
    df = df.sort_values(by=['Occurrences'], ascending=False)
    np.savetxt(r'sorted_keys.txt', df.values, fmt='%s')

def main():
    #read raw data, clean and store to pytable (ONLY NEEDS TO BE DONE ONCE)
    file_path = ".\data\\0_1_1M"
    file_path_clean = file_path + "_clean"
    store = pd.HDFStore(file_path_clean + '.h5')
    # df = read_goodreads(file_path)
    # clean_description(df, store)


    df = store['df'] #load clean df
    df = df.loc[~ df['description'].isnull()]
    sort_mallet_keys('goodreads_llda_big.txt')
    #mallet_savetxt(df)
    #print(df['description'])

    # print(get_labels(df.iloc[0]))
    # print(df.keys())
    #
    # print(df.iloc[0]['description'])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df.keys())
    #     txt = df.loc[1,['description']][0]
    #     print(txt)


if __name__ == "__main__":
    main()
