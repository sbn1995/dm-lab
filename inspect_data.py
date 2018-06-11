import pandas as pd
import numpy as np
from preprocess import *

"""
available keys:
'added_by_all_editions', 'added_by_this_edition', 'author',
'avg_rating_all_editions', 'avg_rating_this_edition', 'description',
'image_urls', 'images', 'isbn', 'name', 'num_pages',
'num_votes_all_editions', 'num_votes_this_edition', 'page',
'release_date', 'series', 'to_reads', 'top_genre', 'lang'
"""
def truncate(f, n=2):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def mallet_savetxt(df, filepath):
    f = open("mallet\\"+filepath, "a")
    for i in range(df.shape[0]):
        labels = get_labels(df.iloc[i])
        description = str(df.iloc[i]['description'])
        print(labels + '\n' + description)
        f.write(str(i)+'\t'+ labels + '\t' + description + '\n')
    f.close()

def sort_mallet_keys(path):
    df = pd.read_csv("mallet\\"+path+".keys", delimiter='\t')
    df = df.sort_values(by=['Occurrences'], ascending=False)
    np.savetxt(r'mallet\\sorted_keys.txt', df.values,  fmt='%s')

def get_labels(row):
    """returns the labels/topics of a specific row in a specific format for mallet"""
    genre = '_'.join(str(row['top_genre']).split(' '))
    numbers = ' '.join([i+"_"+str(row[i]) for i in ['added_by_all_editions',
                                                    #'added_by_this_edition',
                                                    'avg_rating_all_editions',
                                                    #'avg_rating_this_edition',
                                                    'num_votes_all_editions',
                                                    #'num_votes_this_edition',
                                                    ]])
    return genre + " " + numbers

def main():
    filename = 'data/all'
    df = get_df(filename)
    #df = filter_lang(df)
    #df = categorize_vals(df)
    #percentages(df, col = 'num_votes_all_editions')
    df = filter_genres(df, get_top_genres(df))
    df = filter_ratings(df)
    maxlen = 0
    for i in df['description']:
        if(len(set(list(i))) > maxlen):
            maxlen = len(set(list(i)))

    print(maxlen)


    #print(df['avg_rating_this_edition'])
    #df['avg_rating_this_edition'] = (df['avg_rating_this_edition'] - df['avg_rating_this_edition'].min())/df['avg_rating_this_edition'].max() - df['avg_rating_this_edition'].min()

    #print(df['rating_a'])
    #import matplotlib.pyplot as plt
    #plt.hist(df['avg_rating_this_edition'])
    #plt.show()
    #sort_mallet_keys('m_lang')
    #mallet_savetxt(df, 'm_lang')
    #print(df['description'])

if __name__ == "__main__":
    main()
