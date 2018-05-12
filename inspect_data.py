import pandas as pd
import numpy as np

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
    genre = '_'.join(str(row['top_genre']).split(' '))
    numbers = ' '.join([i+"_"+str(row[i]) for i in ['added_by_all_editions',
                                                    #'added_by_this_edition',
                                                    'avg_rating_all_editions',
                                                    #'avg_rating_this_edition',
                                                    'num_votes_all_editions',
                                                    #'num_votes_this_edition',
                                                    ]])
    return genre + " " + numbers

def get_df(filename):
    file_path = ".\data\\0_1_1M"
    file_path_clean = filename+file_path + "_clean"
    store = pd.HDFStore(file_path_clean + '.h5')
    df = store['df'] #load clean df
    df = df.loc[~ df['description'].isnull()]
    return df

def percentages(df, col = 'top_genre', filename='bla.txt'):
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
    genres = ['Fiction', 'Nonfiction', 'Mystery', 'Classics', 'Fantasy', 'Childrens', 'History',
              'Science Fiction', 'Romance', 'Historical', 'Sequential Art', 'Young Adult',
              'Philosophy', 'Poetry', 'Religion', 'Biography', 'Horror', 'Favorites',
              'Science']
    genres_n = genres[0:n]
    return genres_n


def filter_avg_rating(rating):
    return int(round(rating))

def filter_added(added):
    categories = [100, 1000, 10000, 100000, 1000000, 10000000]
    return min(categories, key=lambda x:abs(x-added))


def filter_genres(df, genres):
    genreinds = df['top_genre'].isin(genres)
    return df[genreinds]


def categorize_vals(df):
    df.avg_rating_all_editions = df.avg_rating_all_editions.apply(filter_avg_rating)
    df.avg_rating_this_edition = df.avg_rating_this_edition.apply(filter_avg_rating)
    df.num_votes_all_editions = df.num_votes_all_editions.apply(filter_added)
    df.num_votes_this_edition = df.num_votes_this_edition.apply(filter_added)
    df.added_by_all_editions = df.added_by_all_editions.apply(filter_added)
    df.added_by_this_edition = df.added_by_this_edition.apply(filter_added)
    return df


def filter_lang(df, lang='en'):
    langinds = df['lang'] == lang
    return df[langinds]


def main():
    filename = ''
    #df = get_df(filename)
    #df = filter_lang(df)
    #df = categorize_vals(df)
    #percentages(df, col = 'num_votes_all_editions')
    #df = filter_genres(df, get_top_genres(df))


    sort_mallet_keys('m_lang')
    #mallet_savetxt(df, 'm_lang')
    #print(df['description'])

if __name__ == "__main__":
    main()
