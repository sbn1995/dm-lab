import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Set1
import pandas as pd
import numpy as np
from inspect_data import *


#assuming data is a dict of dicts
def goodreads_pcc(df):
    data_clean = get_rating_length(df)
    plot_data(data_clean)

def plot_data(df):
    output_file("scatter.html")

    p = figure(plot_width=1000, plot_height=600)
    p.yaxis.axis_label = 'Length'
    p.xaxis.axis_label = 'Rating'
    #p.title = "Correlation of Book Length and Rating"

    print(df)
    color_mapper = CategoricalColorMapper(factors=df['top_genre'].unique(), palette=Set1[5])

    # add a circle renderer with a size, color, and alpha
    for ind,genre in enumerate(df['top_genre'].unique()):
        p.circle(source=df[df['top_genre']==genre], x='avg_rating_all_editions', y='num_pages', size=2,
             color=Set1[5][ind],legend=genre, alpha=0.5)
    p.legend.click_policy="hide"


    # show the results
    show(p)


def get_rating_length(df):
    return df[['avg_rating_all_editions','num_pages', 'top_genre']]

def main():
    filename = ''
    df = get_df(filename)
    #write_langs(df, "experiments\\languages.txt")
    df = filter_lang(df)
    #df = categorize_vals(df)
    df = filter_genres(df, get_top_genres(df))

    #df = store['df'] #load clean df
    df = df.dropna(axis=0)
    goodreads_pcc(df)


if __name__ == "__main__":
    main()
