import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
import pandas as pd
import numpy as np
from preprocess import *


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
    # add a circle renderer with a size, color, and alpha
    p.circle(source=df, x='avg_rating_all_editions', y='num_pages', size=2, color="navy", alpha=0.5)

    # show the results
    show(p)


def get_rating_length(df):
    return df[['avg_rating_all_editions','num_pages']]

def main():
    #read raw data, clean and store to pytable (ONLY NEEDS TO BE DONE ONCE)
    file_path = ".\data\\0_1_1M"
    file_path_clean = file_path + "_clean"
    store = pd.HDFStore(file_path_clean + '.h5')


    df = store['df'] #load clean df
    df = df.dropna(axis=0)
    goodreads_pcc(df)


if __name__ == "__main__":
    main()
