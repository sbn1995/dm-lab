import matplotlib.pyplot as plt

from bokeh.plotting import figure, show, output_file


#assuming data is a dict of dicts
def goodreads_pcc(data):
    data_clean = get_rating_length(data)

    plot_data(data_clean)

def plot_data(df):
    #for diff genres
    #colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    #colors = [colormap[x] for x in flowers['species']]
    #
    # p = figure(title = "Iris Morphology")
    # p.xaxis.axis_label = 'Petal Length'
    # p.yaxis.axis_label = 'Petal Width'
    #
    # p.circle(flowers["petal_length"], flowers["petal_width"],
    #          color=colors, fill_alpha=0.2, size=10)
    #
    # output_file("iris.html", title="iris.py example")
    #
    # show(p)

    # output to static HTML file
    output_file("scatter.html")

    p = figure(plot_width=400, plot_height=400)
    p.xaxis.axis_label = 'Length'
    p.yaxis.axis_label = 'Rating'
    p.title = "Correlation of Book Length and Rating"

    # add a circle renderer with a size, color, and alpha
    p.circle(df, x='mpg', y='hp', size=20, color="navy", alpha=0.5)

    # show the results
    show(p)


#TODO
def get_rating_length(data):
    pass
