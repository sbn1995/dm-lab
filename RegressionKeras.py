#!/bin/python

# Import packages
import sys
import numpy as np
from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"""
Keras Convolutional Neural Network train by images and its ratings.

Input file format:
    - Images: All of them in a folder
    - Ratings: Text/csv/etc file with "image name,rating"

Started working a bit, not done yet. Gotta discuss the input format.

Authors: Jana Becker, Sabin Grube, Roc Granada
"""

def main(argv):
    # Check input correctness
    if len(argv) != 3:
        print "Input format: python CoverRatings.py <images folder> <ratings file>"
        sys.exit()

    # Load train images
    images = load_photos(argv[1])
    # Load ratings
    ratings = load_ratings(argv[2])

    image_rating = merge_dicts(images,ratings)
    train_network(column(image_rating,0),column(image_rating,1))


"""
Convolutional Neural Network model
ACHTUNG: UNDER DEVELOPMENT - far from done lmao
To estimate new ratings, we should use regression (right?)
Another option is to have 5 classes (1 to 5 stars) and classify them like this,
but this way does not let use the same neural network to predict book popularity
"""
def train_network(x_train, y_train):
    model = Sequential()
    model.add(Dense(32, input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)

    kfold = KFold(n_splits=2, random_state=seed)
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


"""
Return a dictionary with all the images in the given directory
Dict: {image name, image as numpy array}
Source: https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/
"""
def load_photos(directory):
    images = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(32, 32))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get image id
        image_id = name.split('.')[0]
        images[image_id] = image
    return images

"""
Return a dictionary with all the ratings per image
Dict: {image name, rating}
"""
def load_ratings(filename):
    # Dictionary with {image name, rating}
    ratings = dict()
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        split_line = line.split(',')
        name = split_line[0]
        rating = split_line[1]
        ratings[name] = int(rating)
    file.close()
    return ratings

"""
Given two dictionaries:
  - first {image name, image as array}
  - second {image name, image rating}
merge them as a list with tuples (image as array, image rating)
"""
def merge_dicts(first, second):
    merge = []
    for k in first.keys():
        try:
            merge.append((first[k],second[k]))
        except:
            pass
    return merge

def column(matrix, i):
    return [row[i] for row in matrix]

if __name__ == '__main__':
    main(sys.argv)
