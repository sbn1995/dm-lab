#!/bin/python

# Import packages
import sys
import numpy as np
import pandas as pd
from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

"""
Keras Convolutional Neural Network trained by images and its ratings.

Input file format:
    - Images: All of them in a folder
    - Ratings: Text/csv/etc file with "image name,rating"

Started working a bit, not done yet. Gotta discuss the input format.

Authors: Jana Becker, Sabin Grube, Roc Granada
"""

im_size = 32
channels = 3

def main(argv):
    # Check input correctness
    if len(argv) != 3:
        print "Input format: python CoverRatings.py <images folder> <json file>"
        sys.exit()

    images = dict()
    # Load train images
    load_photos(argv[1],images)
    # Load ratings
    load_ratings(argv[2],images)

    #image_rating = merge_dicts(images,ratings)
    #train_network(column(image_rating,0),column(image_rating,1))

    # Change train-test spliting method, this is just an easy approach
    slice = int(len(images.values()) * 0.2)
    x_train = column(images.values()[slice:],0)
    y_train = column(images.values()[slice:],1)
    x_test = column(images.values()[:slice],0)
    y_test = column(images.values()[:slice],1)
    neural_network(x_train,y_train,x_test,y_test)


"""
Convolutional Neural Network model
ACHTUNG: UNDER DEVELOPMENT - far from done lmao

To look at: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""
def neural_network(x_train, y_train, x_test, y_test):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# define base model
def baseline_model():
    # create model
    model = Sequential()

    input_shape = (im_size,im_size,channels)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(1))
    model.add(Activation("softmax"))

    """
    model.add(Dense(32, input_shape=(im_size,im_size,channels,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))"""

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


"""
Modifies dictionary with all the images in the given directory
Dict: {image name, image as numpy array}
Source: https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/
"""
def load_photos(directory,dictionary):
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(32, 32))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        #image = image.reshape((1,) + image.shape)
        # get image id
        image_id = name.split('.')[0]
        dictionary[image_id] = image

"""
Modifies dictionary with all the ratings per image
Dict: {image name, (image, rating)}
"""
def load_ratings(filename,dictionary):
    df = pd.read_json(filename, lines = True)
    for i in range(df.shape[0]):
        rating = df.loc[i,['avg_rating_this_edition']][0]
        image = df.loc[i,['images']][0][0]['path']
        slice = image.find('/') + 1
        dot = image.find('.')
        image = image[slice:dot]
        dictionary[image] = [dictionary[image],float(rating)]
"""
 -- No need for this --
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

"""
Return a Numpy Array with the column 'i' of the 'matrix'
"""
def column(matrix, i):
    return np.array([row[i] for row in matrix if len(row) == 2])

if __name__ == '__main__':
    main(sys.argv)
