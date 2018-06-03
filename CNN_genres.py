#  -*- coding: utf-8 -*-
import preprocess as pr
import argparse
import sys

import numpy as np
import pandas as pd
from os import listdir, path
import random

import tensorflow as tf
import cv2

"""
Simple Convolutional Neural Network for classification of the genre

Usage:
    - Have in the same folder: CNN_genres.py, preprocess.py, .jl file and folder 'full' with all images
    - Execute --> python CNN_genres.py --json_file <.jl file name (without extension)> --image_size <image size wanted> > results.txt

It is important the part '> results.txt' since the results are displaied on console, which i think
is not shown on mogon.
Training and validation accuracy are the outputs. Validation accuracy should be enough as testing results.

Parameters as batch_size, learning_rate or number of epochs must be changed manually in the code
"""

FLAGS = None
# Using just top 10 genres, to try with all genres, all of them should be listed here (and not delete them while cleaning)
classes = ['Classics','Sequential Art','Science Fiction','Mystery','Fiction','History','Fantasy','Nonfiction','Historical','Childrens']
num_classes = len(classes)

# CNN parameters
batch_size = 10000
learning_rate = 0.001
validation_size = 0.2
epochs = 100

# Image properties
img_size = 150
num_channels = 3

"""
Print progress of the network Training
"""
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, sess, accuracy):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

"""
@parameters
    - 'x' and 'y' arrays samples
    - 'end' where last batch finished
@return next batches for both arrays and new last position of the batch
"""
def next_batch(x,y,end):
    start = end
    end = end + batch_size

    if end > x.shape[0]:
        start = 0
        end = batch_size

    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch, end

"""
Convolutional Neural Network
Returns logits layer
"""
def conv_net(input_layer):
    # Convolutional and Pooling Layer 1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional and Pooling Layer 2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    # Dense Layer
    pool2_flat = tf.reshape(pool2, [tf.shape(pool2)[0], pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=len(classes))
    return logits

"""
Read train images and its labels
"""
def load_train_photos(df):
    # Arrays containing training data
    images = []
    labels = []

    for i in range(df.shape[0]):
        try:
            #TODO: this path isnt correct anymore. get the real path by trying the folders in 'data'
            #im_path = df.loc[i,['images']][0][0]['path']
            im_path = pr.get_image_path(i, df)
            genre = df.loc[i,['top_genre']][0]

            image = cv2.imread(im_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Reshape image and transform values to 0-1 scale
            image = cv2.resize(image, (img_size,img_size))
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            # Add image to list
            images.append(image)

            # Create label's array and asign it to the desired class
            index = classes.index(genre)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
        except:
            pass

    # Shuffle both lists
    union = list(zip(images, labels))
    random.shuffle(union)
    images, labels = zip(*union)

    # Transform to numpy array
    images = np.array(images)
    labels = np.array(labels)

	# Return images and labels separating a validation sample
    idx = int(validation_size * images.shape[0])
    return images[idx:], labels[idx:], images[:idx], labels[:idx]

def main(_):
    global batch_size, img_size

    file_path = FLAGS.json_file

    #file_path_clean = file_path + "_clean_genres"
    #store = pd.HDFStore(file_path_clean + '.h5')

    # Execute this just once!
    #df = pr.read_goodreads(file_path)
    #pr.clean_description(df, store)
    #pr.clean_genres(file_path)
    #sys.exit()

    # Load clean dataset
    #df = store['df']

    #set first_time = True if the .h5 file hasn't been created yet
    df = pr.get_df_cleaned(path.join("data", file_path), first_time = False, filter_genre = True)

    img_size = FLAGS.image_size
    # Load training data
    x_train, y_train, x_val, y_val = load_train_photos(df)

    # Check batch size is not too big, if so, going to have one flag
    if batch_size > x_train.shape[0]:
        batch_size = x_train.shape[0]

    # Number of iterations needed for one epoch
    iterations = int(x_train.shape[0] / batch_size)

    # Auxiliar varaibles to perform batching
    end = 0
    end_val = 0

    # Start session
    sess = tf.Session()

    # Input layer
    x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

    # Output layers
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_classes = tf.argmax(y_true, axis=1)

    # Convolutional network
    logits = conv_net(x)

    # Predictions
    y_pred_classes = tf.argmax(logits, axis=1)
    y_pred = tf.nn.softmax(logits,name='y_pred')

    # Learning parameters
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(y_pred_classes, y_true_classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Define Tensorboard nodes
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./models/genre/log_train", graph=tf.get_default_graph())
    writer_test = tf.summary.FileWriter("./models/genre/log_test", graph=tf.get_default_graph())


    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(epochs * iterations):

        # Batch selectoin and dictionary for training
        x_batch, y_batch, end = next_batch(x_train, y_train, end)
        x_val_batch, y_val_batch, end_val = next_batch(x_val, y_val, end_val)

        feed_dict_tr = {x: x_batch, y_true: y_batch}
        feed_dict_val = {x: x_val_batch, y_true: y_val_batch}

        epoch = int(i / int(x_train.shape[0]/batch_size))

        _, summary = sess.run([optimizer, summary_op], feed_dict=feed_dict_tr)
        # write log
        writer_train.add_summary(summary, epoch * batch_size + i)

        # When one epoch is done, print results
        if i % int(x_train.shape[0]/batch_size) == 0:
            val_loss, summary = sess.run([cost, summary_op], feed_dict=feed_dict_val)
            writer_test.add_summary(summary, epoch * batch_size + i)
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, sess, accuracy)
            saver.save(sess, './models/genre/header-model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='new',
                      help='File containing the books data')
    parser.add_argument('--image_size', type=int, default=150,
                      help='Size of the images to train')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
