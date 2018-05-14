"""
Simple Convolutional Neural Network for classification


Instrucciones:
    - Une todos los ficheros .jl con el codigo de Jana (IMPORTANTE: ponle de nombre new.jl !!)
    - Junta todas las imagenes (las de las carpetas 'full' en una misma carpeta full
    - Incluye en el mismo folder la carpeta full, el fichero new.jl, y los archivos python CNN_ratings.py y preprocess.py

    - Executa con solo "python CNN_ratings.py"

"""

import preprocess as pr
import argparse
import sys

import numpy as np
import pandas as pd
from os import listdir
import random

import tensorflow as tf
import cv2

FLAGS = None
classes = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
#classes = ['Classics','Sequential Art','Science Fiction','Mystery','Fiction','History','Fantasy','Nonfiction','Historical','Childrens','Other']
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
    #count = {}
    #genres = {}
    print("Samples: ",df.shape[0])
    #for i in range(len(classes)):
        #count[i] = 0
    for i in range(df.shape[0]):
        try:
            im_path = df.loc[i,['images']][0][0]['path']
            #genre = df.loc[i,['top_genre']][0]
            """try:
                genres[genre] = genres[genre] + 1
            except:
                genres[genre] = 0"""
                
            rating = df.loc[i,['avg_rating_this_edition']][0]
            image = cv2.imread(im_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Reshape image and transform values to 0-1 scale
            image = cv2.resize(image, (img_size,img_size))
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            # Add image to list
            images.append(image)

            # Create label's array and asign it to the desired class

            """try:
                index = classes.index(genre)
                #count[index] = count[index] + 1
            except:
                index = len(classes) - 1 #Genre 'Other' in last position"""
                #count[index] = count[index] + 1"""
            index = classes.index(int(rating))
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)  # Tensorflow wants class index, not entire array with 0s and 1s
        except:
            pass
    #for i in count.keys():
        #print "Class ", classes[i], " has ", count[i]," samples."
    #print "-----------------------------------------------------"
    #for i in genres.keys():
        #print i, genres[i]
    #sys.exit()

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
    global batch_size

    file_path = "new"
    file_path_clean = file_path + "_clean"
    store = pd.HDFStore(file_path_clean + '.h5')
    # Execute this two just once!
    #df = pr.read_goodreads(file_path)
    #pr.clean_description(df, store)
    #sys.exit()

    df = store['df'] #load clean df

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

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(epochs * iterations):

        # Batch selectoin and dictionary for training
        x_batch, y_batch, end = next_batch(x_train, y_train, end)
        x_val_batch, y_val_batch, end_val = next_batch(x_val, y_val, end_val)

        feed_dict_tr = {x: x_batch, y_true: y_batch}
        feed_dict_val = {x: x_val_batch, y_true: y_val_batch}

        sess.run(optimizer, feed_dict=feed_dict_tr)

        # When one epoch is done, print results
        if i % int(x_train.shape[0]/batch_size) == 0:
            val_loss = sess.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(x_train.shape[0]/batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, sess, accuracy)
            saver.save(sess, './header-model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='train',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
