from preprocess import *
import cv2
import random

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Embedding, LSTM, Concatenate
from keras.models import Model
import keras
import argparse

import json
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from CNN_genres import load_train_photos

EMBED_DIM = 8
CLASSES = ['Classics','Sequential Art','Science Fiction','Mystery','Fiction','History','Fantasy','Nonfiction','Historical','Childrens']
NUM_CLASSES = len(CLASSES)

FLAGS = None

# CNN parameters
BATCH_SIZE = 64
LR = 0.001
VAL_SIZE = 0.2
TEST_SIZE = 0.2
EPOCHS = 100

# Image properties
IMG_SIZE = 150
NUM_CHANNELS = 3


def image_model(img_size):
    # This returns a tensor
    image = Input(batch_shape=(None,img_size,img_size, NUM_CHANNELS))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv2D(16, (3,3), activation='relu')(image)
    x = MaxPooling2D()(x)
    x = Conv2D(8, (3,3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    return image, x


def text_model(charsize):
    # This returns a tensor
    word = Input(batch_shape=(None,None))

    #wordvector embedding
    x = Embedding(charsize, EMBED_DIM)(word)

    #LSTM
    x = LSTM(64)(x)

    return word, x


def merge_models(image, imagepred, word, wordpred):
    imgword = Concatenate()([imagepred, wordpred])

    pred = Dense(NUM_CLASSES, activation = 'softmax')(imgword)

    model = Model(inputs=[image, word], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def text_to_vec(description, charset):
    charints = range(len(charset))
    charlist = list(charset)
    chardict = {charlist[i]:charints[i] for i in range(len(charints))}

    descvec = list(description)
    for i in range(len(descvec)):
        descvec[i] = chardict[description[i]]

    return np.array(descvec)


def get_charset(df):
    charset = set()
    maxlen = 0
    for i in df['description']:
        charset = charset.union(set(i))
        maxlen = len(charset)

    return charset


def load_train_data(df, charset):
    # Arrays containing training data
    images = []
    text = []
    labels = []

    for i in range(df.shape[0]):
        try:
            im_path = get_image_path(i, df)
            genre = df.loc[df.index[i],['top_genre']][0]
            description = df.loc[df.index[i], ['description']][0]

            wordvec = text_to_vec(description, charset)

            text.append(wordvec)

            image = cv2.imread(im_path)

            # Reshape image and transform values to 0-1 scale
            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            # Add image to list
            images.append(image)


            # Create label's array and asign it to the desired class
            index = CLASSES.index(genre)
            label = np.zeros(len(CLASSES))
            label[index] = 1.0
            labels.append(label)
        except Exception as e:
            print("Error during data gathering: ",e)


    # Shuffle both lists
    union = list(zip(images, text, labels))
    random.shuffle(union)
    images, text, labels = zip(*union)

    text = keras.preprocessing.sequence.pad_sequences(text)

    # Transform to numpy array
    images = np.array(images)
    #text = np.array(text)
    labels = np.array(labels)


	# Return images and labels separating a validation sample
    idx1 = int(VAL_SIZE * images.shape[0])
    idx2 = int(TEST_SIZE * images.shape[0]) + idx1

    #train, validation, test
    img_data = [images[idx2:], labels[idx2:], images[:idx1], labels[:idx1], images[idx1:idx2], labels[idx1:idx2]]
    text_data = [text[idx2:], labels[idx2:], text[:idx1], labels[:idx1], text[idx1:idx2], labels[idx1:idx2]]
    return img_data, text_data


def main():
    global BATCH_SIZE, IMG_SIZE

    file_path = FLAGS.json_file

    #set first_time = True if the .h5 file hasn't been created yet
    df = get_df_cleaned(path.join("data", file_path), first_time = False, filter_genre = True)

    #determine set of unique characters in descriptions of dataframe
    charset = get_charset(df)

    IMG_SIZE = FLAGS.image_size
    # Load training data
    img_data, text_data = load_train_data(df, charset)
    x_train_img, y_train_img, x_val_img, y_val_img, x_test_img, y_test_img = img_data
    x_train_text, y_train_text, x_val_text, y_val_text, x_test_text, y_test_text = text_data

    y_train, y_test, y_val = y_train_img, y_test_img, y_val_img

    if BATCH_SIZE > x_train_img.shape[0]:
        BATCH_SIZE = x_train_img.shape[0]

    img, imgpred = image_model(IMG_SIZE)
    txt, txtpred = text_model(len(charset))

    print("BAtch size:", BATCH_SIZE)
    model = merge_models(img, imgpred, txt, txtpred)

    #tensorboard callback
    tbcall = keras.callbacks.TensorBoard(log_dir='./models/imgtext/logs')

    print(x_train_img.shape, x_train_text.shape, y_train.shape)
    print(x_val_img.shape, x_val_text.shape, y_val.shape)
    model.fit(x=[x_train_img, x_train_text], y=y_train,
          batch_size=BATCH_SIZE, epochs=5, shuffle=False,
          validation_data=([x_val_img, x_val_text], y_val), callbacks=[tbcall])

    model.evaluate(x=[x_test_img, x_test_text], y=y_test)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='data/all',
                      help='File containing the books data')
    parser.add_argument('--image_size', type=int, default=150,
                      help='Size of the images to train')
    FLAGS, unparsed = parser.parse_known_args()
    main()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
