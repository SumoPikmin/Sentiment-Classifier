import csv, json
import numpy as np

import tensorflow as tf
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

from preprocess import *


#https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/


def prepare_data():
    # format lang, content, label
    tupled_dataset = []

    with open("resources/train.csv", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        kopfzeile = next(reader)
        for l in reader:
            tp = (l[2], l[3], l[4])
            tupled_dataset.append(tp)

    with open("resources/partI_task2and3_gold_standard.csv", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        kopfzeile = next(reader)
        for l in reader:
            tp = (l[3], l[4], l[5])
            #tupled_dataset.append(tp)

    
    cleaned_dataset = clean_dataset(tupled_dataset)
    tweet_train_set = cleaned_dataset
    # labels as array
    train_y = [x[2] for x in tweet_train_set]
    # content of tweets as arrays
    train_x = [clean_string(x[1]) for x in tweet_train_set]

    return train_x, train_y

def convert_text_to_index_array(text):
# one really important thing that `text_to_word_sequence` does
# is make all texts the same length -- in this case, the length
# of the longest text in the set.
    return [dict_tok[word] for word in kpt.text_to_word_sequence(text)]


train_x, train_y = prepare_data()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
# make dictionary out of words
dict_tok = tokenizer.word_index

# open dict with indexed words
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dict_tok, dictionary_file)


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)
print(allWordIndices)
# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
# num classes it total number of classes (0,1,2)
train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)

# model
model = Sequential()
model.add(Dense(512, activation='hard_sigmoid'))
model.add(Dense(256, activation='hard_sigmoid'))
model.add(Dense(3, activation='sigmoid'))

# to be adapted
model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# hyperparameter
model.fit(train_x, train_y,
    batch_size=64,
    epochs=50,
    verbose=1,
    validation_split=0.2,
    shuffle=True)

# write model to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

