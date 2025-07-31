import tensorflow as tf
import json, csv
import numpy as np
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import model_from_json
import features
from preprocess import *


def read_trial_set():
    '''
    reads content from all tweets from the trial set 
    
    returns: 
    result: list of strings
    '''
    result = []
    with open("resources/trial.csv", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        kopfzeile = next(reader)
        for l in reader:
            result.append(l[3])
    return result

def read_test_set():
    '''
    reads content from all tweets from the test set 
    
    returns: 
    result: list of strings
    '''
    result = []
    with open("resources/test.csv", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        kopfzeile = next(reader)
        for l in reader:
            result.append(l[4])
    return result


def write_to_submission(values):
    '''    
    writes prediction for the labels for the tweets in the file labels.txt

    input: 
    values: list
    '''
    with open("labels.txt", "w") as f:
        for v in values:
            f.write(v)
            f.write('\n')


# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# num words is equal to input dictionary
tokenizer = Tokenizer(num_words=len(dictionary)+1)

labels = [0,1,2]

def convert_text_to_index_array(text):
    '''
    this utility makes sure that all the words in your input
    are registered in the dictionary
    before trying to turn them into a matrix.

    input:
    text: string

    return:
    wordIndices: list of dictionaries
    '''
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')


if (True): # True is trial prepare
    trial_solution = []
    #trial = read_trial_set()
    trial = read_test_set()
    for content in trial:
        # clean tweet and preprocess it
        evalSentence = features.clean_dataset.clean_string(content)
        testArr = convert_text_to_index_array(evalSentence)
        # convert to matrix
        input = tokenizer.sequences_to_matrix([testArr], mode='binary')
        # predit label for tweet
        pred = model.predict(input)
        # append prediction to solution array
        trial_solution.append(str(labels[np.argmax(pred)]))
    print(trial_solution)
    write_to_submission(trial_solution)

else:
    test_set = []
    test_solution=[]
    gold_solution = []
    with open("resources/partI_task2and3_gold_standard.csv", encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        kopfzeile = next(reader)
        for l in reader:
            tp = (l[3], l[4], l[5])
            test_set.append(tp)
    for content in test_set:
        evalSentence = clean_string(content[1])
        gold_solution.append(content[2])
        testArr = convert_text_to_index_array(evalSentence)
        input = tokenizer.sequences_to_matrix([testArr], mode='binary')
        pred = model.predict(input)
        test_solution.append(str(labels[np.argmax(pred)]))
    print("test",test_solution)
    print("gold",gold_solution)#
    c = 0
    for i in range(0,len(test_solution)):
        if (test_solution[i] == gold_solution[i]):
            c += 1
    print(c/len(test_solution))

