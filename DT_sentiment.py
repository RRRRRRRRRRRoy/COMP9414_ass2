###################################################################################
# in the question we need to split the data set from the 'dataset.tsv'
# and then using the training set and test set to train and modify our
# our model of Decision tree
###################################################################################
import sys
import numpy as np
import re
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree

# test in the terminal
# data_set_file = sys.argv[1]
# test_set_file = sys.argv[2]

###################################################################################
# local test
###################################################################################
data_set_file = 'training.tsv'
test_set_file = 'test.tsv'

###################################################################################
# get the training set
# in the tsv file, use '\t' to separate the data
###################################################################################
seperator = '\t'
training_set = pd.read_csv(data_set_file, sep = seperator,header = None)
testing_set = pd.read_csv(test_set_file,sep = seperator, header = None)

###################################################################################
# get the content from the dataset(after we modify)
###################################################################################
# testing the content of the training set which type dataframe of panda
# print(type(training_set),len(training_set))
# print("######################")
# print(training_set[1])
# print("######################")
# print(training_set[2])

###################################################################################
# Getting the training set(raw needs to be modified)
# from the example line 18-25
# line 18-25 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
###################################################################################
# id of training sentence's which is from 0 - 3999.
# total number is 4000
training_ID = np.array(training_set[0])
# The content of the training sentence
training_sentence = np.array(training_set[1])
# the result of the training sentence
# which is used to modify the process of learning
training_result = np.array(training_set[2])
###################################################################################
# Getting the testing set(raw needs to be modified)
# from the example line 18-25
# line 18-25 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
###################################################################################
# the number of the data from 0-999 which is 1000
testing_id = np.array(testing_set[0])
# the content. Number is 4000
testing_sentence = np.array(testing_set[1])
# testing result which is used to calculate the accuracy
# negative positive and neural
testing_result = np.array(testing_set[2])


###################################################################################
# Regular Expression: modified the data from the previous step we gotten
# Using the RE module to process the data
# in this part we need to use RE to describe 2 patterns
# 1. illegal_character_pattern
# 2. url_pattern
# syntax : text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
# Source: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
# Source: https://stackoverflow.com/questions/24399820/expression-to-remove-url-links-from-twitter-tweet/24399874
###################################################################################
def get_legal_sentence(raw_sentence):
    url_string = r'(http|https)+:\/\/.*[\r\n]*'
    char_string = r'[^#@_$%\sa-zA-Z\d]'
    result_sentence = list()
    for index in range(0,len(raw_sentence)):
        delete_url = re.sub(url_string,' ',raw_sentence[index])
        delete_illegal_character = re.sub(char_string,'',delete_url)
        result_sentence.append(delete_illegal_character)
    return result_sentence
# deleting extra illegal info
legal_train_sentence = np.array(get_legal_sentence(training_sentence))
legal_test_sentence = np.array(get_legal_sentence(testing_sentence))

