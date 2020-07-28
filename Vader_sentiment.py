#######################################################################################################################
# Vader
#######################################################################################################################
# Implementation of Vader
#######################################################################################################################
import sys
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree

# test in the terminal
# data_set_file = sys.argv[1]
# test_set_file = sys.argv[2]

#######################################################################################################################
# local test
#######################################################################################################################
data_set_file = 'training.tsv'
test_set_file = 'test.tsv'

#######################################################################################################################
# get the training set
# in the tsv file, use '\t' to separate the data
#######################################################################################################################
seperator = '\t'
training_set = pd.read_csv(data_set_file, sep = seperator,header = None)
testing_set = pd.read_csv(test_set_file,sep = seperator, header = None)

#######################################################################################################################
# get the content from the dataset(after we modify)
#######################################################################################################################
# testing the content of the training set which type dataframe of panda
# print(type(training_set),len(training_set))
# print("######################")
# print(training_set[1])
# print("######################")
# print(training_set[2])

#######################################################################################################################
# Getting the training set(raw needs to be modified)
# from the example line 18-25
# line 18-25 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
#######################################################################################################################
# id of training sentence's which is from 0 - 3999.
# total number is 4000
training_ID = np.array(training_set[0])
# The content of the training sentence
training_sentence = np.array(training_set[1])
# the result of the training sentence
# which is used to modify the process of learning
training_result = np.array(training_set[2])
#######################################################################################################################
# Getting the testing set(raw needs to be modified)
# from the example line 18-25
# line 18-25 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
#######################################################################################################################
# the number of the data from 0-999 which is 1000
testing_id = np.array(testing_set[0])
# the content. Number is 4000
testing_sentence = np.array(testing_set[1])
# testing result which is used to calculate the accuracy
# negative positive and neural
testing_result = np.array(testing_set[2])
