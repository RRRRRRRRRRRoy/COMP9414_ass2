#######################################################################################################################
# Bernoulli Naive Bayes
#######################################################################################################################
# in the question we need to split the data set from the 'dataset.tsv'
# and then using the training set and test set to train and modify our
# our model of bnb
#######################################################################################################################
import pandas as pd
import csv
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

#######################################################################################################################
# local test ----> for dryrun
#######################################################################################################################
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
# Same with MNB_sementiment.py
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

#######################################################################################################################
# Regular Expression: modified the data from the previous step we gotten
# Using the RE module to process the data
# in this part we need to use RE to describe 2 patterns
# 1. illegal_character_pattern
# 2. url_pattern
# defination of regular expression Source: https://en.wikipedia.org/wiki/Regular_expression
# syntax : text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
# Source: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
# Source: https://stackoverflow.com/questions/24399820/expression-to-remove-url-links-from-twitter-tweet/24399874
######################################################################################################################
def polishing_illegal_sentence(raw_sentence):
    protocal_part = r"(http|https|)"
    capital_part = r"([a-zA-Z]|[0-9]|[$-_@.&+]"
    slash_part = r"[!*\(\),]"
    url_pattern_obj = re.compile(fr"{protocal_part}+:{capital_part}|{slash_part})")
    illegal_character_pattern = r'[^#@_$%\sa-zA-Z\d]'
    result_sentence = list()
    for index in range(len(raw_sentence)):
        # delete_url = re.sub(url_pattern_obj,' ',raw_sentence[index])
        # delete_illegal_character = re.sub(illegal_character_pattern,'',delete_url)
        result_sentence.append(re.sub(illegal_character_pattern,'',re.sub(url_pattern_obj,' ',raw_sentence[index])))
    return result_sentence

# deleting extra illegal information from dataset
legal_training_sentence = np.array(polishing_illegal_sentence(training_sentence))
legal_testing_sentence = np.array(polishing_illegal_sentence(testing_sentence))

#######################################################################################################################
# Training
# Using the function called 'CountVectorizer' given by sklearn to extract features
# after extracting features from the sentence using the function call fit to train the module
# 2 parameters in the CountVectorizer
# token_pattern ----> Regular expression denoting what constitutes a "token" ----> regular expression
# max_feature ----> the maximum quantity of the features
# Source: https://stackoverflow.com/questions/33004946/token-pattern-in-countvectorizer-scikit-learn
# from the example line 44-46
# line 44-46 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
#######################################################################################################################
# writing the report to get the data info
maximum_feature = 1000
String_pattern = r'[#@_$%\w\d]{2,}'
count = CountVectorizer(token_pattern= String_pattern)
# Line 46 and 49 Source:  https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
X_training_bag_of_words = count.fit_transform(legal_training_sentence)
X_testing_bag_of_words = count.transform(legal_testing_sentence)

#######################################################################################################################
# Using the BernoulliNB module from sklearn
# From the ass2.pdf there is not other conditions(parameters)
# we need to modify. Therefore, just putting the training data into the model
# The usage of BernoulliNB() is given by example.py
# from the example line 51-54
# line 51-54 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
######################################################################################################################
clf = BernoulliNB()
model = clf.fit(X_training_bag_of_words, training_result)
predict_result = model.predict(X_testing_bag_of_words)

for i in range(len(testing_sentence)):
    print(testing_id[i],predict_result[i])