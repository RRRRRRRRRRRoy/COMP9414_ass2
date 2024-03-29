#######################################################################################################################
# Multinomial Naive Bayes
#######################################################################################################################
# in the question we need to split the data set from the 'dataset.tsv'
# and then using the training set and test set to train and modify our
# our model of MNB
#######################################################################################################################
import pandas as pd
import csv
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.stem import PorterStemmer
#######################################################################################################################
# local test ----> for dryrun
#######################################################################################################################
data_set_file = sys.argv[1]
test_set_file = sys.argv[2]

#######################################################################################################################
# local test
#######################################################################################################################
# data_set_file = 'training.tsv'
# test_set_file = 'test.tsv'

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
# using these 2 function remove_stopwards and stemming_words to test whether stem extraction will effect the result.
# download the data from nltk stopwards
#######################################################################################################################
def stemming_words(sentence):
    ps = PorterStemmer()
    words_in_sentence = sentence.split(" ")
    getting_stem_words = list()
    len_word_in_sentence = len(words_in_sentence)
    for index in range(len_word_in_sentence):
        stemmed = ps.stem(words_in_sentence[index])
        getting_stem_words.append(stemmed)
    stemmed_sentence = " ".join(getting_stem_words[index] for index in range(len(getting_stem_words)))
    return stemmed_sentence
#######################################################################################################################
# Regular Expression: modified the data from the previous step we gotten
# Using the RE module to process the data
# in this part we need to use RE to describe 2 patterns
# 1. illegal_character_pattern
# 2. url_pattern
# defination of regular expression Source: https://en.wikipedia.org/wiki/Regular_expression
# syntax : text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
# Source: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
######################################################################################################################
def polishing_illegal_sentence(raw_sentence):
    # Source: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    url_pattern_obj = r'^https?:\/\/.*[\r\n]*'
    illegal_character_pattern = r'[^#@_$%\sa-zA-Z\d]'
    result_sentence = list()
    for index in range(len(raw_sentence)):
        # only stem
        delete_illegal_part = re.sub(illegal_character_pattern,'',re.sub(url_pattern_obj,' ',raw_sentence[index]))
        Stemm_sentence = stemming_words(delete_illegal_part)
        result_sentence.append(Stemm_sentence)
    return result_sentence

# deleting extra illegal information from dataset
polishing_training_sentence = polishing_illegal_sentence(training_sentence)
polished_testing_sentence = polishing_illegal_sentence(testing_sentence)
legal_training_sentence = np.array(polishing_training_sentence)
legal_testing_sentence = np.array(polished_testing_sentence)

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
maximum_feature = 2000
Low = True
Up = False
String_pattern = r'[#@_$%\w\d]{2,}'
count = CountVectorizer(token_pattern= String_pattern,max_features=maximum_feature,lowercase=Up)
# Line 46 and 49 Source:  https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
X_training_bag_words = count.fit_transform(legal_training_sentence)
X_testing_bag_words = count.transform(legal_testing_sentence)

#######################################################################################################################
# Using the MultinomialNB module from sklearn
# From the ass2.pdf there is not other conditions(parameters)
# we need to modify. Therefore, just putting the training data into the model
# The usage of MultinomialNB() is given by example.py
# from the example line 56-60
# line 56-60 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
######################################################################################################################
classification = MultinomialNB(alpha=1)
MNB_model = classification.fit(X_training_bag_words, training_result)
predict_result_sentiment = MNB_model.predict(X_testing_bag_words)

#######################################################################################################################
# Printing the classification report
# Using the function in Sklearn to print the result of classification
# This is based on the code in fuction predict_and_test in example.py line 15
# Line 15 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
######################################################################################################################
# print(classification_report(testing_result,predict_result_MNB))

length_testing_sentence = len(testing_sentence)
for index in range(length_testing_sentence):
    print(testing_id[index],predict_result_sentiment[index])