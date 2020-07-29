#######################################################################################################################
# Decision Tree
#######################################################################################################################
# in the question we need to split the data set from the 'dataset.tsv'
# and then using the training set and test set to train and modify our
# our model of Decision tree
#######################################################################################################################
import sys
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree

# import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# test in the terminal
# data_set_file = sys.argv[1]
# test_set_file = sys.argv[2]

#######################################################################################################################
# local test
#######################################################################################################################
from sklearn.metrics import classification_report

data_set_file = 'training.tsv'
test_set_file = 'test.tsv'

#######################################################################################################################
# get the training set
# in the tsv file, use '\t' to separate the data
#######################################################################################################################
seperator = '\t'
training_set = pd.read_csv(data_set_file, sep=seperator, header=None)
testing_set = pd.read_csv(test_set_file, sep=seperator, header=None)

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

#######################################################################################################################
# Question 4: using these 2 function remove_stopwards and stemming_words to test whether the stopwords and stem extraction
# will effect the result.
# download the data from nltk stopwards
#######################################################################################################################
def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    words_in_sentence = sentence.split(" ")
    filtered_words = list()
    remove_stop_sentence = ''
    for index in range(len(words_in_sentence)):
        if words_in_sentence[index] not in stop_words:
            filtered_words.append(words_in_sentence[index])
    remove_stop_sentence = ' '.join(filtered_words[index] for index in range(len(filtered_words)))
    return remove_stop_sentence


def stemming_words(sentence):
    ps = PorterStemmer()
    words_in_sentence = sentence.split(" ")
    getting_stem_words = list()
    for index in range(len(words_in_sentence)):
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
    # Url pattern Source: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    url_pattern = r'^https?:\/\/.*[\r\n]*'
    illegal_character_pattern = r'[^#@_$%\sa-zA-Z\d]'
    result_sentence = list()
    for index in range(len(raw_sentence)):
        result_sentence.append(re.sub(illegal_character_pattern, '', re.sub(url_pattern, ' ', raw_sentence[index])))
    return result_sentence

# deleting extra illegal info
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
# Question 2 change the maximum feature
maximum_feature = 1000
String_pattern = r'[#@_$%\w\d]{2,}'
# Question 5 change the lowercase of sentence
Low = False
count = CountVectorizer(token_pattern=String_pattern, lowercase=Low)
# Line 46 and 49 Source:  https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
X_training_bag_of_words = count.fit_transform(legal_training_sentence)
X_testing_bag_of_words = count.transform(legal_testing_sentence)

# This condition is from the question(the pdf from ass2)
# which is (1% of the training set), therefore getting the minimum number leaf
# criterion is entropy
# Line 62-65 Source:  https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
len_training_sentence = len(training_sentence)
minimum_leaf = int(0.01 * len_training_sentence)
ceriterion_condition = 'entropy'
clf = tree.DecisionTreeClassifier(min_samples_leaf=minimum_leaf, criterion=ceriterion_condition, random_state=0)
model = clf.fit(X_training_bag_of_words, training_result)
predict_result = model.predict(X_testing_bag_of_words)

#######################################################################################################################
# Printing the classification report
# Using the function in Sklearn to print the result of classification
# This is based on the code in fuction predict_and_test in example.py line 15
# Line 15 Source: https://www.cse.unsw.edu.au/~cs9414/assignments/example.py
######################################################################################################################
# print(classification_report(testing_result, predict_result))

# for i in range(len(testing_sentence)):
#     print(testing_id[i],predict_result[i])
