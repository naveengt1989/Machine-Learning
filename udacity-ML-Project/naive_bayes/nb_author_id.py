#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
# Gaussian NB -> select label corresponding to maximum P(Label|X) 
# P(Label|X) = P(X|Label).P(Label)/P(X) 
# P(Label) calculated from training data 
# P(X|Label) is calculated from pdf function of (X) for label using mean,variance of X for label calculated from training data.
# As it is naive, P(X|Label) = P(X1|Label)*P(X2|Label)....*P(Xn|Label) where n is number of features
# P(X) = P(X|Label(i))*P(Label(i)) + P(X|Label(j))*P(Label(j))
# https://www.youtube.com/watch?v=r1in0YNetG8
clf = GaussianNB()
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)
print accuracy_score(labels_test, y_pred)
#########################################################


