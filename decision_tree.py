# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:47:52 2019

@author: Vedika Bansal
"""

import numpy as np
import nltk
import math
import random

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import os 
import csv
import xlsxwriter
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pathlib import Path
from scipy.sparse.csr import csr_matrix # to save tdidf  matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import re

lemmatizer = WordNetLemmatizer()            
words = []
contentsList = []
lem_contents = []
lem_pruned_contents = []
category = []
data = []       
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
lemm_all_words_list = []
contents = ''
for filename in Path('bare').glob('**/*.txt'):
    with open(str(filename)) as f:
        #print(f)
        
        wordsAl = []                
        wordA = []
        for contents in f.readlines():
            for i in contents.split():
                # removal of punctuations
                if i in punctuations: 
                    contents = contents.replace(i,"")
                if i not in stopwords.words('english'):
                    if i.isalpha():
                        w = []
                        w.append(i)
                        wordsAl.append(' '.join(w)) 
        #lemmatizing 
        wordsA = [lemmatizer.lemmatize(w) for w in wordsAl]
        
        countDict = Counter(wordsA) 
        
    str1 = ""    
    # actual emails' contents list of all emails
    contentsList.append(contents)
    # email contents collected after lemmatizing
    w = []
    words_lem_spaced = []
    for i in wordsA:
        wor = []
        i = i + ' '
        w.append(i)
        words_lem_spaced.append(' '.join(w))
    lem_contents.append(str1.join(words_lem_spaced))
    lemm_all_words_list.append(wordA)    
    
    #data += word_tokenize(contents)     # full list / vocabulary
    if(str(filename).startswith("bare\\s")):
        category.append(0)
    else :
        category.append(1)

#pruning           
dictionary = {} 
listD = []
for i,j in countDict.items(): 
    if j>5 and j<500:
        listD.append(i) #pruned list of words in that email
        dictionary[i] = j         
print("dictionaries created")

allWords = list(set(data))          # full list / vocabulary

# earlier code in temp.py

# calculating tf-idf based vectors
tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', max_df=0.9, min_df=0.1)

#actual email contents
tfidf_data_unfiltered = tfidf_vectorizer.fit_transform(contentsList)

#lemmatized and pruned contents
tfidf_data = tfidf_vectorizer.fit_transform(lem_contents)


with open('Filtered_tfidfData.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow([tfidf_data])
csvfile.close()
# discretizing the continuous tf-idf based vectors into bins

discretizer = KBinsDiscretizer(n_bins=20, encode='onehot', strategy='uniform')

#unfiltered
dis_data1 = discretizer.fit_transform(tfidf_data_unfiltered.toarray())
#filtered
dis_data2 = discretizer.fit_transform(tfidf_data.toarray())

classifier = DecisionTreeClassifier()

X_train1, X_test1, y_train1, y_test1, indices_train, indices_test = train_test_split(dis_data1, category, range(len(category)), test_size=0.10)

classifier.fit(X_train1, y_train1)
y_pred1 = classifier.predict(X_test1)

X_train2, X_test2, y_train2, y_test2, indices_train, indices_test = train_test_split(dis_data2, category, range(len(category)), test_size=0.10)

#classifier = Pipeline([('vect', TfidfVectorizer()),('clf', DecisionTreeClassifier(max_depth= 2, random_state=0))])
classifier.fit(X_train2, y_train2)
y_pred2 = classifier.predict(X_test2)

#classifier = TfidfVectorizer(input = 'filename', vocabulary = dict3)

accuracy = accuracy_score(y_test1, y_pred1)
c_matrix = confusion_matrix(y_test1, y_pred1)
tn, fp, fn, tp = c_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)
with open('results_vocab1.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['','TN','FP','FN','TP', 'accuracy', 'precision', 'recall', 'f1_score'])
    writer.writerow(["1st vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])
        
csvFile.close()

accuracy = accuracy_score(y_test2, y_pred2)
c_matrix = confusion_matrix(y_test2, y_pred2)
tn, fp, fn, tp = c_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

with open('results_vocab2.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['','TN','FP','FN','TP', 'accuracy', 'precision', 'recall', 'f1_score'])
    writer.writerow(["2nd vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])
    
csvFile.close()
