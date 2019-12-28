# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:02:45 2019

@author: HP
"""

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy import asarray
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def derivative(x):
    return x * (1.0 - x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


words = []
contentsList = []
lem_contents = []
lem_pruned_contents = []
labels = []
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
    if(str(filename).startswith("bare\\s")):
        labels.append([0, 1])
    else :
        labels.append([1, 0])    

tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', max_df=0.9, min_df=0.1)
tfidf_nn_data = tfidf_vectorizer.fit_transform(lem_contents)
discretizer = KBinsDiscretizer(n_bins=20, encode='onehot', strategy='uniform')
discrete_data = discretizer.fit_transform(tfidf_nn_data.toarray())
#labels = asarray(category)

X_train = []
X_test = []
y_train = []
y_test = []
X_train = dis_data[0:2000].toarray()
X_test = dis_data[2000:].toarray()
y_train = labels[0:2000]
y_test = labels[2000:]



X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(discrete_data, labels, range(len(category)), test_size=0.10)
X_train = X_train.toarray()
X_test = X_test.toarray()


dim1 = len(X_train[0])

# No of nodes in the hidden layer

dim2 = 1100
np.random.seed(1)

# 2 output neurons
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 2)) - 1

for j in range(500):
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0, weight0))
    layer_2 = sigmoid(np.dot(layer_1, weight1))

    layer_2_error = y_train - layer_2
    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)

    weight1 += 0.001*(layer_1.T.dot(layer_2_delta))
    weight0 += 0.001*(layer_0.T.dot(layer_1_delta))

layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0, weight0))
layer_2 = sigmoid(np.dot(layer_1, weight1))
known_labels = []
predicted_labels = []
for i in range(len(y_test)):
    if y_test[i][0] == 0 and y_test[i][1] == 1:
        known_labels.append("Spam")
    else:
        known_labels.append("Ham")
    if layer_2[i][0] < 0.1 and layer_2[i][1] > 0.9:
        predicted_labels.append("Spam")
    else:
        predicted_labels.append("Ham")

print("Results")
print((dim1 + dim2 + 2) / (len(X_train) + len(X_test)))
accuracy = accuracy_score(known_labels, predicted_labels)
c_matrix = confusion_matrix(known_labels, predicted_labels)
tn, fp, fn, tp = c_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

with open('results_part_a.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['m/n','TN','FP','FN','TP', 'accuracy', 'precision', 'recall', 'f1_score'])
    writer.writerow([(dim1 + dim2 + 2) / (len(X_train) + len(X_test)),tn, fp, fn, tp, accuracy, precision, recall, f1_score])
    
csvFile.close()
    

# 1 output neuron
layer_2_error = []
layer_2_delta = []
layer_1_error = []
layer_1_delta = []
dim1 = len(X_train[0])
dim2 = 1000
np.random.seed(1)
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 1)) - 1
for j in range(500):
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0, weight0))
    layer_2 = sigmoid(np.dot(layer_1, weight1))

    layer_2_error = y_train - layer_2

    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)

    weight1 += 0.001*(layer_1.T.dot(layer_2_delta))
    weight0 += 0.001*(layer_0.T.dot(layer_1_delta))

layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0, weight0))
layer_2 = sigmoid(np.dot(layer_1, weight1))
known_labels = []
predicted_labels = []

for i in range(len(layer_2)):
    if layer_2[i][0] >= 0.7:
        predicted_labels.append("Spam")
    elif layer_2[i][0] <= 0.3:
        predicted_labels.append("Ham")
    elif layer_2[i][0] == 0.5:
        if random.randrange(0, 1) >= 0.5:
            predicted_labels.append("Spam")
        else:
            predicted_labels.append("Ham")
    else:
        predicted_labels.append("Dont Know")

for i in range(len(y_test)):
    if y_test[i] == [1]:
        known_labels.append("Spam")
    else:
        known_labels.append("Ham")

print("Results")
print((dim1 + dim2 + 2) / (len(X_train) + len(X_test)))
accuracy = accuracy_score(known_labels, predicted_labels)
c_matrix = confusion_matrix(known_labels, predicted_labels)
tn, fp, fn, tp = c_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

with open('results_part_b.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['m/n','TN','FP','FN','TP', 'accuracy', 'precision', 'recall', 'f1_score'])
    writer.writerow([(dim1 + dim2 + 2) / (len(X_train) + len(X_test)),tn, fp, fn, tp, accuracy, precision, recall, f1_score])
    
csvFile.close()
