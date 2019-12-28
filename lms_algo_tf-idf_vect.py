# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:25:15 2019

@author: VEDIKA BANSAL
"""
cwd = os.getcwd()
print(cwd)
import numpy as np
import nltk
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# get working directory

cwd = os.getcwd()
print(cwd)
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# file wise contents in a folder in bare

# ASSUMING CWD RETURNS FILE LOCATION TILL "C:\Users\USER NAME\bare" already set 
def get_path(filename):
    file_path = Path(filename).resolve()
    return file_path

file_path = get_path('bare')
def get_data_emailLabels(filename):
    words = []
    contentsList = []
    category = []
    data = []
    dataspam = []
    dataham = []
    
    # print(os.listdir(cwd + '/' + directory))
    directory = get_path(filename)
    for file in os.listdir(cwd + '/'  + directory):
        with open(cwd + '/'  + directory+ '/'+ file, "r") as ifile:
            if file.endswith(".txt"):
                # print(os.path.join(cwd, file))
                contents = ifile.read()
                for i in contents:
                    # removal of punctuations
                    if i in punctuations:
                        contents = contents.replace(i,"")
               
                contentsList.append(contents)
                data += word_tokenize(contents)
                # print(file)
                if(file.startswith("s")):
                    category.append(0)
                    #dataspam += word_tokenize(contents)
                else :
                    category.append(1)
                    #dataham += word_tokenize(contents)
    return data, contentsList, category  

# get the required 4 dictionaris 
name = os.path.dirname(os.path.abspath(bare))
name
def get_dictionary(wordsAll): 
    
    wordsAl = []
    wordsA = []

    list1 = list(set(wordsAll))
     
    wordsAl = []
    for i in wordsAll:
        if i not in stopwords.words('english'):
            if i.isalpha():
                wordsAl.append(i)
    list2 = list(set(wordsAl))  
        
    # lemmatization
        
    lemmatizer = WordNetLemmatizer()
    wordsA = [lemmatizer.lemmatize(w) for w in wordsAl]
    list3 = list(set(wordsA))  
    
    countDict = Counter(wordsA)
            
    # dictionary4 after frequency pruning
    
    list4 = []    
    for i,j in countDict.items(): 
        if j>5 and j<800:
            print(j)
            print(i)
            list4.append(i)
    list4.remove('aa')
    list4.remove('aaa') 
    list4.sort()
    numList = list(range(len(list4)))
    dict = dict(zip(list4, numList))                     
    dict = {i: list4[i] for i in range(0, len(list4))}
    
    return dict