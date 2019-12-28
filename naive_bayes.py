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

# ASSUMING CWD RETURNS FILE LOCATION TILL BARE

def get_data_emailLabels(directory):
    words = []
    contentsList = []
    category = []
    data = []
    dataspam = []
    dataham = []
    # print(os.listdir(cwd + '/' + directory))
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

def get_dictionaries(wordsAll): 
    
    wordsAl = []
    wordsA = []
    d1 = {}
    d2 = {}
    d3 = {}
    d4 = {}
    list1 = list(set(wordsAll))
    list1.sort()
    
    # dictionary1 of raw data
    
    d1 = {i: list1[i] for i in range(0, len(list1))}
     
    wordsAl = []
    for i in wordsAll:
        if i not in stopwords.words('english'):
            if i.isalpha():
                wordsAl.append(i)
    list2 = list(set(wordsAl))  
    list2.sort()        
        
    # dictionary without stop words and digits
        
    d2 = {i: list2[i] for i in range(0, len(list2))}
        
    # lemmatization
        
    lemmatizer = WordNetLemmatizer()
    wordsA = [lemmatizer.lemmatize(w) for w in wordsAl]
    list3 = list(set(wordsA))  
    list3.sort()
    
    # dictionary3 of lemmatized words
    
    d3 = {i: list3[i] for i in range(0, len(list3))}
    
    countDict = Counter(wordsA)
    #print(countDict)
        
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
    d4 = dict(zip(list4, numList))                     
    d4 = {i: list4[i] for i in range(0, len(list4))}
    
    return list1, list2, list3, list4

# get n x d array for d-dimentional vectors of n emails 

def getArray(dirList, dictionary):
    
    cv = CountVectorizer(input = 'filename', vocabulary = dictionary)
    #print(dirList)
    
    fileList = []
    for directory in dirList:   
        p = os.path.join(os.getcwd(),directory)        
        for i in os.listdir(p):
            pathF = os.path.join(p, i)
            fileList.append(pathF)
    
    #print(fileList)

    X = cv.fit_transform(fileList)
    cv.get_feature_names()

    return X.toarray()
        
# list to store all the 10 directory names

dirList = os.listdir(cwd)
#print(dirList)  
wordsAll = []
emailsAll = []
labelsAll = []

# fetching words, emails with their labels 1 for ham and 0 for spam from 1st 9 folders

for i in range(10):
    print(i)    
    words, emails, labels = get_data_emailLabels(dirList[i])
    wordsAll += words
    emailsAll.append(emails)
    labelsAll.append(labels)

list1, list2, list3, list4 = get_dictionaries(wordsAll)    

# 10 fold cross validation 

vocab1 = list1
vocab2 = list2
vocab3 = list3
vocab4 = list4
acc1 = []
prec1 = []
rec1 = []
f1 = []
acc2 = []
prec2 = []
rec2 = []
f2 = []
acc3 = []
prec3 = []
rec3 = []
f3 = []
acc4 = []
prec4 = []
rec4 = []
f4 = []  
    
with open('results.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Vocabulary','TN','FP','FN','TP', 'accuracy', 'precision', 'recall', 'f1_score'])
    accList = []
    c = 0

    for i in range(10):
        trainList = []
        y_train = []
        print("Validation phase:")
        print(i)
        for j in range(10):
            if j != i :
                y_train += labelsAll[j]
                trainList.append(dirList[j])
        
        y_train = np.array(y_train)
        model = MultinomialNB()
        y_test = labelsAll[i]
        y_test = np.array(y_test)
        
        print("Training on vocabulary 1 with no filtering:")
        
        X_train = getArray(trainList, vocab1)
                
        model.fit(X_train, y_train)  
        
        X_test = getArray([dirList[i]], vocab1)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
            
        c_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = c_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        acc1.append(accuracy)
        prec1.append(precision)
        rec1.append(recall)
        f1.append(f1_score)
        writer.writerow([str(i+1)+"th fold of Validation Process:"])
        writer.writerow(["Unfiltered Vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])
             
        print("Training on vocabulary 2 with stop-words removed:")
        
        X_train = getArray(trainList, vocab2)
                
        model.fit(X_train, y_train)  
        
        X_test = getArray([dirList[i]], vocab2)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        c_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = c_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        acc2.append(accuracy)
        prec2.append(precision)
        rec2.append(recall)
        f2.append(f1_score)

        writer.writerow(["Stop-words removed vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])
        
        print("Stop-words removed vocabulary:")
        
        X_train = getArray(trainList, vocab3)
                
        model.fit(X_train, y_train)  
        
        X_test = getArray([dirList[i]], vocab3)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        c_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = c_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)    

        acc3.append(accuracy)
        prec3.append(precision)
        rec3.append(recall)
        f3.append(f1_score)

        writer.writerow(["Lemmatized Vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])

        print("Training on filtered vocabulary 4:")
        
        X_train = getArray(trainList, vocab4)
                
        model.fit(X_train, y_train)  
        
        X_test = getArray([dirList[i]], vocab4)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        c_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = c_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        acc4.append(accuracy)
        prec4.append(precision)
        rec4.append(recall)
        f4.append(f1_score)

        writer.writerow(["Pruned Vocabulary:",tn, fp, fn, tp, accuracy, precision, recall, f1_score])

    writer.writerow(['Vocabulary', 'Average Accuracy', 'Average Precision', 'Average Recall', 'Average f1_score'])   
    
    writer.writerow(['Vocabulary 1',np.average(acc1), np.average(prec1), np.average(rec1), np.average(f1)])
    writer.writerow(['Vocabulary 2',np.average(acc2), np.average(prec2), np.average(rec2), np.average(f2)])
    writer.writerow(['Vocabulary 3',np.average(acc3), np.average(prec3), np.average(rec3), np.average(f3)])
    writer.writerow(['Vocabulary 4',np.average(acc4), np.average(prec4), np.average(rec4), np.average(f4)])

csvFile.close()
