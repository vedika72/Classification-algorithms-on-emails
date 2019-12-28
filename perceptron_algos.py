import numpy as np
import pandas as pd
import random
import os 
import csv
from collections import Counter
from pathlib import Path

# generate vectors with separation based on first m dimensions and values dependent on a constant a, stored in csv

def gen_vectors (m, a):
    filename = 'data_d'+ str(m) + '.csv'
    x = ['class']
    for i in range(50):
        s = 'feature_' + str(i)
        x.append(s)
    with open(filename, 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(x)    
    # for class 1        
    for i in range(500):
        x = [1]
        
        for j in range(50):
            
            if(j<m):
                x.append(random.randint(-2*a, -1*a))
                print(j)
            else:
                x.append(random.randint(-2*a, 2*a))
        with open(filename, 'a', newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(x)
    # for class 2
    for i in range(500):
        x = [2]
        
        for j in range(50):
            
            if(j<m):
                x.append(random.randint(a, 2*a))
                print(j)
            else:
                x.append(random.randint(-2*a, 2*a))
        with open(filename, 'a', newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(x)

# to get full path of the required file

def get_path(filename):
    pathcsv = Path(filename).resolve()
    return pathcsv
            
a = 4        
gen_vectors(5, a)
gen_vectors(10, a)
gen_vectors(20, a)

# creating data frame from 5-dimensional separating features

filename = 'data_d5.csv'

pathcsv = Path(filename).resolve()
pathcsv

data = pd.read_csv(str(pathcsv))
data.head()

df = data.iloc[:, 1:6]
df.describe()

feature_names = df.columns
feat_names = list(feature_names)

# matrix x of first m-dimensions 
X = df[feat_names]
list(X)
Y = data['class']
list(Y)
Y = np.array(Y)

# creating data frame from 10-dimensional separating features

filename = 'data_d10.csv'

pathcsv = Path(filename).resolve()
pathcsv

data = pd.read_csv(str(pathcsv))
data.head()

df2 = data.iloc[:, 1:11]
df2.describe()

Y2 = data['class']
list(Y2)
Y2 = np.array(Y2)
'''___________________________________________________________________________________________________'''  

# batch perceptron with constant p(t)
def perceptron(w, p, labels, df, file):
    
    x = ['row no.', 'inner product', 'misclassified status', 'b']
    with open(file, 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(x) 
    
    w_new = []
    matrix_X = []
    for index, rows in df.iterrows(): 
        row = list(rows)
        matrix_X.append(row)
    
    size = len(w[0])
    a = np.zeros( size, dtype=int)
    unclassified = []

    for i in range(1000):
        x_vector = []
        x_vector = matrix_X[i]
        x_vector.append(1)
        x_vector = np.array(x_vector)[np.newaxis]
        #print(x_vector.shape)
        #print(np.inner(w, x_vector))
        
        #print(labels[i])
        # direction of w vector is opposite to label 1 x-vectors and therefore inner-product must've been -ve
        if(np.inner(w, x_vector) > 0 and labels[i] == 1):
            sign = 1
        elif(np.inner(w, x_vector) < 0 and labels[i] == 2):
            sign = -1
        else:
            sign = 0            
        b = sign * np.array(x_vector)
        
        if(sign!= 0):
            unclassified.append(b)
        #print(a)    
        #print('b:')
        #print(b)
        
        a = a+b
        if(sign!=0):
            x = [i, np.inner(w, x_vector), sign]
            x.append(b)
            with open(file, 'a', newline = '') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(x)         
    #print(w)
    w_new = w - p * a
    w_newr = np.around(w_new, 2)
    #print('w_newr:')
    #print(w_newr)
    
    return w_newr, unclassified
'''___________________________________________________________________________________________________'''
    
# creating a weight vector for 5-dimensions and the bias

w = [0.4 , -0.3 , -0.7,  0.9,-1, 7]
len(w)
w = np.array(w)[np.newaxis]
w.shape
len(w[0])
# setting a constant value of the parameter
p = 0.03
# perceptron(w,p,Y)
unclassified = [1,2,3]
len(unclassified)
t = 0
x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open('perceptron.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)
    
while (len(unclassified) != 0):
    unclassified = []
    w_new, unclassified = perceptron(w, p, Y, df, 'perceptron.csv')
    print("w:")
    print(w)
    print("w_new:")
    print(w_new)
    with open('perceptron.csv', 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['no of misclassifications:' , len(unclassified)])
        writer.writerow(['old_w', w])
        writer.writerow(['new_w', w_new])
        writer.writerow(['iteration no', t])
        writer.writerow([])
    if(len(unclassified)==0):
        break
    else:
        w = w_new
        t += 1

t

# creating a weight vector for 10-dimensions and the bias

w = [0.4 , -0.3 , -0.7,  0.9, -1, -1 , -3, 0.6, 0.02, 0.004, 5]
w = np.array(w)[np.newaxis]
w.shape
# setting a constant value of the parameter
p = 0.0007
# perceptron(w,p,Y)
x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open('perceptron_d10.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)
unclassified = [1,2,3]
len(unclassified)
t = 0
while (len(unclassified) != 0):
    unclassified = []
    w_new, unclassified = perceptron(w, p, Y2, df2, 'perceptron_d10.csv')
    print("w:")
    print(w)
    print("w_new:")
    print(w_new)
    with open('perceptron_d10.csv', 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['no of misclassifications:' , len(unclassified)])
        writer.writerow(['old_w', w])
        writer.writerow(['new_w', w_new])
        writer.writerow(['iteration no', t])
        writer.writerow([])
    if(len(unclassified)==0):
        break
    else:
        w = w_new
        t += 1

t

'''___________________________________________________________________________________________________'''
# batch perceptron in case variable p(t)

w = [0.4 , -0.3 , -0.7,  0.9,-1, 7]
w = np.array(w)[np.newaxis]
w.shape
c = 0.2
e = 0.4
t = 0

unclassified = [1,2,3]
len(unclassified)
x1 = ['weight vector:']
x2 = w[0]
x4 = ['no. f separating features:' , len(w[0])]
filename = 'batch_perceptron.csv'
with open(filename, 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x4)
while (len(unclassified) != 0):
    unclassified = []
    p = c / (t + e)
    print(p)
    w_new, unclassified = perceptron(w, p, Y, df, filename)
    print("w:")
    print(w)
    print("w_new:")
    print(w_new)
    with open(filename, 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['parameter value:' , p])
        writer.writerow(['no of misclassifications:' , len(unclassified)])
        writer.writerow(['old_w', w])
        writer.writerow(['new_w', w_new])
        writer.writerow(['iteration no', t])
        writer.writerow([])    
    if(len(unclassified)==0):
        break
    else:
        w = w_new
        t += 1

t

# equation of hyperplane 
#w = str(w)
#eq = 'trans(x1, x2, ---, xd, 1)'+ w

'''___________________________________________________________________________________________________'''

# pocket algorithm for non-separable

def gen_nonseparable_data (m, a):
    filename = 'data_incorrect50_d'+ str(m) + '.csv'
    x = ['class']
    for i in range(50):
        s = 'feature_' + str(i)
        x.append(s)
    with open(filename, 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(x)            
    for i in range(500):
        x = [1]
        
        for j in range(50):
            # generating feature values for class1 using class2 generator function
            if(i<50 and j<m):
                x.append(random.randint(a, 2*a))
            elif(i>50 and j<m):
                x.append(random.randint(-2*a, -1*a))
                print(j)
            else:
                x.append(random.randint(-2*a, 2*a))
        with open(filename, 'a', newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(x)
            
    for i in range(500):
        x = [2]
        
        for j in range(50):
            # generating feature values for class2 using class1 generator function            
            if(i<50 and j<m):
                x.append(random.randint(-2*a, -1*a))
            elif(i>50 and j<m):            
                x.append(random.randint(a, 2*a))
                print(j)
            else:
                x.append(random.randint(-2*a, 2*a))
        with open(filename, 'a', newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(x)
    return filename
a = 4 
       
file = gen_nonseparable_data(5, a)

pathcsv = get_path(file)

data = pd.read_csv(str(pathcsv))
data.head()

df3 = data.iloc[:, 1:6]
df3.describe()
feature_names = df3.columns
feat_names = list(feature_names)

# matrix x of first m-dimensions 
X = df3[feat_names]
list(X)
Y3 = data['class']
list(Y3)
Y3 = np.array(Y3)

# initialization
hs = 0
t = 0
w = [0.4 , -0.3 , -0.7,  0.9,-1, 7]
w = np.array(w)[np.newaxis]
p = 0.18

filename = 'unseparable_case.csv'
x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open(filename, 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)
while(t<10):
    w_new, unclassified = perceptron(w, p, Y3, df3, 'dont_need.csv')
    # h is no of correct classifications
    h = 1000 - len(unclassified)
    print(h)
    print(t)
    # change the weight vector if only number of correct classifications increase
    if(h > hs):
        w = w_new
    t += 1
    print(w)
    with open(filename, 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['no of misclassifications:' , len(unclassified)])
        writer.writerow(['old_w', w])
        writer.writerow(['new_w', w_new])
        writer.writerow(['iteration no', t])
        writer.writerow([])    

# least mean squared algorithm

def lms_algo(w, p, df, labels):
    
    w_new = []
    matrix_X = []
    x_vector = df.sample
    
    for index, rows in df.iterrows(): 
        row = list(rows)
        matrix_X.append(row) 
    
    i = random.randint(0,999)
    x_vector = []
    x_vector = matrix_X[i]
    x_vector.append(1)
    x_vector = np.array(x_vector)[np.newaxis]
    y = labels[i]
    print(x_vector)
    print(x_vector.shape)
    print(y)

    
        #print(np.inner(w, x_vector))
    
    if(y==2):
        y = 1 
    else:
        y = -1

    print(y)
    e = (y - np.inner(w, x_vector))
           
    print("error")
    print(e)

    w_new = w + p * e * x_vector
    w_newr = np.around(w_new, 2)
    #print('w_newr:')
    #print(w_newr)
    
    return w_newr, e

# creating data frame from 5-dimensional separating features

filename = 'data_d5.csv'

pathcsv = Path(filename).resolve()

data = pd.read_csv(str(pathcsv))

df = data.iloc[:, 1:6]
Y = data['class']
list(Y)
Y = np.array(Y)    
# creating a weight vector for m-dimensions

w = [0.4 , -0.3 , -0.7,  0.9,-1, 3]
w = np.array(w)[np.newaxis]

# setting a constant value of the parameter
p = 0.0005

t = 0

x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open('lms_s.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)

while (t<100):
    unclassified = []
    w_new, error = lms_algo(w, p, df, Y)
    #print("w:")
    #print(w)
    #print("w_new:")
    #print(w_new)
    x1 = ['old_w']
    x2 = ['new_w']
    for i in w:
        x1 = x1.append(i)
    for i in w_new:
        x2 = x2.append(i)
    with open('lms_s.csv', 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['error:', error])
        writer.writerow(['iteration no:', t])
        writer.writerow(["w"])
        writer.writerow(w)
        writer.writerow(["w_new"])
        writer.writerow(w_new)
        
    w = w_new
    t += 1

t    

w, unclassified = perceptron(w, p, Y, df, 'lms_separable_2.csv')
with open('lms_separable_2.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['unclassified', len(unclassified)])

# creating data frame from 20-dimensional separating features

filename = 'data_d20.csv'

pathcsv = Path(filename).resolve()

data = pd.read_csv(str(pathcsv))

df20 = data.iloc[:, 1:21]
Y = data['class']
list(Y)
Y = np.array(Y)

w = [0.4 , -0.3 , -0.7,  0.9,-1, 0.4 , -0.3 , -0.7,  0.9,-1, 0.4 , -0.3 , -0.7,  0.9,-1, 0.4 , -0.3 , -0.7,  0.9,-1, 3]
w = np.array(w)[np.newaxis]
w
# setting a constant value of the parameter
p = 0.0005

t = 0

x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open('lms_d20.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)

while (t<100):
    unclassified = []
    w_new, error = lms_algo(w, p, df20, Y)
    #print("w:")
    #print(w)
    #print("w_new:")
    #print(w_new)
    x1 = ['old_w']
    x2 = ['new_w']
    for i in w:
        x1 = x1.append(i)
    for i in w_new:
        x2 = x2.append(i)
    with open('lms_d20.csv', 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['error:', error])
        writer.writerow(['iteration no:', t])
        writer.writerow(["w"])
        writer.writerow(w)
        writer.writerow(["w_new"])
        writer.writerow(w_new)
        
    w = w_new
    t += 1

t    

w, unclassified = perceptron(w, p, Y, df20, 'lms_separable_d20.csv')
with open('lms_separable_d20.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['unclassified', len(unclassified)])

# creating dataframe for unseparable vectors from 5-dimensional separating features

filename = 'data_incorrect50_d5.csv'

pathcsv = Path(filename).resolve()

data = pd.read_csv(str(pathcsv))

df = data.iloc[:, 1:6]
Y = data['class']
list(Y)
Y = np.array(Y)    
# creating a weight vector for m-dimensions

w = [0.4 , -0.3 , -0.7,  0.9,-1, 4]
w = np.array(w)[np.newaxis]

# setting a constant value of the parameter
p = 0.0008

t = 0

x1 = ['weight vector:']
x2 = w[0]
x3 = ['parameter value:' , p]
x4 = ['no. f separating features:' , len(w[0])]
with open('lms_u.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(x1)
    writer.writerow(x2)
    writer.writerow(x3)
    writer.writerow(x4)
    
while (t<100):
    unclassified = []
    w_new, error = lms_algo(w, p, df, Y)
    #print("w:")
    #print(w)
    #print("w_new:")
    #print(w_new)
    x1 = ['old_w']
    x2 = ['new_w']
    for i in w:
        x1 = x1.append(i)
    for i in w_new:
        x2 = x2.append(i)
    with open('lms_u.csv', 'a', newline = '') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([])
        writer.writerow(['error:', error])
        writer.writerow(['iteration no:', t])
        writer.writerow(["w"])
        writer.writerow(w)
        writer.writerow(["w_new"])
        writer.writerow(w_new)
        
    w = w_new
    t += 1

t    

w, unclassified = perceptron(w, p, Y, df, 'lms_unseparable.csv')
with open('lms_unseparable.csv', 'a', newline = '') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['unclassified', len(unclassified)])