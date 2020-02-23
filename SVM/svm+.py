#!/usr/bin/env python3
from sklearn.linear_model import LogisticRegression
import re
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsOneClassifier
import sys, time
from collections import defaultdict
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
ngmin = 1 
ngmax = 5
def char_tokenizer(s):
    return list(s)
f = open('train.csv', '+r')
f2 = open('dev.csv', '+r')
f3 = open('train1.csv', 'w')
f4 = open('dev1.csv', 'w')
f5 = open('f5.csv', 'w')
doc_train = []
y_train  = []
y_test  = []
doc_test  = []
ngmin = 1 
ngmax = 5
id = []
k = ""
k1 = ""
freq = []
freq1 = []
for i, line in enumerate(f):
    line=line.replace('\n','').replace(' ,"','').replace('"','').replace('\'','').replace(' , ',' ')
    line = re.sub(' \d+,\d+','',line)
    words = line.split(",")
    words1 = [words[1],words[4]]
    words2 = re.sub('<[\S*\s*]*/>','%s' %(words[2]), words[1])
    words1[0] = words2
    y_train.append(words1[1])
    list_of_words = words2.split()
    if (i == 0):
       k = ""
    else:
      if ((len(list_of_words) > 1) and  ((len(list_of_words)-1) > list_of_words.index(words[2])) ):
        next_word = list_of_words[list_of_words.index(words[2]) + 1]
        sn = words[2] + "\t" +  next_word 
        if sn in open('ngrams_words_2.txt').read():
          k = "1"
        else: 
         k = "0"
      else: 
         k = "0"
    if (i == 0):
       k1 = ""
    else:
      if ((len(list_of_words) > 1) and  ( list_of_words.index(words[2]) > 0 ) ):
        prev_word = list_of_words[list_of_words.index(words[2]) - 1]
        pr =  prev_word  + "\t" + words[2]  
        if sn in open('ngrams_words_2.txt').read():
          k1 = "1"
        else: 
          k1 = "0"
      else: 
       k1 = "0"
    doc_train.append(words1[0] + k + k1)
    f3.write(words1[0] + k + k1 + "," + words1[1] + "\n")
print(doc_train, y_train)   
for i, line in enumerate(f2):
    line=line.replace('\n','').replace('"','').replace('\'','').replace(' , ',' ')
    line = re.sub(' \d+,\d','',line)
    words = line.split(",")
    words2 = re.sub('<[\S*\s*]*/>','%s' %(words[2]), words[1])
    list_of_words = words2.split()
    if (i == 0):
       k = ""
    else:
      if ((len(list_of_words) > 1) and  ((len(list_of_words)-1) > list_of_words.index(words[2])) ):
        next_word = list_of_words[list_of_words.index(words[2]) + 1]
        sn = words[2] + "\t" +  next_word 
        if sn in open('ngrams_words_2.txt').read():
          k = "1"
        else: 
         k = "0"
      else: 
         k = "0"
    if (i == 0):
       k1 = ""
    else:
      if ((len(list_of_words) > 1) and  ( list_of_words.index(words[2]) > 0 ) ):
        prev_word = list_of_words[list_of_words.index(words[2]) - 1]
        pr =  prev_word  + "\t" + words[2]  
        if sn in open('ngrams_words_2.txt').read():
          k1 = "1"
        else: 
          k1 = "0"
      else: 
       k1 = "0"
    doc_test.append(words2 + k + k1)
    id.append(words[0])
    f4.write(words2  + k + k1 + "\n")
print(doc_test)
v = CountVectorizer(tokenizer=char_tokenizer, ngram_range=(ngmin,ngmax))
v.fit(doc_train + doc_test)
x_train = v.transform(doc_train)
x_test = v.transform(doc_test)

print("Training... ", end="")
m = LogisticRegression(solver='lbfgs', max_iter=5500)
m =  OneVsRestClassifier(m)
m.fit(x_train, y_train)

pred = m.predict(x_test)

for lab in pred:
    y_test.append(lab)
    
for a, b in zip(id, y_test):
    print (a + "," + b)

for a, b in zip(id, y_test):
    f5.write(a + "," + b + "\n")


