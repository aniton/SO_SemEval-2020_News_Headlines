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
def char_tokenizer(s):
    return list(s)
f = open('task-1-output.csv', '+r')
f2 = open('test.csv', '+r')
d = {}
for i, line in enumerate(f):
    line=line.replace('\n','')
    words = line.split(",")
    d.update({words[0]: words[1]})   
for i, line in enumerate(f2):
        words = line.split(",")
        if words[0][0] in ["0","1","2","3","4","5","6","7","8","9"]:
           num = words[0].split("-")
           if  ((d.get(str(int(num[0])))) > (d.get(str(int(num[1]))))):
                      print(words[0] + "," + "1")
           elif  ((d.get(str(int(num[0])))) < (d.get(str(int(num[1]))))):
                      print(words[0] + "," + "2") 
           else:
                      print(words[0] + "," + "0")
