#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:54:11 2019

@author: aniton
"""
from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
from deeppavlov.models.preprocessors.str_lower import str_lower
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
import json
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
import numpy as np
from deeppavlov.models.embedders.tfidf_weighted_embedder import TfidfWeightedEmbedder
import tensorflow as tf
from deeppavlov.models.sklearn import SklearnComponent
import tensorflow_hub as hub
from deeppavlov.core.data.utils import download
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from deeppavlov.metrics.accuracy import sets_accuracy
from deeppavlov import configs
from deeppavlov.models.classifiers.proba2labels import Proba2Labels
from keras.layers import Input, Dense, Activation, Dropout, Flatten, GlobalMaxPooling1D
from keras import Model
from deeppavlov.models.preprocessors.one_hotter import OneHotter
from deeppavlov.models.classifiers.keras_classification_model import KerasClassificationModel
from deeppavlov.metrics.accuracy import sets_accuracy
#config_path = configs.classifiers.rusentiment_elmo_twitter_cnn
#print(type(config_path), config_path)

#with open(config_path, "r") as f:
    #config = json.load(f)

#print(json.dumps(config, indent=2))


#print(json.dumps(config["dataset_reader"], indent=3))
#from deeppavlov.core.data.simple_vocab import SimpleVocabulary
#from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
#from deeppavlov.vocabs.typos import RussianWordsVocab
f5 = open('f5.csv', '+w')
dr = BasicClassificationDatasetReader().read(
    data_path='./',
    train='train.csv',
    valid='valid.csv',
    test='test.csv',
    x = 'original',
    y = 'meanGrade'
)
train_iterator = BasicClassificationDatasetIterator(
        data=dr, seed=42) 
x_train, y_train = train_iterator.get_instances(data_type='train')
for x, y in list(zip(x_train, y_train))[:5]:
    print('x:', x)
    print('y:', y)
    print('=================')
tokenizer = NLTKMosesTokenizer()
train_x_lower_tokenized = str_lower(tokenizer(train_iterator.get_instances(data_type='train')[0]))
classes_vocab = SimpleVocabulary(
    save_path='./snips/classes.dict',
    load_path='./snips/classes.dict')
vocab = SimpleVocabulary(save_path="./binary_classes.dict")
classes_vocab.fit((train_iterator.get_instances(data_type='train')[1]))
classes_vocab.save()
token_vocab = SimpleVocabulary(
   save_path='./snips/tokens.dict',
   load_path='./snips/tokens.dict',
   min_freq=2,
   special_tokens=('<PAD>', '<UNK>',),
   unk_token='<UNK>')
token_vocab.fit(train_x_lower_tokenized)
token_vocab.save()
token_vocab.freqs.most_common()[:10]
tfidf = SklearnComponent(
    model_class="sklearn.feature_extraction.text:TfidfVectorizer",
    infer_method="transform",
    save_path='./tfidf_v0.pkl',
    load_path='./tfidf_v0.pkl',
    mode='train')
tfidf.fit(str_lower(train_iterator.get_instances(data_type='train')[0]))
tfidf.save()
embedder = GloVeEmbedder("http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip",
                         dim=100, pad_zero=True)
weighted_embedder = TfidfWeightedEmbedder(
    embedder=embedder,  # our GloVe embedder instance
    tokenizer=tokenizer,  # our tokenizer instance
    mean=True,  # to return one vector per sample
    vectorizer=tfidf  # our TF-IDF vectorizer
)
x_train, y_train = train_iterator.get_instances(data_type="train")
x_valid, y_valid = train_iterator.get_instances(data_type="test")
cls = SklearnComponent(
    model_class="sklearn.linear_model:LogisticRegression",
    infer_method="predict",
    save_path='./logreg_v0.pkl',
    load_path='./logreg_v0.pkl',
    C=1,
    mode='train')
cls.fit(tfidf(x_train), y_train)
cls.save()
y_valid_pred = cls(tfidf(x_valid))
print("Text sample: {}".format(x_valid[0]))
print("True label: {}".format(y_valid[0]))
print("Predicted label: {}".format(y_valid_pred[0]))
for a in y_valid_pred:
    f5.write(a + "\n")
    print(a + "\n")
