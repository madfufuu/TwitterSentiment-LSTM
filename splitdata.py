'''splitdata.py

CS 6375.003 Final Project: splitdata.py
Authors: Usuma Thet, Merissa Fulfer, Rishi Dandu, Yuncheng Gao

    python splitdata.py training_size sequence_length dataset.csv

    training_size should be an integer specifying the training size of the model
    sequence_length should be the hyperparameter used for training
    dataset.csv is the full dataset

'''

import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras import utils
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from numpy import savetxt
import gensim
import re
import numpy as np
import os
import logging
import time
import pickle
import itertools
import sys

nltk.download('stopwords')
# Get filenames from arguments
def main(argv):
    if len(argv) != 3:
        raise Exception("Incorrect number of arguments")
    TRAIN_SIZE = float(argv[0])
    SEQUENCE_LENGTH = int(argv[1])
    DATASET_NAME = argv[2]
    
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"

    # Preprocessing of tweets
    # Removes all @ signs and urls
    PREPROCESSING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 32
    W2V_MIN_COUNT = 10

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

    df = pd.read_csv(DATASET_NAME, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

    labels = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

    def decode(label):
        return labels[int(label)]

    df.target = df.target.apply(lambda x: decode(x))

    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")

    def preprocess(text, stem=False):
        text = re.sub(PREPROCESSING_RE, ' ', str(text).lower())
        text = text.strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
                    
        return " ".join(tokens)

    df.text = df.text.apply(lambda x: preprocess(x))

    df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
    print("Training size:", len(df_train))
    print("Testing size:", len(df_test))

    documents = [_text.split() for _text in df_train.text] 

    wordToVecModel = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)

    wordToVecModel.build_vocab(documents)

    words = wordToVecModel.wv.vocab.keys()
    vocab_size = len(words)

    wordToVecModel.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.text)

    vocab_size = len(tokenizer.word_index) + 1

    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)

    labels = df_train.target.unique().tolist()
    labels.append(NEUTRAL)

    encoder = LabelEncoder()
    encoder.fit(df_train.target.tolist())

    y_train = encoder.transform(df_train.target.tolist())
    y_test = encoder.transform(df_test.target.tolist())

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    df_test.to_csv("testSet.csv")
    df_train.to_csv("trainSet.csv")
    savetxt('x_test.csv', x_test, delimiter=',')

if __name__ == "__main__":
    main(sys.argv[1:])