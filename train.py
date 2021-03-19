'''train.py

CS 6375.003 Final Project: train.py
Authors: Usuma Thet, Merissa Fulfer, Rishi Dandu, Yuncheng Gao

    python train.py

'''
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gensim
import re
import numpy as np
import sys
import os
from collections import Counter
import logging
import time
import pickle
import itertools

def main(argv):
    if len(argv) != 1:
        raise Exception("Incorrect number of arguments")
    DATASET_NAME = argv[0]
    
    cols = ["target", "ids", "date", "flag", "user", "text"]
    encoding = "ISO-8859-1"
    TRAIN_SIZE = 0.8

    PREPROCESSING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    SEQUENCE_LENGTH = 20
    EPOCHS = 32
    BATCH = 1024
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

    df = pd.read_csv(DATASET_NAME, encoding = encoding , names= cols)

    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    df.target = df.target.apply(lambda x: decode_map[int(x)])

    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")

    def preprocess(text, stem=False):
        text = re.sub(PREPROCESSING_RE, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        # append the tokens with a space between each other
        return " ".join(tokens)

    df.text = df.text.apply(lambda x: preprocess(x))

    TRAININGSET, TESTINGSET = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=1)
    print("TRAIN size:", len(TRAININGSET))
    print("TEST size:", len(TESTINGSET))

    # Preprocessing
    documents = [_text.split() for _text in TRAININGSET.text] 
    w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                                window=7, 
                                                min_count=10, 
                                                workers=8)

    w2v_model.build_vocab(documents)

    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)

    w2v_model.train(documents, total_examples=len(documents), epochs=32)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(TRAININGSET.text)

    vocab_size = len(tokenizer.word_index) + 1

    x_train = pad_sequences(tokenizer.texts_to_sequences(TRAININGSET.text), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(TESTINGSET.text), maxlen=SEQUENCE_LENGTH)

    labels = TRAININGSET.target.unique().tolist()
    labels.append(NEUTRAL)

    encoder = LabelEncoder()
    encoder.fit(TRAININGSET.target.tolist())

    y_train = encoder.transform(TRAININGSET.target.tolist())
    y_test = encoder.transform(TESTINGSET.target.tolist())

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    embedding_matrix = np.zeros((vocab_size, wordtovec_size))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        print(embedding_matrix.shape)

    embedding_layer = Embedding(vocab_size, wordtovec_size, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)

    # Creating the model
    model = Sequential()
    # Added the embedding layer
    model.add(embedding_layer)
    # One drop layer to reduce overfitting
    model.add(Dropout(0.6))
    # 100 LSTM Layer
    model.add(LSTM(200, dropout=0.1, recurrent_dropout=0.2))
    # Final layer for output
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    callbackFunction = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0), EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
    print("Model compiled")
    his = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS, validation_split=0.1, verbose=1, callbacks=callbackFunction)
    print("Model finished training")
    score = model.evaluate(x_test, y_test, batch_size=BATCH)
    print("Accuracy:",score[1])
    print("Loss:",score[0])

    model.save("model.h5")

if __name__ == "__main__":
    main(sys.argv[1:])