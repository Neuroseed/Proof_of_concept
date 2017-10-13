import time
import os

import numpy

import keras
from keras import layers
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import newsgroup20


EMBEDDING_DIM = 128
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 80
VALIDATION_SPLIT = 0.2
TEXT_DATA_DIR = '20newsgroup'
GLOVE_DIR = 'glove'


def create_model():
    sequence_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = layers.Embedding(len(newsgroup20.word_index) + 1,
        EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True)
    embedded_sequences = embedding_layer(sequence_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = layers.MaxPooling1D(5)(x)
    #x = layers.Conv1D(128, 5, activation='relu')(x)
    #x = layers.MaxPooling1D(5)(x)
    #x = layers.Conv1D(128, 5, activation='relu')(x)
    #x = layers.MaxPooling1D(35)(x)  # global max pooling
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(len(newsgroup20.labels_index), activation='softmax')(x)

    model = models.Model(sequence_input, out)

    return model


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['acc'])


def train_model(model, x_train, y_train, x_test, y_test):
    t = time.time()

    model.fit(x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=1,
        batch_size=128)

    print('t =', time.time() - t)


def test_model(model):
    score, acc = model.evaluate(x_test, y_test)

    return score, acc


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = newsgroup20.get_dataset()

    model = create_model()

    compile_model(model)
    train_model(model, x_train, y_train, x_test, y_test)

    test_model(model)

