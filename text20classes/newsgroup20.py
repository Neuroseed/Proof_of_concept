import os

import numpy

import keras
from keras import layers
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


EMBEDDING_DIM = 128
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 80
VALIDATION_SPLIT = 0.2
TEXT_DATA_DIR = '20newsgroup'
GLOVE_DIR = 'glove'


def get_dataset():
    global word_index, labels_index

    texts = []
    labels_index = {}
    labels = []

    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    with open(fpath, encoding='latin-1') as f:
                        t = f.read()
                    i = t.find('\n\n')
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))


    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = keras.utils.to_categorical(numpy.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_test = data[-nb_validation_samples:]
    y_test = labels[-nb_validation_samples:]

    return (x_train, y_train), (x_test, y_test)


def slice_dataset(x, y, partn=2, classn=None):
    "Разрезает датасет на partn частей"

    if not classn:
        classn = y.shape[1]

    step = float(classn) / partn

    xa = []
    ya = []

    for i in range(partn):
        down = round(step * i)
        top = round(step * (i + 1))
        cond = numpy.any(y[:, down:top], axis=1)

        xa.append(x[cond])
        ya.append(y[cond, down:top])

    return (xa, ya)


def prepare_embedded():
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    embedding_layer = layers.Embedding(len(word_index) + 1,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)

