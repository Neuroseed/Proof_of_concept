import time
import numpy
import keras
from keras import layers
from keras import models

import newsgroup20


N_PARTS = 2
MAX_SEQUENCE_LENGTH = 80


def load_submodel(part):
    model = models.load_model('slice_models/newsgroup20_%s_%s.h5' % (N_PARTS, part))

    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    return model


def create_model():
    input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    submodel0 = load_submodel(0)
    submodel1 = load_submodel(1)

    submodel0_layer = submodel0(input)
    submodel1_layer = submodel1(input)

    output = layers.concatenate([submodel0_layer, submodel1_layer])

    #output = layers.Dense(20)(conc)

    model = models.Model(input, output)

    model.summary()

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


def test_model(model, x_test, y_test):
    score, acc = model.evaluate(x_test, y_test)

    return score, acc


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = newsgroup20.get_dataset()

    model = create_model()

    compile_model(model)
    train_model(model, x_train, y_train, x_test, y_test)

    score, acc = test_model(model, x_test, y_test)

