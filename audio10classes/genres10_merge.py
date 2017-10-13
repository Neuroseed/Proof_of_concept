import os
import time
import glob

from keras import models
from keras import layers
from keras import optimizers
from keras import utils

import genres10


N_PARTS = 3
num_classes = 10
epochs = 2


def load_submodel(part):
    model = models.load_model('submodels/model_%s_%s.h5' % (N_PARTS, part))

    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    return model


def get_submodels():
    submodels = []
    for part in range(N_PARTS):
        submodels.append(load_submodel(part))

    return submodels


def create_model(shape):
    input = layers.Input(shape=shape)

    submodels = get_submodels()
    submodels_layers = [submodel(input) for submodel in submodels]

    conc = layers.concatenate(submodels_layers)

    #output = layers.Dense(num_classes)(conc)

    model = models.Model(input, conc)

    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    return model


def compile_model(model):
    # Compile the model
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])


def train_model(model, x_train, y_train):
    t = time.time()

    model.fit(x_train, y_train,
        epochs=epochs)

    print('t =', time.time() - t)


def test_model(model, x_test, y_test):
    score, acc = model.evaluate(x_test, y_test)

    print('Model acc', acc)

    return score, acc


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = genres10.get_dataset()

    #show_spectogram(x_train[0])

    classes = 10
    y_train = utils.np_utils.to_categorical(y_train, classes)
    y_test = utils.np_utils.to_categorical(y_test, classes)

    input_shape = x_train[0].shape

    model = create_model(input_shape)
    compile_model(model)
    train_model(model, x_train, y_train)

    score, acc = test_model(model, x_test, y_test)

