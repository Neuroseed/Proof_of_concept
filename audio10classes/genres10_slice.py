import time
import os

import numpy

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import utils

import genres10


N_PARTS = 3

batch_size = 32
num_classes = 10
epochs = 3


def slice_dataset(x, y, partn=2, classn=None):
    "Разрезает датасет на partn частей"

    if not classn:
        classn = numpy.max(y) + 1

    step = float(classn) / partn

    xa = []
    ya = []

    for i in range(partn):
        down = round(step * i)
        top = round(step * (i + 1))
        cond = numpy.logical_and((down <= y), (y < top)).flatten()

        xa.append(x[cond])
        ya.append(y[cond].flatten() - down)

    return (xa, ya)


def create_model(input_shape, output_shape, part):
    input = layers.Input(shape=input_shape)

    x = layers.BatchNormalization(name="batchnorm_1_p%s" % part)(input)
    x = layers.Convolution2D(16, (7, 7), padding='same', activation='relu', name="conv2_1_p%s" % part)(input)
    x = layers.MaxPooling2D((3, 3), name="maxpool_1_p%s" % part)(x)
    
    x = layers.BatchNormalization(name="batchnorm_2_p%s" % part)(x)
    x = layers.Convolution2D(32, (5, 5), padding='same', activation='relu', name="conv2_2_p%s" % part)(x)
    x = layers.MaxPooling2D((3, 3), name="maxpool_2_p%s" % part)(x)

    x = layers.BatchNormalization(name="batchnorm_3_p%s" % part)(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', activation='relu', name="conv2_3_p%s" % part)(x)
    x = layers.MaxPooling2D((3, 3), name="maxpool_3_p%s" % part)(x)

    x = layers.BatchNormalization(name="batchnorm_4_p%s" % part)(x)
    x = layers.Flatten(name="flatten_1_p%s" % part)(x)
    x = layers.Dense(500, activation='relu', name="dense_1_p%s" % part)(x)

    x = layers.Dropout(0.5, name="dropout_1_p%s" % part)(x)
    output = layers.Dense(output_shape[0], activation='softmax', name="dense_2_p%s" % part)(x)

    model = models.Model(input, output)

    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    return model


def compile_model(model):
    # initiate RMSprop optimizer
    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    t = time.time()

    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)

    print('t', time.time() - t)


def test_model(model, x_test, y_test):
    score, acc = model.evaluate(x_test, y_test)

    print('Model Accuracy = %.2f' % acc)

    return score, acc


def save_model(model, part):
    models.save_model(model, 'submodels/model_%s_%s.h5' % (N_PARTS, part))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = genres10.get_dataset()

    xa_train, ya_train = slice_dataset(x_train, y_train, N_PARTS, num_classes)
    xa_test, ya_test = slice_dataset(x_test, y_test, N_PARTS, num_classes)

    ya_train = [utils.np_utils.to_categorical(y, numpy.max(y)+1) for y in ya_train]
    ya_test = [utils.np_utils.to_categorical(y, numpy.max(y)+1) for y in ya_test]

    input_shape = x_train[0].shape

    for part in range(N_PARTS):
        x_train = xa_train[part]
        y_train = ya_train[part]
        x_test = xa_test[part]
        y_test = ya_test[part]

        #import pdb; pdb.set_trace()

        output_shape = y_train.shape[1:]

        model = create_model(input_shape, output_shape, part)

        compile_model(model)
        train_model(model, x_train, y_train, x_test, y_test)
        score, acc = test_model(model, x_test, y_test)

        save_model(model, part)

