import time
import os

import numpy as np

import keras
from keras import models
from keras import layers
from keras import optimizers

from nsutils import cifar10slice  # custom module


batch_size = 32
num_classes = 5
epochs = 2

MAX_PART = 2
MODELS_DIR = os.path.join(os.getcwd(), 'cifar10_slice_models')
MODEL_ARCH_FILE = 'cifar10_{max_part}_{part}_{acc}.json'
MODEL_WEIGHTS_FILE = 'cifar10_{max_part}_{part}_{acc}.h5'


def create_model(part: int, classn: int):
    input = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), padding='same', name="conv2_1_p%s" % part)(input)
    x = layers.Activation('relu', name="relu_1_p%s" % part)(x)
    x = layers.Conv2D(32, (3, 3), name="conv2_2_p%s" % part)(x)
    x = layers.Activation('relu', name="relu_2_p%s" % part)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2_1_p%s" % part)(x)
    x = layers.Dropout(0.25, name="drop_1_p%s" % part)(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name="conv2_3_p%s" % part)(x)
    x = layers.Activation('relu', name="relu_3_p%s" % part)(x)
    x = layers.Conv2D(64, (3, 3), name="conv2_4_p%s" % part)(x)
    x = layers.Activation('relu', name="relu_4_p%s" % part)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2_2_p%s" % part)(x)
    x = layers.Dropout(0.25, name="drop_2_p%s" % part)(x)

    x = layers.Flatten(name="flatten_1_p%s" % part)(x)
    x = layers.Dense(512, name="dense_1_p%s" % part)(x)
    x = layers.Activation('relu', name="relu_5_p%s" % part)(x)
    x = layers.Dropout(0.5, name="drop_3_p%s" % part)(x)

    x = layers.Dense(classn, name="dense_2_p%s" % part)(x)

    output = layers.Activation('softmax', name="softmax_1_p%s" % part)(x)

    model = models.Model(inputs=input, outputs=output)

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
    evaluation = model.evaluate(x_test, y_test)

    print('Model Accuracy = %.2f' % (evaluation[1]))

    return evaluation


def save_model(model, part, acc):
    # Save model and weights
    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    filename = MODEL_ARCH_FILE.format(max_part=MAX_PART, part=part, acc=acc)
    arch_file = os.path.join(MODELS_DIR, filename)
    model_json = model.to_json()
    with open(arch_file, 'w') as f:
        f.write(model_json)

    filename = MODEL_WEIGHTS_FILE.format(max_part=MAX_PART, part=part, acc=acc)
    weights_file = os.path.join(MODELS_DIR, filename)
    model.save(weights_file)

    print('Saved trained model at %s ' % weights_file)


if __name__ == '__main__':
    (xa_train, ya_train_cls), (xa_test, ya_test_cls) = cifar10slice.get_dataset(MAX_PART)

    for i in range(MAX_PART):
        x_train = xa_train[i]
        y_train = ya_train_cls[i]
        x_test = xa_test[i]
        y_test = ya_test_cls[i]

        #import pdb; pdb.set_trace()

        classn = ya_train_cls[i].shape[1]

        model = create_model(i, classn)

        model = compile_model(model)

        train_model(model,
            x_train, 
            y_train,
            x_test,
            y_test)

        evaluation = test_model(model,
            x_test,
            y_test)

        save_model(model, part=i, acc=evaluation[1])

