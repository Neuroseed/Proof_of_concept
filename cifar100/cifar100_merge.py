import time
import os
from os import path
import json
import numpy

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import utils
from keras import datasets
from keras import callbacks
from keras.models import model_from_json


MAX_PART = 4

MODEL_PARTS_DIR = 'cifar100_slice_models'
MODEL_PARTS_ARCH_FILES = [
    'cifar100_4_0_0.4384.json',
    'cifar100_4_1_0.4284.json',
    'cifar100_4_2_0.4584.json',
    'cifar100_4_3_0.4352.json']

MODEL_PARTS_WEIGHTS_FILES = [
    'cifar100_4_0_0.4384.h5',
    'cifar100_4_1_0.4284.h5',
    'cifar100_4_2_0.4584.h5',
    'cifar100_4_3_0.4352.h5']

MODELS_DIR = 'cifar10_merge_models'
MODEL_ARCH_FILE = 'cifar10_merge_{acc}.json'
MODEL_WEIGHTS_FILE = 'cifar10_merge_{acc}.h5'

batch_size = 32
num_classes = 100
epochs = 10

class History(callbacks.Callback):
    def __init__(self, filepath):
        callbacks.Callback.__init__(self)

        self.filepath = filepath
        self.history = {'acc': [], 'time': []}
        self.start_time = None

    def on_batch_begin(self, batch, logs={}):
        if not self.start_time:
            self.start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        acc = logs.get('acc', 0)
        spent_time = time.time() - self.start_time

        self.history['acc'].append(float(acc))
        self.history['time'].append(spent_time)

    def on_train_end(self, logs):
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)

h = History('cifar100_merge_p4_eph%s.json' % epochs)


def load_model_part(part):
    filename = MODEL_PARTS_ARCH_FILES[part]
    filepath = path.join(MODEL_PARTS_DIR, filename)
    with open(filepath) as f:
        model = model_from_json(f.read())

    filename = MODEL_PARTS_WEIGHTS_FILES[part]
    filepath = path.join(MODEL_PARTS_DIR, filename)
    model.load_weights(filepath)

    weights, bias = model.layers[-2].get_weights()

    # delete layers
    for i in range(2):
        #layer = model.layers[-1]

        #for node_index, node in enumerate(layer.inbound_nodes):
        #    node_key = layer._node_key(layer, original_node_index)
        #    if node_key in model.container_nodes:
        #        model.container_nodes.remove(node_key)

        model.layers[-1].inbound_nodes = []
        model.layers[-1].outbound_nodes = []
        model.layers.pop()

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    # freeze layers
    for layer in model.layers[:-3]:
        layer.trainable = False
    #model.layers[-1].trainable = True

    model.summary()

    return model, weights, bias


def merge_weights(weights, bias):
    wshapes = [w.shape for w in weights]
    shape = [sum(ar) for ar in zip(*wshapes)]

    nweights = numpy.zeros(shape)

    delimx, delimy = 0, 0

    for w in weights:
        delimx += w.shape[0]
        delimy += w.shape[1]

        nweights[(delimx - w.shape[0]):delimx, (delimy - w.shape[1]):delimy] = w
    
    bias = numpy.concatenate(bias, axis=0)

    return nweights, bias


def create_merge_model():
    parts = [load_model_part(i) for i in range(MAX_PART)]

    model_parts, weights, bias = zip(*parts)

    input = keras.layers.Input(shape=(32, 32, 3))

    models_layers = [model(input) for model in model_parts]

    conc = keras.layers.concatenate(models_layers)

    dense10 = layers.Dense(num_classes, activation='softmax')
    output = dense10(conc)

    weights, bias = merge_weights(weights, bias)

    dense10.set_weights((weights, bias))

    model = models.Model(inputs=input, outputs=output)

    model.summary()

    return model


def compile_model(model):
    # initiate RMSprop optimizer
    optimizer = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_test, y_test):
    t = time.time()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks = [h])

    print('t', time.time() - t)

    return model


def test_model(model):
    evaluation = model.evaluate(x_test, y_test)

    print('Model Accuracy = %.2f' % (evaluation[1]))

    return evaluation


def save_model(model, acc):
    # Save model and weights
    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    filename = MODEL_ARCH_FILE.format(acc=acc)
    arch_file = os.path.join(MODELS_DIR, filename)
    model_json = model.to_json()
    with open(arch_file, 'w') as f:
        f.write(model_json)

    filename = MODEL_WEIGHTS_FILE.format(acc=acc)
    weights_file = os.path.join(MODELS_DIR, filename)
    model.save(weights_file)

    print('Saved trained model at %s ' % weights_file)


def predict(model):
    import PIL as pil
    import scipy

    #filepath = 'images/plane.jpg'
    filepath = 'images/car.jpg'
    image = scipy.misc.imread(filepath, mode='RGB')
    image = scipy.misc.imresize(image, (32, 32))

    img = pil.Image.fromarray(image, 'RGB')
    img.show()

    #import pdb; pdb.set_trace()

    image = image.astype('float32')

    image = image.reshape((32, 32, 3))

    image /= 255

    x_test = numpy.array([image,])

    result = model.predict(x_test)
    #result = result.reshape((10,))
    for i in range(10):
        print('for', i, 'p =', result[0][i])


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    #x_train = x_train[:10]
    #y_train = y_train[:10]
    #x_test = x_test[:10]
    #y_test = y_test[:10]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    model = create_merge_model()
    model = compile_model(model)
    model = train_model(model, x_train, y_train, x_test, y_test)
    evaluation = test_model(model)
    #save_model(model, evaluation[1])
    #predict(model)
