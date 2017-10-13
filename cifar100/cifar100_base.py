import time
import os
import json
import pickle
import numpy as np
import keras
from keras import models
from keras import layers
from keras import callbacks
from keras import datasets
from keras.preprocessing.image import ImageDataGenerator


batch_size = 32
num_classes = 100
epochs = 10
num_predictions = 20

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


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

h = History('cifar100_base_epochs_%s_history.json' % epochs)


input = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, (3, 3), padding='same')(input)
x = layers.Activation('relu')(x)
x = layers.Conv2D(32, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, (3, 3), padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(512)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes)(x)
output = layers.Activation('softmax')(x)

model = models.Model(inputs=input, outputs=output)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

model.fit(x_train[:10], y_train[:10],
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[h])

