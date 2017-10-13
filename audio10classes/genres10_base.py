import time
import os
import numpy
import scipy

import librosa
from librosa import display

from keras import models
from keras import layers
from keras import utils
from keras.models import Sequential

#from kapre.time_frequency import Spectrogram
#from kapre.time_frequency import Melspectrogram
#from kapre.utils import Normalization2D
#from kapre.augmentation import AdditiveNoise

import genres10


# 6 channels (!), maybe 1-sec audio signal, for an example.
epochs = 2


def show_spectogram(data):
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    display.specshow(data)
    plt.title('log-Spectrogram by Librosa')
    plt.show()


def create_model(shape):
    input = layers.Input(shape=shape)

    # A mel-spectrogram layer
    #model.add(Melspectrogram(n_dft=512, n_hop=256, 
    #    input_shape=input_shape,
    #    border_mode='same', sr=sr, n_mels=128,
    #    fmin=0.0, fmax=sr/2, power=1.0,
    #    return_decibel_melgram=False, trainable_fb=False,
    #    trainable_kernel=False,
    #    name='trainable_stft'))

    # Maybe some additive white noise.
    #model.add(AdditiveNoise(power=0.2))

    # If you wanna normalise it per-frequency
    #model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'

    x = layers.BatchNormalization()(input)
    x = layers.Convolution2D(16, (7, 7), padding='same', activation='relu')(input)
    x = layers.MaxPooling2D((3, 3))(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Convolution2D(32, (5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(500, activation='relu')(x)

    #x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = models.Model(input, output)

    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    # After this, it's just a usual keras workflow. For example..
    # Add some layers, e.g., model.add(some convolution layers..)

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

