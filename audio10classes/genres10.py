import os
import numpy
import scipy
import librosa

N_MELS = 96
N_FFT = 512
HOP_LEN = 256
DURA = 1

dataset_dir = 'datasets/genres'
dataset_numpy = 'datasets/genres.npz'
cut = 0.2
len_second = 15.0 # 1 second


def prepare_dataset():
    data = []
    labels = []
    classes_id = []

    for label in os.listdir(dataset_dir):
        label_id = len(labels)
        labels.append(label)

        path = os.path.join(dataset_dir, label)
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)

                src, sr = librosa.load(fpath, sr=None, mono=True, duration=len_second)

                #src = src[numpy.newaxis, :]

                n_sample = src.shape[0]

                logam = librosa.logamplitude
                melgram = librosa.feature.melspectrogram

                #src = logam(melgram(y=src, sr=sr, 
                #    n_mels=N_MELS)**2,
                #    ref_power=1.0)

                mel = librosa.feature.melspectrogram(src, sr=sr)
                #spc = logam(numpy.abs(librosa.stft(src))**2, ref_power=numpy.max)
                #f, t, sxx = scipy.signal.spectrogram(src, sr)

                #import pdb; pdb.set_trace()

                data.append(mel)
                classes_id.append(label_id)

    x = numpy.array(data)
    y = numpy.array(classes_id)

    index = list(range(y.shape[0]))
    numpy.random.shuffle(index)

    x = x[index]
    y = y[index]

    bound = int(y.shape[0] * cut)

    x_train = x[bound:]
    y_train = y[bound:]
    x_test = x[:bound]
    y_test = y[:bound]

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def save_dataset(x_train, y_train, x_test, y_test):
    numpy.savez(dataset_numpy,
        x_train=x_train, 
        y_train=y_train, 
        x_test=x_test, 
        y_test=y_test)


def get_dataset():
    if os.path.isfile(dataset_numpy):
        npz = numpy.load(dataset_numpy)

        x_train = npz['x_train']
        y_train = npz['y_train']
        x_test = npz['x_test']
        y_test = npz['y_test']

        return x_train, y_train, x_test, y_test
    else:
        x_train, y_train, x_test, y_test = prepare_dataset()

        save_dataset(x_train, y_train, x_test, y_test)

