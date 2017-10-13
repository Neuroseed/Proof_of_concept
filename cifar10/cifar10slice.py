import numpy
import keras
from keras import datasets
from keras import utils

# делить датасет на partn частей:
partn = 5

# Загрузить датасет cifar10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


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

        ya.append()

    return (xa, ya)


def get_dataset(partn):
    # slice train
    xa_train, ya_train = slice_dataset(x_train, y_train, partn)

    # slice test
    xa_test, ya_test = slice_dataset(x_test, y_test, partn)

    # преобразовать все элементы в тип float32
    xa_train = [x.astype('float32') for x in xa_train]
    xa_test = [x.astype('float32') for x in xa_test]

    # нормировать все элементы 0..255 -> 0..1
    xa_train = [x / 255 for x in xa_train]
    xa_test = [x / 255 for x in xa_test]

    # преобразование в one-hot encoding
    ya_train_cls = [utils.to_categorical(y, numpy.max(y)+1) for y in ya_train]
    ya_test_cls = [utils.to_categorical(y, numpy.max(y)+1) for y in ya_test]

    return (xa_train, ya_train_cls), (xa_test, ya_test_cls)


(xa_train, ya_train_cls), (xa_test, ya_test_cls) = get_dataset(partn)

