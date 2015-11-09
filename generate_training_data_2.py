__author__ = 'mbz'

import pandas as pd
import numpy as np
import keras as keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from os import listdir
from os.path import isfile, join


def load_attacks():
    with open('/home/mbz/data/Sauber/attacks.txt') as f:
        return [x.strip() for x in f.readlines()]


def load_labels():
    df = pd.read_csv('/home/mbz/data/Sauber/to_neu_10sep2015.csv', sep=',')
    d = dict()
    for row in df[[1, 2]].iterrows():
        d[row[1][0] + '.h5'] = row[1][1]
    return d


attacks = load_attacks()
in_size = len(attacks)

labels = load_labels()
labels_set = list(set(labels.values()))
labels_set.sort()
out_size = len(labels_set)

print('%d X %d' % (in_size, out_size))

begin = 53000
end = 53300
step = 1
length = 30

batch_size = 1024

print('Build model...')
# model = Sequential()
# model.add(Dense(300, input_dim=in_size))
# model.add(Activation('relu'))
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(out_size))
# model.add(Activation('softmax'))
#
# #model.weights = pickle.load(open("/home/mbz/data/Sauber/model.p", "rb"))
# model.compile(loss='categorical_crossentropy', optimizer='SGD')
# model.load_weights('/home/mbz/data/Sauber/training/model_107.keras')


X = np.zeros((batch_size, in_size), dtype=np.float)
Y = np.zeros((batch_size, out_size), dtype=np.bool)

example_id = 0
model_id = 1
input_path = '/home/mbz/data/Sauber/groups/'
for f in listdir(input_path):
    full_path = join(input_path, f)
    if not isfile(full_path):
        continue

    print(full_path)
    df = pd.read_hdf(full_path, 'data')

    for i in range(begin, end, step):
        j = i + length
        part = df.query('@i <= Time < @j')

        if len(part) == 0:
            continue

        XX = np.zeros(in_size, dtype=np.double)
        for row in part.iterrows():
            attack = attacks.index(row[1]['Attack'])
            XX[attack] += row[1]['Count']

        XX = (XX - min(XX)) / (max(XX) - min(XX))
        X[example_id, :] = XX

        label = labels[f]
        Y[example_id, labels_set.index(label)] = True

        example_id += 1
        print('Data: ', example_id)
        if example_id == batch_size:
            #print('='*20)
            #p = model.predict(X)
            #yy = np.argmax(p, axis=1)
            #print('='*20)

            np.savetxt('/home/mbz/data/Sauber/training/input_%d.csv' % model_id, X, delimiter=",")
            np.savetxt('/home/mbz/data/Sauber/training/output_%d.csv' % model_id, Y, delimiter=",")

            example_id = 0
            model_id += 1
            X = np.zeros((batch_size, in_size), dtype=np.float)
            Y = np.zeros((batch_size, out_size), dtype=np.bool)

