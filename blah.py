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


print('Loading data...')
input = pd.read_csv('/home/mbz/data/Sauber/training/_input.csv').astype(float)
output = pd.read_csv('/home/mbz/data/Sauber/training/_output.csv').astype(float)

in_size = input.shape[1]
out_size = output.shape[1]
data_size = input.shape[0]
batch_size = data_size # 8192

print('Build model...')
model = Sequential()
model.add(Dense(500, input_dim=in_size))
model.add(Activation('relu'))
model.add(Dropout(300))
model.add(Activation('relu'))
model.add(Dropout(50))
model.add(Activation('relu'))
model.add(Dense(out_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.load_weights('/home/mbz/data/Sauber/models/model_70_0.csv')

X = np.zeros((batch_size, in_size), dtype=np.float)
Y = np.zeros((batch_size, out_size), dtype=np.bool)

for iteration in range(1, 2):
    for example_id in range(0, data_size, batch_size):
        print('=' * 20)
        print(iteration, ' ', example_id)
        print('=' * 20)

        X[:, :] = input.ix[example_id:example_id+batch_size-1, :]
        Y[:, :] = output.ix[example_id:example_id+batch_size-1, :] > 0.5

        #model.fit(X, Y, batch_size=4096, nb_epoch=100)
        p = model.predict(X)
        yy = np.argmax(p, axis=1)
        yyy = np.argmax(Y, axis=1)

        a = np.equal(yy, yyy)
        print('+' * 20)
        print(sum(a))
        print(Y.shape[0])
        print('+' * 20)


        #model.save_weights('/home/mbz/data/Sauber/models/model_%d_%d.csv' % (iteration, example_id))
        #

