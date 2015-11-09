__author__ = 'mbz'

import pandas as pd

with pd.HDFStore('/home/mbz/data/Sauber/hdf.h5') as store:
    for df in pd.read_csv('/home/mbz/data/Sauber/bitsight.dat', sep=',', header = None, chunksize=100):
        data = df[[1, 3, 4, 6]]
        store.append('data', data, min_itemsize=100, index=False)


