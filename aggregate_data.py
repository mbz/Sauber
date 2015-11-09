__author__ = 'mbz'

import pandas as pd
from os import listdir
from os.path import isfile, join

input_path = '/home/mbz/data/Sauber/entities/'
output_folder = '/home/mbz/data/Sauber/groups/'
for f in listdir(input_path):
    full_path = join(input_path, f)
    if not isfile(full_path):
        continue

    print(full_path)
    df = pd.read_hdf(full_path, 'data')
    with pd.HDFStore(join(output_folder, f)) as output:
        for g in df.groupby(['Time', 'Attack']):
            output.append('data', g[1], min_itemsize=100, index=False)

