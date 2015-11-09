__author__ = 'mbz'

import pandas as pd
from os import listdir
from os.path import isfile, join

output = set()
input_folder = '/home/mbz/data/Sauber/groups/'
for f in listdir(input_folder):
    full_path = join(input_folder, f)
    if not isfile(full_path):
        continue

    print(full_path)
    df = pd.read_hdf(full_path, 'data')
    for g in df.groupby('Attack'):
        output.add(g[0])

for item in output:
    print(item)