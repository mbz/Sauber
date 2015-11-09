__author__ = 'mbz'

import pandas as pd
from collections import deque

# dic = dict()
# data = pd.read_csv('/home/mbz/data/Sauber/to_neu_10sep2015.csv')
# for index, row in data.iterrows():
#     dic[row['entity_id_hash']] = row['industry_sector']
#
# stores = dict()
# for s in set(dic.values()):
#     name = '/home/mbz/data/Sauber/stores/%s.h5' % s.replace('/', '_')
#     stores[s] = pd.HDFStore(name)


queues = dict()

def add_data_to_store(entity, row):
    global entities, queues
    if entity not in queues.keys():
        queues[entity] = deque()

    d = {'Time': int(row[1]),
         'Attack': row[4],
         'Count': int(row[6])}

    queues[entity].append(d)

    if len(queues[entity]) == 10000:
        name = '/home/mbz/data/Sauber/entities/%s.h5' % entity
        with pd.HDFStore(name) as store:
            df = pd.DataFrame.from_records(queues[entity])
            store.append('data', df, min_itemsize=100, index=False)
            queues[entity].clear()


k = 0
with pd.HDFStore('/home/mbz/data/Sauber/hdf.h5') as input:
    for df in input.select('data', chunksize=10000):
        for row_index, row in df.iterrows():
            entity = row[3]
            add_data_to_store(entity, row)

        k += 1
        print(k)

for entity in queues.keys():
    name = '/home/mbz/data/Sauber/entities/%s.h5' % entity
    with pd.HDFStore(name) as store:
        df = pd.DataFrame.from_records(queues[entity])
        store.append('data', df, min_itemsize=100, index=False)
        queues[entity].clear()




