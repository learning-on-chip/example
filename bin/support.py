import numpy as np
import sqlite3

DATABASE_PATH = 'output/database.sqlite3'

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.sqrt(np.var(data, axis=0))

def partition(start_id, finish_id, min_length=5, path=DATABASE_PATH):
    print('Reading markers from "{}"...'.format(path))
    sql = 'SELECT sequence_id, kind FROM markers ' \
        'WHERE component_id = 0 AND sequence_id >= {} AND sequence_id < {} ' \
        'ORDER BY sequence_id ASC, kind DESC'
    sql = sql.format(start_id, finish_id)
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(sql)
    partition = []
    start, finish = [], []
    for row in cursor:
        if row[1] == 0:
            start.append(row[0])
        elif row[1] == 1:
            finish.append(row[0])
        else:
            assert(False)
        if len(start) == 0 or len(finish) == 0:
            continue
        if finish[0] < start[0]:
            start.insert(0, start_id)
        i = start[0]
        del start[0]
        j = finish[-1]
        del finish[-1]
        if j - i >= min_length:
            partition.append([i, j])
    connection.close()
    assert(len(start) <= 1 and len(finish) <= 1)
    return np.array(partition)

def select(component_ids=None, sample_count=None, path=DATABASE_PATH):
    print('Reading profiles from "{}"...'.format(path))
    total_sample_count, total_component_count = _count(path)
    component_ids, component_count, component_query, component_mapping = _prepare_components(
        component_ids, total_component_count)
    if sample_count is None: sample_count = total_sample_count
    sample_count = min(sample_count, total_sample_count)
    print('Processing {}/{} samples for {}/{} components...'.format(
        sample_count, total_sample_count, component_count, total_component_count))
    sql = 'SELECT sequence_id, component_id, power, temperature FROM profiles ' \
        'WHERE component_id in ({}) ORDER BY sequence_id ASC LIMIT {}'
    sql = sql.format(component_query, sample_count * component_count)
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    cursor.execute(sql)
    data = np.zeros([sample_count, 2 * component_count])
    for row in cursor:
        i = row[0]
        j = component_mapping[row[1]]
        data[i, 2*j + 0] = row[2]
        data[i, 2*j + 1] = row[3]
    connection.close()
    return data

def shift(data, amount, padding=0.0):
    data = np.roll(data, amount, axis=0)
    if amount > 0:
        data[:amount, :] = padding
    elif amount < 0:
        data[amount:, :] = padding
    return data

def _count(path):
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    sql = 'SELECT count(*), count(DISTINCT component_id) FROM profiles'
    cursor.execute(sql)
    sample_count, component_count = cursor.fetchone()
    assert(sample_count % component_count == 0)
    sample_count //= component_count
    connection.close()
    return sample_count, component_count

def _prepare_components(ids, total_count):
    if ids is None: ids = list(range(0, total_count))
    ids = list(filter(lambda id: id < total_count, ids))
    count = len(ids)
    query = ', '.join([str(id) for id in ids])
    mapping = np.zeros(total_count, dtype='int')
    mapping[ids] = np.arange(0, count)
    return ids, count, query, mapping
