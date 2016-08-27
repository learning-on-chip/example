import numpy as np
import sqlite3

DATABASE_PATH = 'tests/fixtures/database.sqlite3'

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.sqrt(np.var(data, axis=0))

def partition(power, left_margin=0, right_margin=0, min_length=10):
    power = np.int_(power > (np.min(power) + 1e-6))
    sample_count = len(power)
    activity = np.diff(power)
    switch = np.reshape(list(np.nonzero(activity)), [-1])
    if activity[switch[0]] == -1:
        switch = np.insert(switch, 0, -1)
    if activity[switch[-1]] == 1:
        switch = np.append(switch, sample_count - 1)
    assert(len(switch) % 2 == 0)
    total_count = len(switch) // 2
    partition = np.zeros([total_count, 2], dtype='uint')
    chosen_count = 0
    for i in range(total_count):
        j = switch[2 * i] + 1
        k = switch[2*i + 1] + 1
        assert(np.all(power[j:k] == 1))
        partition[chosen_count, 0] = max(0, j - left_margin)
        partition[chosen_count, 1] = min(sample_count, k + right_margin)
        if partition[chosen_count, 1] - partition[chosen_count, 0] >= min_length:
            chosen_count += 1
    return partition[:chosen_count, :]

def select(components=None, sample_count=None, path=DATABASE_PATH):
    print('Reading "{}"...'.format(path))
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    sql = 'SELECT count(*), count(DISTINCT component_id) FROM profiles'
    cursor.execute(sql)
    total_sample_count, total_component_count = cursor.fetchone()
    assert(total_sample_count % total_component_count == 0)
    total_sample_count //= total_component_count
    if components is None: components = list(range(0, total_component_count))
    components = list(filter(lambda c: c < total_component_count, components))
    component_count = len(components)
    if sample_count is None: sample_count = total_sample_count
    sample_count = min(sample_count, total_sample_count)
    print('Processing {}/{} samples for {}/{} components...'.format(
        sample_count, total_sample_count, component_count, total_component_count))
    sql = 'SELECT rowid, component_id, power, temperature FROM profiles ' \
        'WHERE component_id in ({}) ORDER BY time ASC LIMIT {}'
    sql = sql.format(', '.join([str(id) for id in components]), sample_count * component_count)
    cursor.execute(sql)
    data = np.zeros([sample_count, 2 * component_count])
    mapping = np.zeros(total_component_count, dtype='int')
    mapping[components] = np.arange(0, component_count)
    for row in cursor:
        i = (row[0] - 1) // total_component_count
        j = mapping[row[1]]
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
