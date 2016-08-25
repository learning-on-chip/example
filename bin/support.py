import numpy as np
import sqlite3

DATABASE_PATH = 'tests/fixtures/database.sqlite3'

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.sqrt(np.var(data, axis=0))

def partition(power, left_margin=1, right_margin=1):
    power = np.int_(power > (np.min(power) + 1e-6))
    sample_count = len(power)
    activity = np.diff(power)
    switch = np.reshape(list(np.nonzero(activity)), [-1])
    if activity[switch[0]] == -1:
        switch = np.insert(switch, 0, -1)
    if activity[switch[-1]] == 1:
        switch = np.append(switch, sample_count - 1)
    assert(len(switch) % 2 == 0)
    count = len(switch) // 2
    partition = np.zeros([count, 2], dtype='uint')
    for i in range(count):
        j = switch[2 * i] + 1
        k = switch[2*i + 1] + 1
        assert(np.all(power[j:k] == 1))
        partition[i, 0] = max(0, j - left_margin)
        partition[i, 1] = min(sample_count, k + right_margin)
    return partition

def select(components=None, sample_count=None, path=DATABASE_PATH):
    print('Reading "%s"...' % path)
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
