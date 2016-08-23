import numpy as np
import sqlite3

DATABASE_PATH = 'tests/fixtures/database.sqlite3'

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.sqrt(np.var(data, axis=0))

def select(component_ids=None, sample_limit=None, path=DATABASE_PATH):
    print('Reading "%s"...' % path)
    connection = sqlite3.connect(path)
    cursor = connection.cursor()
    sql = 'SELECT count(*), count(DISTINCT component_id) FROM profiles'
    cursor.execute(sql)
    record_count, component_count = cursor.fetchone()
    assert(record_count % component_count == 0)
    sample_count = record_count // component_count
    if component_ids is None: component_ids = list(range(0, component_count))
    component_ids = list(filter(lambda c: c < component_count, component_ids))
    if sample_limit is None: sample_limit = sample_count
    sample_limit = min(sample_limit, sample_count)
    print('Processing {}/{} samples for {}/{} components...'.format(
        sample_limit, sample_count, len(component_ids), component_count))
    sql = 'SELECT rowid, component_id, temperature FROM profiles ' \
        'WHERE component_id in ({}) ORDER BY time ASC LIMIT {}'
    sql = sql.format(', '.join([str(id) for id in component_ids]),
                     sample_limit * len(component_ids))
    cursor.execute(sql)
    data = np.zeros([sample_limit, len(component_ids)])
    mapping = np.zeros(component_count, dtype='int')
    mapping[component_ids] = np.arange(0, len(component_ids))
    for row in cursor:
        i = (row[0] - 1) // component_count
        j = mapping[row[1]]
        data[i, j] = row[2]
    connection.close()
    return data
