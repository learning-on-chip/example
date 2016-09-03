import numpy as np
import os, sqlite3

OUTPUT_PATH = 'output'

class Database:
    def __init__(self, path=None):
        path = _find() if path is None else path
        print('Reading the data in "{}"…'.format(path))
        self.connection = sqlite3.connect(path)

    def partition(self):
        query = 'SELECT time, kind FROM events ORDER BY time ASC, kind DESC'
        cursor = self.connection.cursor()
        cursor.execute(query)
        data = []
        start = None
        for row in cursor:
            if row[1] == 0:
                assert(start is None)
                start = row[0]
            else:
                assert(row[1] == 1 and start is not None and start < row[0])
                data.append([start, row[0]])
                start = None
        return np.array(data)

    def read(self, start=None, finish=None):
        condition = []
        if start is not None:
            condition.append('time >= {}'.format(start))
        if finish is not None:
            condition.append('time < {}'.format(finish))
        condition = ' WHERE ' + ' AND '.join(condition) if len(condition) > 0 else ''
        cursor = self.connection.cursor()
        cursor.execute('SELECT count(*) FROM profiles{}'.format(condition))
        count = cursor.fetchone()[0]
        cursor.execute('SELECT time, power, temperature FROM profiles{}'.format(condition))
        data = np.zeros([count, 2])
        start = start if start is not None else 0
        for row in cursor:
            i = row[0] - start
            data[i, 0] = row[1]
            data[i, 1] = row[2]
        return data

def _find():
    for root, _, files in os.walk(OUTPUT_PATH):
        for file in files:
            if file.endswith('.sqlite3'):
                return os.path.join(root, file)
    raise('filed to find a database')
