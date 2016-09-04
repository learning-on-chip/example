import numpy as np
import sqlite3

class Config:
    def __init__(self, options={}):
        self.update(options)

    def update(self, options):
        for key in options:
            setattr(self, key, options[key])

class Database:
    def __init__(self, config):
        self.connection = sqlite3.connect(config.database_path)

    def count(self, *arguments):
        cursor = self.connection.cursor()
        condition = Database._condition(*arguments)
        cursor.execute('SELECT count(*) FROM profiles{}'.format(condition))
        return cursor.fetchone()[0]

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

    def read(self, *arguments):
        count = self.count(*arguments)
        cursor = self.connection.cursor()
        condition = Database._condition(*arguments)
        cursor.execute('SELECT time, power, temperature FROM profiles{}'.format(condition))
        data = np.zeros([count, 2])
        start = None
        for row in cursor:
            if start is None:
                start = row[0]
            i = row[0] - start
            data[i, 0] = row[1]
            data[i, 1] = row[2]
        return data

    def _condition(start=None, finish=None):
        condition = []
        if start is not None:
            condition.append('time >= {}'.format(start))
        if finish is not None:
            condition.append('time < {}'.format(finish))
        return ' WHERE ' + ' AND '.join(condition) if len(condition) > 0 else ''

def shift(data, amount, padding=0.0):
    data = np.roll(data, amount, axis=0)
    if amount > 0:
        data[:amount, :] = padding
    elif amount < 0:
        data[amount:, :] = padding
    return data
