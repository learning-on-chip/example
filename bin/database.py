import numpy as np
import sqlite3

DATABASE_PATH = 'output/database.sqlite3'

class Database:
    def __init__(self, component_id, path=DATABASE_PATH):
        self.component_id = component_id
        self.connection = sqlite3.connect(path)

    def partition(self):
        query = 'SELECT time, kind FROM markers ' \
                'WHERE component_id = {} ' \
                'ORDER BY time ASC, kind DESC'
        cursor = self.connection.cursor()
        cursor.execute(query.format(self.component_id))
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

    def read(self, start, finish, quantity='power'):
        query = 'SELECT time, {} FROM profiles ' \
                'WHERE component_id = {} AND time >= {} AND time < {} '
        cursor = self.connection.cursor()
        cursor.execute(query.format(quantity, self.component_id, start, finish))
        data = np.zeros([finish - start])
        for row in cursor:
            data[row[0] - start] = row[1]
        return data
