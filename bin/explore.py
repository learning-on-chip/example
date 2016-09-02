#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

from database import Database
import matplotlib.collections as cl
import matplotlib.pyplot as pp
import numpy as np
import random, support

def main(component_id=0, window=int(1e5)):
    database = Database(component_id=component_id)
    partition = database.partition()
    horizon = partition[-1, 1]
    window = min(window, horizon)
    pp.figure(figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        start = random.randint(0, horizon - window)
        finish = start + window
        print('Reading data from {} to {} ({})…'.format(start, finish, finish - start))
        _plot(database.read(start, finish), _find(partition, start, finish) - start)
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

def _find(partition, start, finish):
    i, j = None, None
    for k in range(partition.shape[0]):
        if partition[k, 0] >= start:
            i = k
            break
    for k in reversed(range(partition.shape[0])):
        if partition[k, 1] <= finish:
            j = k + 1
            break
    assert(i is not None and j is not None)
    return partition[i:j, :]

def _plot(data, partition):
    lines = []
    for i in range(partition.shape[0]):
        j, k = partition[i, 0], partition[i, 1]
        top = np.max(data[j:k])
        lines.append([(j, top), (k, top)])
    pp.plot(data)
    pp.gca().add_collection(cl.LineCollection(lines, colors='red', linewidths=2))

if __name__ == '__main__':
    main()
