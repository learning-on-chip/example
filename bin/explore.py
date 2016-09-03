#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

from database import Database
import matplotlib.collections as cl
import matplotlib.pyplot as pp
import numpy as np
import random, support

def main(window=int(1000)):
    database = Database()
    partition = database.partition()
    horizon = partition[-1, 1]
    window = min(window, horizon)
    pp.figure(figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        start = random.randint(0, horizon - window)
        finish = start + window
        window_data = database.read(start, finish)
        window_partition = _find(partition, start, finish) - start
        pp.subplot(2, 1, 1)
        _plot(window_data[:, 0], window_partition)
        pp.subplot(2, 1, 2)
        _plot(window_data[:, 1], window_partition)
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
