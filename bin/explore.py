#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

from support import Config, Database
import matplotlib.collections as cl
import matplotlib.pyplot as pp
import numpy as np
import random, support

def main(config):
    database = Database(config)
    data = database.read()
    partition = database.partition()
    sample_count = data.shape[0]
    window_size = min(config.window_size, sample_count)
    pp.figure(figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        i = random.randint(0, sample_count - window_size)
        j = i + window_size
        subset = _select(partition, i, j) - i
        pp.subplot(2, 1, 1)
        _plot(data[i:j, 0], subset)
        pp.subplot(2, 1, 2)
        _plot(data[i:j, 1], subset)
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

def _plot(data, partition):
    lines = []
    for i in range(partition.shape[0]):
        j, k = partition[i, 0], partition[i, 1]
        top = np.max(data[j:k])
        lines.append([(j, top), (k, top)])
    pp.plot(data)
    pp.gca().add_collection(cl.LineCollection(lines, colors='red', linewidths=2))

def _select(partition, start, finish):
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

if __name__ == '__main__':
    config = Config({
        'database_path': Database.find(),
        'window_size': 1000,
    })
    main(config)
