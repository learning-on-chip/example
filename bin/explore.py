#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.collections as cl
import matplotlib.pyplot as pp
import numpy as np
import random, support

def main(component_ids=[0], sample_count=100000, chunk_size=1000):
    component_count = len(component_ids)
    data = support.select(component_ids=component_ids, sample_count=sample_count)
    sample_count = data.shape[0]
    chunk_size = min(chunk_size, sample_count)
    power_limit = _limit(data[:, 0:(2 * component_count):2])
    temperature_limit = _limit(data[:, 1:(2 * component_count):2])
    pp.figure(figsize=(16, 5), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        k = random.randint(0, sample_count - chunk_size)
        for i in range(component_count):
            partition = support.partition(k, k + chunk_size) - k
            pp.subplot(2 * component_count, 1, 2*i + 1)
            _plot(data[k:(k + chunk_size), 2 * i], partition)
            pp.ylim(power_limit)
            pp.subplot(2 * component_count, 1, 2*i + 2)
            _plot(data[k:(k + chunk_size), 2*i + 1], partition)
            pp.ylim(temperature_limit)
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

def _limit(data):
    limit = [np.min(data), np.max(data)]
    delta = 0.05*(limit[1] - limit[0])
    return [limit[0] - delta, limit[1] + delta]

def _plot(data, partition):
    delta = 0.01*(np.max(data) - np.min(data))
    pp.plot(data)
    lines = []
    for i in range(partition.shape[0]):
        j, k = partition[i, 0], partition[i, 1]
        top = delta + np.max(data[j:k])
        lines.append([(j, top), (k, top)])
    pp.gca().add_collection(cl.LineCollection(lines, colors='red'))

if __name__ == '__main__':
    main()
