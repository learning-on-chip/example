#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.collections as cl
import matplotlib.pyplot as pp
import numpy as np
import random, support

def plot(data, partition):
    pp.plot(data)
    lines = []
    for i in range(partition.shape[0]):
        j, k = partition[i, 0], partition[i, 1]
        top = np.max(data[j:k])
        lines.append([(j, top), (k, top)])
    pp.gca().add_collection(cl.LineCollection(lines, colors='red'))

def main():
    sample_count = 100000
    chunk_size = 1000
    data = support.select(components=[0], sample_count=sample_count)
    component_count = data.shape[1] // 2
    pp.figure(figsize=(16, 5), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        k = random.randint(0, sample_count - chunk_size)
        for i in range(0, 2 * component_count, 2):
            power = data[k:(k + chunk_size), 0]
            temperature = data[k:(k + chunk_size), 1]
            partition = support.partition(power)
            pp.subplot(component_count, 2, i + 1)
            plot(power, partition)
            pp.subplot(component_count, 2, i + 2)
            plot(temperature, partition)
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

if __name__ == '__main__':
    main()
