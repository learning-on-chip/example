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
    components = [0]
    sample_count = 100000
    chunk_size = 1000
    component_count = len(components)
    data = support.select(components=components, sample_count=sample_count)
    power = data[:, 0:(2 * component_count):2]
    power_limit = [np.min(power), np.max(power)]
    temperature = data[:, 1:(2 * component_count):2]
    temperature_limit = [np.min(temperature), np.max(temperature)]
    pp.figure(figsize=(16, 5), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        k = random.randint(0, sample_count - chunk_size)
        for i in range(component_count):
            power_chunk = power[k:(k + chunk_size), i]
            temperature_chunk = temperature[k:(k + chunk_size), i]
            partition = support.partition(power_chunk)
            pp.subplot(component_count, 2, 2*i + 1)
            plot(power_chunk, partition)
            pp.ylim(power_limit)
            pp.subplot(component_count, 2, 2*i + 2)
            plot(temperature_chunk, partition)
            pp.ylim(temperature_limit)
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

if __name__ == '__main__':
    main()
