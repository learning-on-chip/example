#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import random, support
import matplotlib.pyplot as pp

def main():
    data = support.select(components=[0], sample_count=100000)
    sample_count, component_count = data.shape[0], data.shape[1] // 2
    chunk_size = 10000
    pp.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    while True:
        pp.clf()
        k = random.randint(0, sample_count - chunk_size)
        for i in range(0, 2 * component_count, 2):
            pp.subplot(component_count, 2, i + 1)
            pp.plot(data[k:(k + chunk_size), i + 0])
            pp.subplot(component_count, 2, i + 2)
            pp.plot(data[k:(k + chunk_size), i + 1])
        pp.pause(1e-3)
        if input('More? ') == 'no':
            break

if __name__ == '__main__':
    main()
