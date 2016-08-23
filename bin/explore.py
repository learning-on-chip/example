#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import random, support
import matplotlib.pyplot as pp

def main():
    data = support.select(component_ids=[0], sample_limit=100000)
    sample_count, component_count = data.shape
    chunk_size = 10000
    support.figure()
    while True:
        pp.clf()
        k = random.randint(0, sample_count - chunk_size)
        for i in range(component_count):
            pp.subplot(component_count, 1, i + 1)
            pp.plot(data[k:(k + chunk_size), i])
        pp.pause(1e-3)
        if input('More? ') == 'no': break

if __name__ == '__main__': main()
