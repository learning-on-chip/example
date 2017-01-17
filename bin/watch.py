#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from support import Config
import matplotlib.pyplot as pp
import numpy as np
import socket

def main(config):
    print('Connecting to {}...'.format(config.address))
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(config.address)
    client = client.makefile(mode="r")
    plots = _prepare(config.dimension_count)
    x_limit = [0, 1]
    y_limit = [-1, 1]
    while True:
        row = [float(number) for number in client.readline().split(',')]
        sample_count = len(row) // 2
        if sample_count <= 1:
            continue
        x = np.arange(0, sample_count)
        y = np.reshape(np.array(row[0:sample_count]), [-1, config.dimension_count])
        y_hat = np.reshape(np.array(row[sample_count:]), [-1, config.dimension_count])
        x_limit[1] = sample_count - 1
        y_limit[0] = min(y_limit[0], np.min(y), np.min(y_hat))
        y_limit[1] = max(y_limit[1], np.max(y), np.max(y_hat))
        for i in range(config.dimension_count):
            plots[3*i + 0].set_xdata(x)
            plots[3*i + 0].set_ydata(y[:, i])
            plots[3*i + 1].set_xdata(x)
            plots[3*i + 1].set_ydata(y_hat[:, i])
            plots[3*i + 2].set_xdata(x_limit)
            pp.subplot(config.dimension_count, 1, i + 1)
            pp.xlim(x_limit)
            pp.ylim(y_limit)
        pp.pause(1e-3)

def _prepare(dimension_count):
    pp.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    plots = []
    for i in range(dimension_count):
        pp.subplot(dimension_count, 1, i + 1)
        plots.append(pp.plot([0, 1], [0, 0], 'b')[0])
        plots.append(pp.plot([0, 1], [0, 0], 'g')[0])
        plots.append(pp.plot([0, 1], [0, 0], 'r')[0])
        pp.legend(['Observed', 'Predicted'])
    pp.pause(1e-3)
    return plots

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    chunks = sys.argv[1].split(':')
    assert(len(chunks) == 2)
    config = Config({
        'dimension_count': 1,
        'address': (chunks[0], int(chunks[1])),
    })
    main(config)
