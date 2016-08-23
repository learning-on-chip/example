#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import socket, support

def main(dimension_count, address):
    print('Connecting to {}...'.format(address))
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(address)
    client = client.makefile(mode="r")
    support.figure()
    pp.pause(1e-3)
    y_limit = [-1, 1]
    while True:
        row = [float(number) for number in client.readline().split(',')]
        half = len(row) // 2
        y = np.reshape(np.array(row[0:half]), [-1, dimension_count])
        y_hat = np.reshape(np.array(row[half:]), [-1, dimension_count])
        y_limit[0] = min(y_limit[0], np.min(y), np.min(y_hat))
        y_limit[1] = max(y_limit[1], np.max(y), np.max(y_hat))
        pp.clf()
        for i in range(dimension_count):
            pp.subplot(dimension_count, 1, i + 1)
            pp.plot(y[:, i])
            pp.plot(y_hat[:, i])
            pp.xlim([0, y.shape[0] - 1])
            pp.ylim(y_limit)
            pp.legend(['Observed', 'Predicted'])
        pp.pause(1e-3)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    chunks = sys.argv[1].split(':')
    assert(len(chunks) == 2)
    main(1, (chunks[0], int(chunks[1])))
