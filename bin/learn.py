#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as pp
import numpy as np
import subprocess, support, tempfile
import tensorflow as tf

def learn(f, dimension_count, sample_count, epoch_count, train_each,
          train_monitor, predict_each, predict_count, predict_pace,
          predict_monitor):

    epoch_sample_count = sample_count - predict_count
    epoch_sample_count -= epoch_sample_count % predict_each
    predict_phases = np.cumsum(predict_pace)

    layer_count = 1
    unit_count = 20
    learning_rate = 1e-2
    gradient_norm = 1

    model = configure(dimension_count, layer_count, unit_count)
    graph = tf.get_default_graph()
    tf.train.SummaryWriter('log', graph)

    x = tf.placeholder(tf.float32, [1, None, dimension_count], name='x')
    y = tf.placeholder(tf.float32, [1, None, dimension_count], name='y')
    (y_hat, loss), (start, finish) = model(x, y)

    with tf.variable_scope('optimization'):
        parameters = tf.trainable_variables()
        gradient = tf.gradients(loss, parameters)
        gradient, _ = tf.clip_by_global_norm(gradient, gradient_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.apply_gradients(zip(gradient, parameters))

    initialize = tf.initialize_variables(tf.all_variables(), name='initialize')

    session = tf.Session(graph=graph)
    session.run(initialize)

    parameter_count = np.sum([int(np.prod(p.get_shape())) for p in parameters])
    print('Parameters: %d' % parameter_count)
    print('Epoch samples: %d' % epoch_sample_count)
    for e in range(epoch_count):
        train_fetches = {'finish': finish, 'train': train, 'loss': loss}
        train_feeds = {
            start: np.zeros(start.get_shape(), dtype=np.float32),
            x: np.zeros([1, train_each, dimension_count], dtype=np.float32),
            y: np.zeros([1, train_each, dimension_count], dtype=np.float32),
        }
        predict_fetches = {'finish': finish, 'y_hat': y_hat}
        predict_feeds = {start: None, x: None}

        Y = np.zeros([predict_count, dimension_count])
        Y_hat = np.zeros([predict_count, dimension_count])
        for s, t in zip(range(epoch_sample_count - 1),
                        range(1, epoch_sample_count)):

            train_feeds[x] = np.roll(train_feeds[x], -1, axis=1)
            train_feeds[y] = np.roll(train_feeds[y], -1, axis=1)
            train_feeds[x][0, -1, :] = f(s)
            train_feeds[y][0, -1, :] = f(t)

            if t % train_each == 0:
                train_results = session.run(train_fetches, train_feeds)
                train_feeds[start] = train_results['finish']
                train_monitor(progress=(e, t // train_each, t),
                              loss=train_results['loss'].flatten())

            phase = np.nonzero(predict_phases >= (s % predict_phases[-1]))[0][0]
            if phase % 2 == 1 and t % predict_each == 0:
                lag = t % train_each
                predict_feeds[start] = train_feeds[start]
                predict_feeds[x] = np.reshape(
                    train_feeds[y][0, (train_each - 1 - lag):, :],
                    [1, 1 + lag, -1])
                for i in range(predict_count):
                    predict_results = session.run(predict_fetches,
                                                  predict_feeds)
                    predict_feeds[start] = predict_results['finish']
                    Y_hat[i, :] = predict_results['y_hat'][-1, :]
                    predict_feeds[x] = np.reshape(Y_hat[i, :], [1, 1, -1])
                    Y[i, :] = f(t + i + 1)
                predict_monitor(Y, Y_hat)

def configure(dimension_count, layer_count, unit_count):
    def compute(x, y):
        with tf.variable_scope('network') as scope:
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            cell = tf.nn.rnn_cell.LSTMCell(unit_count, initializer=initializer,
                                           forget_bias=0.0, use_peepholes=True,
                                           state_is_tuple=True)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_count,
                                               state_is_tuple=True)
            start, state = initialize()
            h, state = tf.nn.dynamic_rnn(cell, x, initial_state=state,
                                         parallel_iterations=1)
            finish = finalize(state)
        return regress(h, y), (start, finish)

    def finalize(state):
        parts = []
        for i in range(layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.pack(parts, name='finish')

    def initialize():
        start = tf.placeholder(tf.float32, [2 * layer_count, 1, unit_count],
                               name='start')
        parts = tf.unpack(start)
        state = []
        for i in range(layer_count):
            c, h = parts[2 * i], parts[2*i + 1]
            state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return start, state

    def regress(x, y):
        with tf.variable_scope('regression') as scope:
            unroll_count = tf.shape(x)[1]
            x = tf.squeeze(x, squeeze_dims=[0])
            y = tf.squeeze(y, squeeze_dims=[0])
            initializer = tf.random_normal_initializer(stddev=0.1)
            w = tf.get_variable('w', [unit_count, dimension_count],
                                initializer=initializer)
            b = tf.get_variable('b', [1, dimension_count])
            y_hat = tf.matmul(x, w) + tf.tile(b, [unroll_count, 1])
            loss = tf.reduce_mean(tf.squared_difference(y_hat, y))
        return y_hat, loss

    return compute

filename = os.path.join(tempfile.gettempdir(), 'learn.pdf')
open(filename, 'w').close()
print('Monitor: {}'.format(filename))
subprocess.call(['open', filename])
support.figure()
y_limit = [-1, 1]

def train_monitor(progress, loss):
    sys.stdout.write('%4d %8d %10d' % progress)
    [sys.stdout.write(' %12.4e' % l) for l in loss]
    sys.stdout.write('\n')

def predict_monitor(y, y_hat):
    pp.clf()
    dimension_count = y.shape[1]
    y_limit[0] = min(y_limit[0], np.min(y), np.min(y_hat))
    y_limit[1] = max(y_limit[1], np.max(y), np.max(y_hat))
    for i in range(dimension_count):
        pp.subplot(dimension_count, 1, i + 1)
        pp.plot(y[:, i])
        pp.plot(y_hat[:, i])
        pp.xlim([0, y.shape[0] - 1])
        pp.ylim(y_limit)
        pp.legend(['Observed', 'Predicted'])
    pp.savefig(filename)

data = support.normalize(support.select(component_ids=[0]))
learn(lambda i: data[i, :],
      dimension_count=data.shape[1],
      sample_count=data.shape[0],
      epoch_count=100,
      train_each=50,
      train_monitor=train_monitor,
      predict_each=10,
      predict_count=100,
      predict_pace=[863890 // 10 - 1000, 1000],
      predict_monitor=predict_monitor)
