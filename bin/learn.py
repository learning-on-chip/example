#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import queue, socket, subprocess, support, threading
import tensorflow as tf

class Config:
    def __init__(self, options={}):
        self.layer_count = 1
        self.unit_count = 20
        self.learning_rate = 1e-2
        self.gradient_norm = 1.0
        self.forget_bias = 0.0
        self.use_peepholes = True
        self.network_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.regression_initializer = tf.random_normal_initializer(stddev=0.1)
        for key in options:
            setattr(self, key, options[key])

class Learn:
    def __init__(self, config):
        graph = tf.Graph()
        with graph.as_default():
            model = Model(config)
            with tf.variable_scope('optimization'):
                parameters = tf.trainable_variables()
                gradient = tf.gradients(model.loss, parameters)
                gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_norm)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                train = optimizer.apply_gradients(zip(gradient, parameters))
            with tf.variable_scope('summary'):
                tf.scalar_summary('log_loss', tf.log(tf.reduce_sum(model.loss)))
            logger = tf.train.SummaryWriter('log', graph)
            summary = tf.merge_all_summaries()
            initialize = tf.initialize_variables(tf.all_variables(), name='initialize')

        self.graph = graph
        self.model = model
        self.parameters = parameters
        self.train = train
        self.logger = logger
        self.summary = summary
        self.initialize = initialize

    def count_parameters(self):
        return np.sum([int(np.prod(p.get_shape())) for p in self.parameters])

    def run(self, target, config):
        config.sample_count -= config.predict_count
        config.sample_count -= config.sample_count % config.predict_each
        config.predict_phases = np.cumsum(config.predict_phases)

        print('Parameters: %d' % self.count_parameters())
        print('Epoch samples: %d' % config.sample_count)

        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        for epoch in range(config.epoch_count):
            self._run_epoch(target, config, session, epoch)

    def _run_epoch(self, target, config, session, epoch):
        model = self.model
        train_fetches = {
            'finish': model.finish,
            'train': self.train,
            'loss': model.loss,
            'summary': self.summary,
        }
        train_feeds = {
            model.start: np.zeros(model.start.get_shape(), np.float32),
            model.x: np.zeros([1, config.train_each, config.dimension_count], np.float32),
            model.y: np.zeros([1, config.train_each, config.dimension_count], np.float32),
        }
        predict_fetches = {'finish': model.finish, 'y_hat': model.y_hat}
        predict_feeds = {model.start: None, model.x: None}
        y = np.zeros([config.predict_count, config.dimension_count])
        y_hat = np.zeros([config.predict_count, config.dimension_count])
        for s, t in zip(range(config.sample_count - 1), range(1, config.sample_count)):
            train_feeds[model.x] = np.roll(train_feeds[model.x], -1, axis=1)
            train_feeds[model.y] = np.roll(train_feeds[model.y], -1, axis=1)
            train_feeds[model.x][0, -1, :] = target(s)
            train_feeds[model.y][0, -1, :] = target(t)

            if t % config.train_each == 0:
                total_sample_count = epoch*config.sample_count + t
                total_train_count = total_sample_count // config.train_each
                train_results = session.run(train_fetches, train_feeds)
                train_feeds[model.start] = train_results['finish']
                config.monitor.train((epoch, total_train_count, total_sample_count),
                                     train_results['loss'].flatten())
                self.logger.add_summary(train_results['summary'], total_train_count)

            phase = config.predict_phases >= (s % config.predict_phases[-1])
            phase = np.nonzero(phase)[0][0]
            if phase % 2 == 1 and t % config.predict_each == 0:
                lag = t % config.train_each
                predict_feeds[model.start] = train_feeds[model.start]
                y_tail = train_feeds[model.y][0, (config.train_each - 1 - lag):, :]
                predict_feeds[model.x] = np.reshape(y_tail, [1, 1 + lag, -1])
                for i in range(config.predict_count):
                    predict_results = session.run(predict_fetches, predict_feeds)
                    predict_feeds[model.start] = predict_results['finish']
                    y_hat[i, :] = predict_results['y_hat'][-1, :]
                    predict_feeds[model.x] = np.reshape(y_hat[i, :], [1, 1, -1])
                    y[i, :] = target(t + i + 1)
                config.monitor.predict(y, y_hat)

class Model:
    def __init__(self, config):
        x = tf.placeholder(tf.float32, [1, None, config.dimension_count], name='x')
        y = tf.placeholder(tf.float32, [1, None, config.dimension_count], name='y')
        with tf.variable_scope('network') as scope:
            cell = tf.nn.rnn_cell.LSTMCell(config.unit_count, state_is_tuple=True,
                                           forget_bias=config.forget_bias,
                                           use_peepholes=config.use_peepholes,
                                           initializer=config.network_initializer)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.layer_count, state_is_tuple=True)
            start, state = Model._initialize(config)
            h, state = tf.nn.dynamic_rnn(cell, x, initial_state=state, parallel_iterations=1)
            finish = Model._finalize(state, config)
        y_hat, loss = Model._regress(h, y, config)

        self.x = x
        self.y = y
        self.y_hat = y_hat
        self.loss = loss
        self.start = start
        self.finish = finish

    def _finalize(state, config):
        parts = []
        for i in range(config.layer_count):
            parts.append(state[i].c)
            parts.append(state[i].h)
        return tf.pack(parts, name='finish')

    def _initialize(config):
        start = tf.placeholder(tf.float32, [2 * config.layer_count, 1, config.unit_count],
                               name='start')
        parts = tf.unpack(start)
        state = []
        for i in range(config.layer_count):
            c, h = parts[2 * i], parts[2*i + 1]
            state.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return start, tuple(state)

    def _regress(x, y, config):
        with tf.variable_scope('regression') as scope:
            unroll_count = tf.shape(x)[1]
            x = tf.squeeze(x, squeeze_dims=[0])
            y = tf.squeeze(y, squeeze_dims=[0])
            w = tf.get_variable('w', [config.unit_count, config.dimension_count],
                                initializer=config.regression_initializer)
            b = tf.get_variable('b', [1, config.dimension_count])
            y_hat = tf.matmul(x, w) + tf.tile(b, [unroll_count, 1])
            loss = tf.reduce_mean(tf.squared_difference(y_hat, y))
        return y_hat, loss

class Monitor:
    def __init__(self, address=('0.0.0.0', 4242)):
        self.address = address
        self.channels = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._predict_server).start()

    def train(self, progress, loss):
        sys.stdout.write('%4d %8d %10d' % progress)
        [sys.stdout.write(' %12.4e' % loss) for loss in loss]
        sys.stdout.write('\n')

    def predict(self, y, y_hat):
        self.lock.acquire()
        try:
            for channel in self.channels:
                channel.put((y, y_hat))
        finally:
            self.lock.release()

    def _predict_client(self, connection, address):
        print('Start serving {}.'.format(address))
        channel = queue.Queue()
        self.lock.acquire()
        try:
            self.channels[channel] = True
        finally:
            self.lock.release()
        try:
            client = connection.makefile(mode="w")
            while True:
                y, y_hat = channel.get()
                row = np.concatenate((y.flatten(), y_hat.flatten()))
                line = ','.join(['%.16e' % value for value in row]) + '\n'
                client.write(line)
        except Exception as e:
            print('Stop serving {} ({}).'.format(address, e))
        self.lock.acquire()
        try:
            del self.channels[channel]
        finally:
            self.lock.release()

    def _predict_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(1)
        print('Listening to {}...'.format(self.address))
        while True:
            try:
                connection, address = server.accept()
                threading.Thread(target=self._predict_client, args=(connection, address)).start()
            except Exception as e:
                print('Encountered a problem ({}).'.format(e))

def main():
    data = np.reshape(support.normalize(support.select(components=[0])[:, 1]), [-1, 1])
    config = Config({
        'dimension_count': data.shape[1],
        'sample_count': data.shape[0],
        'epoch_count': 100,
        'train_each': 50,
        'predict_each': 5,
        'predict_count': 100,
        'predict_phases': [10000 - 1000, 1000],
        'monitor': Monitor(),
    })
    learn = Learn(config)
    learn.run(lambda i: data[i, :], config)

if __name__ == '__main__':
    main()
