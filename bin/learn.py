#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.dirname(__file__))

from database import Database
import numpy as np
import queue, math, socket, subprocess, support, threading
import tensorflow as tf

class Config:
    def __init__(self, options={}):
        self.layer_count = 1
        self.unit_count = 200
        self.cell_clip = 1.0
        self.forget_bias = 1.0
        self.use_peepholes = True
        self.network_initializer = tf.random_uniform_initializer(-0.01, 0.01)
        self.regression_initializer = tf.random_normal_initializer(stddev=0.01)
        self.learning_rate = 1e-3
        self.gradient_clip = 1.0
        self.epoch_count = 100
        self.bind_address = ('0.0.0.0', 4242)
        self.schedule = [1000 - 10, 10]
        self.log_path = 'output/log'
        self.save_path = 'output/model.ckpt'
        self.update(options)

    def update(self, options):
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
                gradient, _ = tf.clip_by_global_norm(gradient, config.gradient_clip)
                optimizer = tf.train.AdamOptimizer(config.learning_rate)
                train = optimizer.apply_gradients(zip(gradient, parameters))
            with tf.variable_scope('summary'):
                tf.scalar_summary('log_loss', tf.log(tf.reduce_sum(model.loss)))
            logger = tf.train.SummaryWriter(config.log_path, graph)
            summary = tf.merge_all_summaries()
            initialize = tf.initialize_variables(tf.all_variables(), name='initialize')
            saver = tf.train.Saver()

        self.graph = graph
        self.model = model
        self.parameters = parameters
        self.train = train
        self.logger = logger
        self.summary = summary
        self.initialize = initialize
        self.saver = saver

    def count_parameters(self):
        return np.sum([int(np.prod(parameter.get_shape())) for parameter in self.parameters])

    def run(self, target, monitor, config):
        print('Parameters: %d' % self.count_parameters())
        print('Samples: %d' % config.sample_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        if os.path.isfile(config.save_path):
            if input('Found a model in "{}". Restore? '.format(config.save_path)) != 'no':
                self.saver.restore(session, config.save_path)
                print('Restored. Continue learning...')
        for e in range(config.epoch_count):
            self._run_epoch(target, monitor, config, session, e)
            path = self.saver.save(session, config.save_path)
            print('Saved the model in "{}".'.format(path))

    def _run_epoch(self, target, monitor, config, session, e):
        for s in range(config.sample_count):
            t = e*config.sample_count + s
            if monitor.should_train(t):
                self._run_train(target, monitor, config, session, e, s, t)
            if monitor.should_predict(t):
                self._run_predict(target, monitor, config, session, e, s, t)

    def _run_train(self, target, monitor, config, session, e, s, t):
        sample = target.compute(s)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, config.dimension_count]),
            self.model.y: np.reshape(support.shift(sample, -1), [1, -1, config.dimension_count]),
        }
        fetch = {'train': self.train, 'loss': self.model.loss, 'summary': self.summary}
        result = session.run(fetch, feed)
        loss = result['loss'].flatten()
        assert(np.all([not math.isnan(loss) for loss in loss]))
        monitor.train((e, s, t), loss)
        self.logger.add_summary(result['summary'], t)

    def _run_predict(self, target, monitor, config, session, e, s, t):
        sample = target.compute((s + 1) % config.sample_count)
        step_count = sample.shape[0]
        feed = {self.model.start: self._zero_start()}
        fetch = {'y_hat': self.model.y_hat, 'finish': self.model.finish}
        for i in range(step_count):
            feed[self.model.x] = np.reshape(sample[:(i + 1), :], [1, i + 1, -1])
            y_hat = np.zeros([step_count, config.dimension_count])
            for j in range(step_count - i - 1):
                result = session.run(fetch, feed)
                feed[self.model.start] = result['finish']
                y_hat[j, :] = result['y_hat'][-1, :]
                feed[self.model.x] = np.reshape(y_hat[j, :], [1, 1, -1])
            if not monitor.predict(support.shift(sample, -i - 1), y_hat):
                break

    def _zero_start(self):
        return np.zeros(self.model.start.get_shape(), np.float32)

class Model:
    def __init__(self, config):
        x = tf.placeholder(tf.float32, [1, None, config.dimension_count], name='x')
        y = tf.placeholder(tf.float32, [1, None, config.dimension_count], name='y')
        with tf.variable_scope('network') as scope:
            cell = tf.nn.rnn_cell.LSTMCell(config.unit_count,
                                           state_is_tuple=True,
                                           cell_clip=config.cell_clip,
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
    def __init__(self, config):
        self.bind_address = config.bind_address
        self.schedule = np.cumsum(config.schedule)
        self.channels = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._predict_server, daemon=True).start()

    def should_train(self, t):
        return True

    def should_predict(self, t):
        return (len(self.channels) > 0 and
            np.nonzero(self.schedule >= (t % self.schedule[-1]))[0][0] % 2 == 1)

    def train(self, progress, loss):
        sys.stdout.write('%4d %10d %10d' % progress)
        [sys.stdout.write(' %12.4e' % loss) for loss in loss]
        sys.stdout.write('\n')

    def predict(self, y, y_hat):
        self.lock.acquire()
        try:
            for channel in self.channels:
                channel.put((y, y_hat))
        finally:
            self.lock.release()
        return len(self.channels) > 0

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
                client.write(','.join([str(value) for value in y.flatten()]) + ',')
                client.write(','.join([str(value) for value in y_hat.flatten()]) + '\n')
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
        server.bind(self.bind_address)
        server.listen(1)
        print('Listening to {}...'.format(self.bind_address))
        while True:
            try:
                connection, address = server.accept()
                threading.Thread(target=self._predict_client, daemon=True,
                                 args=(connection, address)).start()
            except Exception as e:
                print('Encountered a problem ({}).'.format(e))

class Target:
    def __init__(self, config):
        self.database = Database(config.component_id)
        self.quantity = config.quantity
        self.normalization = config.normalization
        self.reduction = config.reduction
        self.partition = self.database.partition()
        self.dimension_count = 1
        self.sample_count = self.partition.shape[0]
        self.cache = {}

    def compute(self, k):
        if k in self.cache:
            return self.cache[k]
        print('Reading sample {}…'.format(k))
        i, j = self.partition[k]
        sample = self.database.read(i, j, quantity=self.quantity)
        sample = (sample - self.normalization.mean) / self.normalization.deviation
        length = len(sample)
        result = np.zeros([int(math.ceil(length / self.reduction)), 1])
        for i in range(result.shape[0]):
            j = i * self.reduction
            l = j + self.reduction
            result[i, 0] = np.sum(sample[j:min(l, length)]) / self.reduction
        self.cache[k] = result
        return self.cache[k]

class TestTarget:
    def __init__(self, config):
        self.dimension_count = 1
        self.sample_count = 100000

    def compute(self, k):
        return np.reshape(np.sin(4 * np.pi / 40 * np.arange(0, 40)), [-1, 1])

def main():
    config = Config()
    monitor = Monitor(config)
    config.update({
        'component_id': 0,
        'quantity': 'temperature',
        'normalization': Config({
            'mean': 320.72,
            'deviation': 3.45,
        }),
        'reduction': 100,
    })
    target = Target(config)
    config.update({
        'dimension_count': target.dimension_count,
        'sample_count': target.sample_count,
    })
    learn = Learn(config)
    learn.run(target, monitor, config)

if __name__ == '__main__':
    main()
