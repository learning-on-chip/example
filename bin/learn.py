#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from support import Database, Config
import numpy as np
import queue, math, socket, subprocess, support, threading
import tensorflow as tf

class Learn:
    def __init__(self, config):
        graph = tf.Graph()
        with graph.as_default():
            model = Model(config)
            with tf.variable_scope('optimization'):
                epoch = tf.Variable(0, name='epoch', trainable=False)
                increment_epoch = epoch.assign_add(1)
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
            saver = Saver(config)

        self.graph = graph
        self.model = model
        self.epoch = epoch
        self.increment_epoch = increment_epoch
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
        print('Samples: %d' % target.sample_count)
        session = tf.Session(graph=self.graph)
        session.run(self.initialize)
        self.saver.restore(session)
        epoch = session.run(self.epoch)
        epoch_count = config.epoch_count - epoch % config.epoch_count
        for e in range(epoch, epoch + epoch_count):
            self._run_epoch(target, monitor, config, session, e)
            assert(session.run(self.increment_epoch) == e + 1)
            self.saver.save(session)

    def _run_epoch(self, target, monitor, config, session, e):
        for s in range(target.sample_count):
            t = e*target.sample_count + s
            if monitor.should_train(t):
                self._run_train(target, monitor, config, session, e, s, t)
            if monitor.should_predict(t):
                self._run_predict(target, monitor, config, session, e, s, t)

    def _run_train(self, target, monitor, config, session, e, s, t):
        sample = target.compute(s)
        feed = {
            self.model.start: self._zero_start(),
            self.model.x: np.reshape(sample, [1, -1, target.dimension_count]),
            self.model.y: np.reshape(support.shift(sample, -1), [1, -1, target.dimension_count]),
        }
        fetch = {'train': self.train, 'loss': self.model.loss, 'summary': self.summary}
        result = session.run(fetch, feed)
        loss = result['loss'].flatten()
        assert(np.all([not math.isnan(loss) for loss in loss]))
        monitor.train((e, s, t), loss)
        self.logger.add_summary(result['summary'], t)

    def _run_predict(self, target, monitor, config, session, e, s, t):
        sample = target.compute((s + 1) % target.sample_count)
        step_count = sample.shape[0]
        feed = {self.model.start: self._zero_start()}
        fetch = {'y_hat': self.model.y_hat, 'finish': self.model.finish}
        for i in range(step_count):
            feed[self.model.x] = np.reshape(sample[:(i + 1), :], [1, i + 1, -1])
            y_hat = np.zeros([step_count, target.dimension_count])
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
        self.work_schedule = np.cumsum(config.work_schedule)
        self.channels = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._predict_server, daemon=True).start()

    def should_train(self, t):
        return True

    def should_predict(self, t):
        return (len(self.channels) > 0 and
            np.nonzero(self.work_schedule >= (t % self.work_schedule[-1]))[0][0] % 2 == 1)

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

class Saver:
    def __init__(self, config):
        self.backend = tf.train.Saver()
        self.path = config.save_path

    def save(self, session):
        path = self.backend.save(session, self.path)
        print('Saved the model in "{}".'.format(path))

    def restore(self, session):
        if os.path.isfile(self.path):
            if input('Found a model in "{}". Restore? '.format(self.path)) != 'no':
                self.backend.restore(session, self.path)
                print('Restored. Continue learning...')

class Target:
    def __init__(self, config):
        database = Database(config)
        data = database.read()[:, 0]
        partition = database.partition()
        sample_count = partition.shape[0]
        samples, stack = {}, []
        for k in range(sample_count):
            i, j = partition[k]
            samples[k] = data[i:j]
            stack.append(samples[k])
        data = np.concatenate(stack)
        offset, scale = np.mean(data), np.std(data)
        for k in range(sample_count):
            samples[k] = np.reshape((samples[k] - offset) / scale, [-1, 1])
        self.dimension_count = 1
        self.sample_count = sample_count
        self.samples = samples

    def compute(self, k):
        return self.samples[k]

class TestTarget:
    def __init__(self, config):
        self.dimension_count = 1
        self.sample_count = 100000

    def compute(self, k):
        return np.reshape(np.sin(4 * np.pi / 40 * np.arange(0, 40)), [-1, 1])

def main(config):
    learn = Learn(config)
    target = Target(config)
    monitor = Monitor(config)
    learn.run(target, monitor, config)

if __name__ == '__main__':
    database_path = Database.find()
    output_path = os.path.dirname(database_path)
    name = os.path.basename(database_path).replace('.sqlite3', '')
    config = Config({
        'dimension_count': 1,
        'database_path': database_path,
        'layer_count': 1,
        'unit_count': 200,
        'cell_clip': 1.0,
        'forget_bias': 1.0,
        'use_peepholes': True,
        'network_initializer': tf.random_uniform_initializer(-0.01, 0.01),
        'regression_initializer': tf.random_normal_initializer(stddev=0.01),
        'learning_rate': 1e-3,
        'gradient_clip': 1.0,
        'epoch_count': 100,
        'log_path': os.path.join(output_path, 'log'),
        'save_path': os.path.join(output_path, '{}.model'.format(name)),
        'bind_address': ('0.0.0.0', 4242),
        'work_schedule': [1000 - 10, 10],
    })
    main(config)
