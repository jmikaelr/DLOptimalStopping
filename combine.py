import tensorflow as tf
import numpy as np
import time
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.training.moving_averages import assign_moving_average

disable_eager_execution()

def neural_net(x, neurons, is_training, dtype=tf.float32, decay=0.9):
    def batch_normalization(y):
        shape = y.get_shape().as_list()
        y = tf.reshape(y, [-1, shape[1] * shape[2]])
        beta = tf.compat.v1.get_variable(
            name='beta', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.zeros_initializer())
        gamma = tf.compat.v1.get_variable(
            name='gamma', shape=[shape[1] * shape[2]],
            dtype=dtype, initializer=tf.ones_initializer())
        mv_mean = tf.compat.v1.get_variable(
            'mv_mean', [shape[1] * shape[2]],
            dtype=dtype, initializer=tf.zeros_initializer(), trainable=False)
        mv_var = tf.compat.v1.get_variable(
            'mv_var', [shape[1] * shape[2]],
            dtype=dtype, initializer=tf.ones_initializer(), trainable=False)
        mean, variance = tf.nn.moments(y, [0], name='moments')
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_mean, mean, decay, zero_debias=True))
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_var, variance, decay, zero_debias=False))
        mean, variance = tf.cond(is_training, lambda: (mean, variance),
                                 lambda: (mv_mean, mv_var))
        y = tf.nn.batch_normalization(y, mean, variance, beta, gamma, 1e-6)
        return tf.reshape(y, [-1, shape[1], shape[2]])

    def fc_layer(y, out_size, activation, is_single):
        shape = y.get_shape().as_list()
        w = tf.compat.v1.get_variable(
            name='weights', shape=[shape[2], shape[1], out_size],
            dtype=dtype, initializer=tf.initializers.glorot_uniform())
        y = tf.transpose(tf.matmul(tf.transpose(y, [2, 0, 1]), w), [1, 2, 0])
        if is_single:
            b = tf.compat.v1.get_variable(
                name='bias', shape=[out_size, shape[2]],
                dtype=dtype, initializer=tf.zeros_initializer())
            return activation(y + b)
        return activation(batch_normalization(y))

    x = batch_normalization(x)
    for i in range(len(neurons)):
        with tf.compat.v1.variable_scope('layer_' + str(i)):
            x = fc_layer(x, neurons[i], tf.nn.relu if i < len(neurons) - 1 else tf.nn.sigmoid, False)
    return x

def deep_optimal_stopping(x, t, n, g, neurons, batch_size, train_steps, mc_runs, lr_boundaries, lr_values, betal=0.9, beta2=0.999, epsilon=1e-8, decay=0.9):
    is_training = tf.compat.v1.placeholder(tf.bool, [])
    p = g(t, x)
    print(x)
    nets = neural_net(tf.concat([x[:, :, :-1], p[:, :, :-1]], axis=1), neurons, is_training, decay=decay)

    u_list = [nets[:, :, 0]]
    u_sum = u_list[-1]
    for k in range(1, n - 1):
        u_list.append(nets[:, :, k] * (1. - u_sum))
        u_sum += u_list[-1]

    u_list.append(1. - u_sum)
    u_stack = tf.concat(u_list, axis=1)
    p = tf.squeeze(p, axis=1)
    loss = tf.reduce_mean(tf.reduce_sum(-u_stack * p, axis=1))
    idx = tf.argmax(tf.cast(tf.cumsum(u_stack, axis=1) + u_stack >= 1, dtype=tf.uint8), axis=1,
                    output_type=tf.int32)
    stopped_payoffs = tf.reduce_mean(
        tf.gather_nd(p, tf.stack([tf.range(0, batch_size, dtype=tf.int32), idx], axis=1)))

    global_step = tf.Variable(0)
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=betal, beta2=beta2, epsilon=epsilon)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        epoch_times = []
        for _ in range(train_steps):
            start_time = time.time()
            _, current_loss, current_step = sess.run([train_op, loss, global_step],
                                                    feed_dict={is_training: True})
            end_time = time.time()
            epoch_time = start_time - end_time
            epoch_times.append(epoch_time)

            if current_step % 10 == 0:
                print("Step:", current_step, "Current loss:", current_loss)
                print(f"Average Epoch Time: {np.mean(epoch_times)}")

        px_mean = 0.
        for _ in range(mc_runs):
            px_mean += sess.run(stopped_payoffs, feed_dict={is_training: False})

    return px_mean / mc_runs

time_now = time.time()
batch_size = 8192
sigma = 0.2 * np.sqrt(1)
train_steps = 1500
lr_boundaries = [400, 800]
lr_values = [0.05, 0.005, 0.0005]
mc_runs = 1500
d = 2
K = 95
S = 100
r = 0.01
T = 1
N = 50

def g(s, x, k):
    #payoff = tf.maximum(tf.reduce_max(x, axis=1, keepdims=True) - k, 0)
    payoff = tf.maximum(k - tf.reduce_max(x, axis=1, keepdims=True), 0)

    return tf.exp(-r * s) * payoff

tf.compat.v1.reset_default_graph()
t0 = time.time()
Q = np.ones([d, d], dtype=np.float32) * 0.5
np.fill_diagonal(Q, 1.)
L = tf.constant(np.linalg.cholesky(Q).transpose(), dtype = tf.float32)
neurons = [d + 50, d + 50, 1]
W = tf.matmul(tf.compat.v1.random_normal(
shape=[batch_size * N, d],
stddev=np.sqrt(T / N), dtype = tf.float32), L)
W = tf.cumsum(tf.transpose(tf.reshape(
W, [batch_size, N, d]),
[0, 2, 1]), axis=2)
t = tf.constant(np.linspace(
start=T / N, stop=T, num=N,
endpoint=True, dtype=np.float32))
X = tf.exp((r - sigma ** 2 / 2.) * t + sigma * W) * S
start_time = time.time()
px_mean = deep_optimal_stopping(
X, t, N, lambda s, x: g(s, x, K), neurons, batch_size,
train_steps, mc_runs, lr_boundaries, lr_values, epsilon=1e-3)
