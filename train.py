import numpy as np
import pickle
import os
import tensorflow as tf
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden', type=int,
                    help='Enter num of hidden LSTM cells', default = 256)
parser.add_argument('--time_step', type=int,
                    help='Enter Time Step', default = 10)
parser.add_argument('--lr', type=float,
                    help='Enter Learning Rate', default = 0.001)
parser.add_argument('--nb_epoch', type=int,
                    help='Enter Number of Epochs', default = 100)


args = parser.parse_args()

num_notes = 128
num_vel = 128
n_hidden = args.n_hidden
num_time = 1
time_steps = args.time_step
lr = args.lr
num_epoch = args.nb_epoch

print('[i] Number of Epochs:          ', num_epoch)
print('[i] Learning Rate:          ', lr)
print('[i] Number of hidden units:          ', n_hidden)
print('[i] Time Step:          ', time_steps)

with tf.variable_scope('input'):
    note_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_notes])
    note_y = tf.placeholder(dtype=tf.int32, shape=[None, num_notes])
    vel_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_vel])
    vel_y = tf.placeholder(dtype=tf.int32, shape=[None, num_vel])
    time_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_time])
    time_y = tf.placeholder(dtype=tf.float32, shape=[None, num_time])

with tf.variable_scope('note_pred'):
    weights_note = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_notes]))
    }
    biases_note = {
        'out': tf.Variable(tf.random_normal([num_notes]))
    }

    input_note = tf.unstack(note_x, time_steps, 1)
    # input_note = tf.transpose( note_x, [1, 0, 2] )

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_note, dtype=tf.float32)

    note_pred = tf.matmul(outputs[-1], weights_note['out']) + biases_note['out']

with tf.variable_scope('vel_pred'):
    weights_vel = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_vel]))
    }
    biases_vel = {
        'out': tf.Variable(tf.random_normal([num_vel]))
    }

    input_vel = tf.unstack(vel_x, time_steps, 1)

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_vel, dtype=tf.float32)

    vel_pred = tf.matmul(outputs[-1], weights_vel['out']) + biases_vel['out']

with tf.variable_scope('time_pred'):
    weights_time = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_time]))
    }
    biases_time = {
        'out': tf.Variable(tf.random_normal([num_time]))
    }

    input_time = tf.unstack(time_x, time_steps, 1)

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_time, dtype=tf.float32)

    time_pred = tf.matmul(outputs[-1], weights_time['out']) + biases_time['out']

with tf.variable_scope('costs'):
    note_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=note_y, logits=note_pred))
    vel_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=vel_y, logits=vel_pred))
    time_cost = tf.reduce_mean(tf.squared_difference(time_pred, time_y))

with tf.variable_scope('train'):
    opt_note = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(note_cost)
    opt_vel = tf.train.AdamOptimizer(learning_rate=lr).minimize(vel_cost)
    opt_time = tf.train.AdamOptimizer(learning_rate=lr).minimize(time_cost)

init = tf.global_variables_initializer()


def Get_Next_X(note, vel, time, curr):
    n = np.array([note[curr: curr + time_steps]])
    v = np.array([vel[curr: curr + time_steps]])
    t = np.array([time[curr: curr + time_steps]])
    return n, v, t


def Get_Next_Y(note, vel, time, curr):
    n = np.array([note[curr + time_steps]])
    v = np.array([vel[curr + time_steps]])
    t = np.array([time[curr + time_steps]])
    return n, v, t


train_dir = './Training_Data/'
pickle_files = os.listdir(train_dir)

with tf.Session() as sess:
    sess.run(init)

    print("Training has Started")
    print ("--------------------")

    for epoch in range(num_epoch):

        print ("Epoch {}:".format(epoch + 1))

        train_note_loss = 0.0
        train_vel_loss = 0.0
        train_time_loss = 0.0
        tot_samples_train = 0

        test_note_loss = 0.0
        test_vel_loss = 0.0
        test_time_loss = 0.0
        tot_samples_test = 0

        for pkl in pickle_files:
            if pkl[-3:] != 'pkl':
                continue

            with open(train_dir + pkl, 'rb') as f:
                data = pickle.load(f)

            n_samples = data['note_train'].shape[0]

            curr = 0
            while curr + time_steps < n_samples:
                n_x, v_x, t_x = Get_Next_X(data['note_train'], data['vel_train'], data['time_train'], curr)
                n_y, v_y, t_y = Get_Next_Y(data['note_train'], data['vel_train'], data['time_train'], curr)

                sess.run([opt_note, opt_vel, opt_time], feed_dict={
                    note_x: n_x,
                    note_y: n_y,
                    vel_x: v_x,
                    vel_y: v_y,
                    time_x: t_x,
                    time_y: t_y
                })

                curr += 1

            tot_samples_train += curr

            curr = 0
            while curr + time_steps < n_samples:
                n_x, v_x, t_x = Get_Next_X(data['note_train'], data['vel_train'], data['time_train'], curr)
                n_y, v_y, t_y = Get_Next_Y(data['note_train'], data['vel_train'], data['time_train'], curr)

                n_loss, v_loss, t_loss = sess.run([note_cost, vel_cost, time_cost], feed_dict={
                    note_x: n_x,
                    note_y: n_y,
                    vel_x: v_x,
                    vel_y: v_y,
                    time_x: t_x,
                    time_y: t_y
                })

                train_note_loss += n_loss
                train_vel_loss += v_loss
                train_time_loss += t_loss

                curr += 1



            n_samples = data['note_test'].shape[0]

            curr = 0
            while curr + time_steps < n_samples:
                n_x, v_x, t_x = Get_Next_X(data['note_test'], data['vel_test'], data['time_test'], curr)
                n_y, v_y, t_y = Get_Next_Y(data['note_test'], data['vel_test'], data['time_test'], curr)

                n_loss, v_loss, t_loss = sess.run([note_cost, vel_cost, time_cost], feed_dict={
                    note_x: n_x,
                    note_y: n_y,
                    vel_x: v_x,
                    vel_y: v_y,
                    time_x: t_x,
                    time_y: t_y
                })

                test_note_loss += n_loss
                test_vel_loss += v_loss
                test_time_loss += t_loss

                curr += 1

            tot_samples_test += curr

    train_note_loss /= tot_samples_train
    train_vel_loss /= tot_samples_train
    train_time_loss /= tot_samples_train

    test_note_loss /= tot_samples_test
    test_vel_loss /= tot_samples_test
    test_time_loss /= tot_samples_test

    print ("Training Note Loss is {:,.6f} & Testing Note Loss is {:,.6f}".format(train_note_loss, test_note_loss))
    print ("Training Velocity Loss is {:,.6f} & Testing Velocity Loss is {:,.6f}".format(train_vel_loss, test_vel_loss))
    print ("Training Time Loss is {:,.6f} & Testing Time Loss is ".format(train_time_loss, test_time_loss))

    print ("-------------------")

    print("Training is Complete")
