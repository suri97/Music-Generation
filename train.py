import numpy as np
import Process_Data
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_notes = 128
num_vel = 128
n_hidden = 256
num_time = 1
time_steps = 3
lr = 0.001
num_epoch = 1

with tf.variable_scope('input'):
    note_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_notes])
    note_y = tf.placeholder( dtype=tf.int32, shape=[None, num_notes] )
    vel_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_vel])
    vel_y = tf.placeholder(dtype=tf.int32, shape=[None, num_vel])
    time_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_time])
    time_y = tf.placeholder( dtype=tf.float32, shape=[None, num_time] )

with tf.variable_scope('note_pred'):
    weights_note = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_notes]))
    }
    biases_note = {
        'out': tf.Variable(tf.random_normal([num_notes]))
    }

    input_note = tf.unstack(note_x, time_steps, 1)
    #input_note = tf.transpose( note_x, [1, 0, 2] )

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_note , dtype=tf.float32)

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
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_vel , dtype=tf.float32)

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
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, input_time , dtype=tf.float32)

    time_pred = tf.matmul(outputs[-1], weights_time['out']) + biases_time['out']

with tf.variable_scope('costs'):
    note_cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=note_y, logits=note_pred) )
    vel_cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=vel_y, logits=vel_pred) )
    time_cost = tf.reduce_mean( tf.squared_difference(time_pred, time_y) )

with tf.variable_scope('train'):
    opt_note = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(note_cost)
    opt_vel = tf.train.AdamOptimizer( learning_rate=lr ).minimize( vel_cost )
    opt_time = tf.train.AdamOptimizer( learning_rate=lr ).minimize( time_cost )

init = tf.global_variables_initializer()

def Get_Next_X(note, vel, time, curr):
    n = np.array( [ note[curr: curr + time_steps] ] )
    v = np.array( [ vel[ curr: curr + time_steps ] ] )
    t = np.array( [ time[ curr: curr + time_steps ] ] )
    return n,v,t


def Get_Next_Y(note, vel, time, curr):
    n = np.array( [ note[curr + time_steps] ] )
    v = np.array( [ vel[curr + time_steps] ]  )
    t = np.array( [ time[curr + time_steps] ] )
    return n,v,t

path = './mozart/mz_545_3.mid'

with tf.Session() as sess:

    sess.run(init)

    data = Process_Data.Processed_Data(path)

    data['note'] = data['note'][:100]
    data['velocity'] = data['velocity'][:100]
    data['time'] = data['time'][:100]

    n_samples = data['note'].shape[0]

    print("Training has Started")

    for epoch in range(num_epoch):

        curr = 0
        while curr + time_steps < n_samples:

            n_x,v_x,t_x = Get_Next_X( data['note'], data['velocity'], data['time'], curr)
            n_y, v_y, t_y = Get_Next_Y(data['note'], data['velocity'], data['time'], curr)

            sess.run([opt_note, opt_vel, opt_time], feed_dict={
                note_x: n_x,
                note_y: n_y,
                vel_x: v_x,
                vel_y: v_y,
                time_x: t_x,
                time_y: t_y
            })

            curr += 1

        train_acc = 0.0

        curr = 0
        while curr + time_steps < n_samples:
            n_x, v_x, t_x = Get_Next_X(data['note'], data['velocity'], data['time'], curr)
            n_y, v_y, t_y = Get_Next_Y(data['note'], data['velocity'], data['time'], curr)

            train_acc += sess.run(note_cost, feed_dict={
                note_x: n_x,
                note_y: n_y,
                vel_x: v_x,
                vel_y: v_y,
                time_x: t_x,
                time_y: t_y
            })

            curr += 1

        train_acc /= n_samples

        print ("Training Accuracy is ", train_acc)

    print("Training is Complete")