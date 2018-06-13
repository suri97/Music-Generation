import numpy as np
import pickle
import os
import tensorflow as tf
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden', type=int,
                    help='Enter num of hidden LSTM cells', default = 512)
parser.add_argument('--time_step', type=int,
                    help='Enter Time Step', default = 10)
parser.add_argument('--lr', type=float,
                    help='Enter Learning Rate', default = 0.001)
parser.add_argument('--nb_epoch', type=int,
                    help='Enter Number of Epochs', default = 100)
parser.add_argument('--d_step', type=int,
                    help='Enter Display Step', default = 1)


args = parser.parse_args()

num_notes = 128
num_vel = 128
n_hidden = args.n_hidden
num_time = 1
time_steps = args.time_step
lr = args.lr
num_epoch = args.nb_epoch
num_total = num_notes + num_vel + num_time
display_step = args.d_step
hidden_layer = 512

print('[i] Number of Epochs:          ', num_epoch)
print('[i] Learning Rate:          ', lr)
print('[i] Number of hidden units:          ', n_hidden)
print('[i] Time Step:          ', time_steps)
print('[i] Display Step:          ', display_step)

with tf.variable_scope('input'):
    note_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_notes])
    note_y = tf.placeholder(dtype=tf.float32, shape=[None, num_notes])
    vel_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_vel])
    vel_y = tf.placeholder(dtype=tf.float32, shape=[None, num_vel])
    time_x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, num_time])
    time_y = tf.placeholder(dtype=tf.float32, shape=[None, num_time])

    input_x = tf.concat( (note_x, vel_x, time_x), axis=2 )
    output_y = tf.concat( (note_y, vel_y, time_y), axis=1 )

with tf.variable_scope('rnn'):
    weights = {
        'hidden_layer': tf.Variable(tf.random_normal( [n_hidden, hidden_layer] )),
        'out': tf.Variable(tf.random_normal( [hidden_layer, num_total] ))
    }
    biases = {
        'hidden_layer': tf.Variable(tf.random_normal( [hidden_layer] )),
        'out': tf.Variable(tf.random_normal([num_total])),
    }

    input = tf.unstack(input_x, time_steps, 1)

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs_rnn, _ = tf.contrib.rnn.static_rnn(rnn_cell, input, dtype=tf.float32)

    outputs_rnn = tf.nn.dropout(outputs_rnn, keep_prob=0.6)

    outputs = tf.matmul( outputs_rnn[-1], weights['hidden_layer'] ) + biases['hidden_layer']

    outputs = tf.nn.dropout( outputs, keep_prob=0.5 )

    pred = tf.matmul(outputs, weights['out']) + biases['out']

    pred_note = pred[:,:num_notes]
    pred_vel = pred[:,num_notes: num_notes + num_vel]
    pred_time = pred[:,-1]

with tf.variable_scope('costs'):
    note_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=note_y, logits=pred_note))
    vel_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=vel_y, logits=pred_vel))
    time_cost = tf.reduce_mean(tf.squared_difference(pred_time, time_y))

with tf.variable_scope('train'):
    opt_note = tf.train.AdamOptimizer(learning_rate=lr).minimize(note_cost)
    opt_vel = tf.train.AdamOptimizer(learning_rate=lr).minimize(vel_cost)
    opt_time = tf.train.AdamOptimizer(learning_rate=lr).minimize(time_cost)

with tf.variable_scope('logging'):
    tf.summary.scalar('note_cost', note_cost)
    tf.summary.scalar('velocity_cost', vel_cost)
    tf.summary.scalar('time_cost', time_cost)
    summary = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

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

            train_writer = tf.summary.FileWriter('./logs/training', sess.graph)
            test_writer = tf.summary.FileWriter('./logs/testing', sess.graph)

            curr = 0
            while curr + time_steps < n_samples:
                n_x, v_x, t_x = Get_Next_X(data['note_train'], data['vel_train'], data['time_train'], curr)
                n_y, v_y, t_y = Get_Next_Y(data['note_train'], data['vel_train'], data['time_train'], curr)

                n_loss, v_loss, t_loss, train_summary = sess.run([note_cost, vel_cost, time_cost,summary], feed_dict={
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

                n_loss, v_loss, t_loss, test_summary = sess.run([note_cost, vel_cost, time_cost, summary], feed_dict={
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

        if epoch % display_step == 0:
            train_writer.add_summary(train_summary, epoch)
            test_writer.add_summary(test_summary, epoch)
            saver.save(sess, './logs/model_' + str(epoch) + '.ckpt')

        print ("Training Note Loss is {:,.6f} & Testing Note Loss is {:,.6f}".format(train_note_loss, test_note_loss))
        print ("Training Velocity Loss is {:,.6f} & Testing Velocity Loss is {:,.6f}".format(train_vel_loss, test_vel_loss))
        print ("Training Time Loss is {:,.6f} & Testing Time Loss is {:,.6f}".format(train_time_loss, test_time_loss))

        print ("-------------------")

    print("Training is Complete")
    save_path = saver.save(sess, './logs/final.ckpt')
    print ("Model saved: {}".format(save_path))