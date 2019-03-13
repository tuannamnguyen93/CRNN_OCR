import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import math

with open("sequences.txt", "r") as f:
    seq_labels = pickle.load(f)
with open("sequences_features.txt", "r") as f2:
    seq_features = pickle.load(f2)

def get_data(a,b):
    list1 = []
    list2 = []
    for i in range(a, b):
        list1.append(seq_features["Sequence_" + str(i) + ".png"])
        list2.append(seq_labels["Sequence_" + str(i) + ".png"])

    arr1 = np.asarray(list1, dtype=np.float32)
    arr2 = np.asarray(list2, dtype=np.float32)

    return arr1, arr2,np.ones((b-a,1))*30,np.ones((b-a,1))*5

# Parameters# Param
learning_rate = 0.001
batch_size = 128
# Network Parameters
n_input = 32
# n_steps = 32 # timesteps
n_steps = tf.placeholder(tf.int32, [None])
n_hidden = 128 # hidden layer num of features
n_classes = 72
n_epoch = 10

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# SparseTensor placeholder required by ctc_loss op.
y = tf.sparse_placeholder(tf.int32)
# y = tf.placeholder("float", [None, n_classes])

# Define weights
# weight = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
# bias = tf.Variable(tf.random_normal([n_classes]))
weight = tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1), name='weights')
bias = tf.Variable(tf.constant(0., shape=[n_classes]), name='bias')

def BLSTM(x, weights, biases):
    # x = tf.transpose(x, [1, 0, 2])
    # # Reshape to (n_steps*batch_size, n_input)
    # x = tf.reshape(x, [-1, n_input])
    # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.split(x, n_steps)

    # # Define lstm cells with tensorflow

    # # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights) + biases

    # def lstm_cell():
    #     return tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)

    # # Stacking rnn cells.
    # stack = tf.contrib.rnn.MultiRNNCell(
    #     [lstm_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True)

    # # Creates a recurrent neural network.
    # outputs, _ = tf.nn.dynamic_rnn(stack, inputs_placeholder, sequence_length_placeholder, dtype=tf.float32)

    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    shape = tf.shape(x)
    batch_size, time_steps = shape[0], shape[1]

    # Reshaping to apply the same weights over the time steps.
    outputs = tf.reshape(outputs, [-1, n_hidden])

    # Doing the affine projection.
    logits = tf.matmul(outputs, weights) + bias

    # Reshaping back to the original shape.
    logits = tf.reshape(logits, [batch_size, -1, n_classes])

    # Time is major.
    logits = tf.transpose(logits, (1, 0, 2))

    return logits

with tf.name_scope('Model'):
    pred = BLSTM(x, weight, bias)
with tf.name_scope('Loss'):
    loss = tf.nn.ctc_loss(y, pred, n_steps)
    cost = tf.reduce_mean(loss)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
with tf.name_scope('Gradient'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(cost)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

# CTC decoder.
decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(pred, n_steps)
label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))

with tf.Session() as sess:
    sess.run(init)

    # Saver op to save and restore all the variables.
    saver = tf.train.Saver()

    # step = 1
    # while step * batch_size < training_iters:
    #     batch_x, batch_y = mnist.train.next_batch(batch_size)
    #     batch_x = batch_x.reshape(batch_size,n_steps,n_input)
    #     sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})
    #     if step % display_step == 0:
    #         loss, accuracy = sess.run([cost, acc], feed_dict={x:batch_x, y:batch_y})
    #         print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
    #               "{:.4f}".format(loss) + ", Training Accuracy= " + \
    #               "{:.4f}".format(accuracy)
    #     step += 1
    # saver = tf.train.Saver()
    # saver.save(sess, './maxout', global_step=1000)
    # print "Testing Accuracy:", sess.run(acc, feed_dict={x: mnist.test.images.reshape((-1, n_steps, n_input)), y: mnist.test.labels})
    for e in n_epoch:
        for step in range(int(80000/batch_size)):
            input, labels, input_length, labels_length = get_data(step * batch_size + 1, (step + 1) * batch_size + 1)
            feed = {x: }

