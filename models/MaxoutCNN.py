import tensorflow as tf
import os
import shutil
import numpy as np

def load_data():
    train_folder = "data/ETL_data"
    test_folder = "data/test_data"
    files = os.listdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    for f in files:
        for img in os.listdir(train_folder + '/' + f):
            if not os.path.exists(test_folder + '/' + f):
                os.mkdir(test_folder + '/' + f)
            if np.random.rand(1) < 0.2:
                shutil.move(train_folder + '/' + f + '/' + img, test_folder + '/' + f + '/' + img)

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def init_weight(name, input_shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    weight = tf.Variable(initializer(shape=input_shape),name=name)
    return weight

def init_bias(name, input_shape):
    initializer = tf.constant_initializer(value=0.0,dtype=tf.float32)
    bias = tf.Variable(initializer(shape=input_shape),name=name)
    return bias

# parameters
learning_rate = 0.001 #(Adam: 0.001, SGD: 0.01)
training_epochs = 20
batch_size = 100
display_step = 1
# ETL data image of shape 64*63=4032
x = tf.placeholder(tf.float32, [None, 4032], name='input')
# chars recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='label')
# dropout value
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def hidden_conv_maxout():
    w1 = init_weight("weight",[12, 12, 1, 96])
    b1 = init_bias("bias",[96])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # resized_images = tf.image.resize_images(x_image, [32, 32])
    # X_2=tf.reshape(resized_images, [-1, 32, 32, 1])
    h_conv1 = max_out(conv2d(x_image, w1) + b1, 48)

    w2 = init_weight("weight2",[12, 12, 48, 128])
    b2 = init_bias("bias2",[128])

    h_conv2 = max_out(conv2d(h_conv1, w2) + b2, 64)
    # h_conv2=tf.nn.dropout(h_conv2_a,keep_prob)
    w3 = init_weight("weight3", [6, 6, 64, 512])
    b3 = init_bias("bias3", [512])

    h_conv4 = max_out(conv2d(h_conv2, w3) + b3, 128)
    w5 = init_weight("weight5", [1, 1, 128, 40])
    b5 = init_bias("bias5", [40])

    soft_max = tf.reshape(max_out(conv2d(h_conv4, w5) + b5, 10),[-1,10])

    return soft_max

with tf.name_scope('Model'):
    pred = hidden_conv_maxout()
with tf.name_scope('Loss'):
    # cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
with tf.name_scope('Gradient'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle Convolutional Neural Network
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print('Epoch: %d' % (epoch + 1), 'cost = {:.9f}'.format(avg_cost))

    print("test accuracy: %g" % acc.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))