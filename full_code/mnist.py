#! /usr/bin/env python2.7

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#getting data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#data variables
image_width = 28
image_height = 28
categories = 10


#build the model
x = tf.placeholder(tf.float32, [None, image_width * image_height])
W = tf.Variable(tf.zeros([image_width * image_height, categories]))
b = tf.Variable(tf.zeros([categories]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, categories])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#train model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#test model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
