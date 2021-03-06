#! /usr/bin/env python2.7

import tensorflow as tf
import ops

#getting data
test_iris_data, test_iris_labels, train_iris_data, train_iris_labels = ops.get_iris()

#data variables
parameters = 4
categories = 3

#build the model
x = tf.placeholder(tf.float32, [None, parameters])
W = tf.Variable(tf.zeros([parameters, categories]))
b = tf.Variable(tf.zeros([categories]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, categories])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#train model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    sess.run(train_step, feed_dict={x: train_iris_data, y_: train_iris_labels})


#test model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_iris_data,
                                    y_: test_iris_labels}))
