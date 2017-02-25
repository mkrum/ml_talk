#! /usr/bin/env python2.7

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn import datasets
from random import shuffle

#getting data
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target

#shuffle data

c = list(zip(iris_data, iris_target))
shuffle(c)
iris_data, iris_target = zip(*c)


#split the data
def convert_to_one_hot(val, categories):
    one_hot = [ 0 for x in range(categories) ]
    one_hot[val] = 1
    return one_hot

train_iris_data = iris_data[:120]
test_iris_data = iris_data[120:]

train_iris_labels = [ convert_to_one_hot(x, 3) for x in iris_target[:120] ]
test_iris_labels = [ convert_to_one_hot(x, 3) for x in iris_target[120:] ]

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
for i in range(1000):
    sess.run(train_step, feed_dict={x: train_iris_data, y_: train_iris_labels})


#test model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_iris_data,
                                    y_: test_iris_labels}))
