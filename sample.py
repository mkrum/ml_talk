#! /usr/bin/env python2.7

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import ops

#getting data
test_iris_data, test_iris_labels, train_iris_data, train_iris_labels = ops.get_iris()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#data variables

#build the model

#train model

#test model
