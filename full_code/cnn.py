#! /usr/bin/env python2.7

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math

#getting data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#data variables
image_width = 28
image_height = 28
categories = 10
channels = 1 #dimensionality of the color of the image (Grayscale = 1, RGB = 3)

#model variables
kernel = 5
features = [32, 64]
fcneurons = 1024

#Utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def softmax(inp, weights, bias):
    return tf.nn.softmax(tf.matmul(inp, weights) + bias)

def fully_connected(inp, weights, bias, inWidth, inHeight, inFeat, outFeat):
    inp_reshape = tf.reshape(inp, [-1 , inHeight * inWidth * inFeat])
    return tf.nn.relu(tf.matmul(inp_reshape, weights) + bias)

def conv_relu(inp, weight, bias):
    conv = tf.nn.conv2d(inp, weight, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + bias)


# NETWORK DEFINITION

x = tf.placeholder(tf.float32, [None, image_width * image_height])
y_ = tf.placeholder(tf.float32, [None, categories])

x_image = tf.reshape(x, [-1, image_height, image_width, 1])

#first conv layer        
weights_cv1 = weight_variable([kernel, kernel, channels, features[0]])
bias_cv1 = bias_variable([features[0]])
conv1 = conv_relu(x_image, weights_cv1, bias_cv1)

#pool layer
pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#second conv layer
weights_cv2 = weight_variable([kernel, kernel, features[0], features[1]])
bias_cv2 = bias_variable([features[1]])
conv2 = conv_relu(pool, weights_cv2, bias_cv2)

#fully connected layer
weights_fc1 = weight_variable([int(math.ceil(image_height/2.0)) * int(math.ceil(image_width/2.0)) * features[1], fcneurons])
bias_fc1 = bias_variable([fcneurons])
fc1 = fully_connected(conv2, weights_fc1, bias_fc1, int(math.ceil(image_height/2.0)) , int(math.ceil(image_width/2.0)), features[1], fcneurons)

#softmax
weights_sf = weight_variable([fcneurons, categories])
bias_sf = bias_variable([categories])
y_conv = softmax(fc1, weights_sf, bias_sf)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-10), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(5e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#test model
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
