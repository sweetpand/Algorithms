#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import tensorflow as tf
import numpy as np

def random_normal(shape):
  return (np.random.random(shape) - 0.5) * 2

input_size = 2
hidden_size = 16
output_size = 1

x = tf.placeholder(dtype=tf.float32, name="X")
y = tf.placeholder(dtype=tf.float32, name="Y")

W1 = tf.Variable(random_normal((input_size, hidden_size)), dtype=tf.float32, name="W1")
W2 = tf.Variable(random_normal((hidden_size, output_size)), dtype=tf.float32, name="W2")

b1 = tf.Variable(random_normal(hidden_size), dtype=tf.float32, name="b1")
b2 = tf.Variable(random_normal(output_size), dtype=tf.float32, name="b2")

l1 = tf.sigmoid(tf.add(tf.matmul(x, W1), b1), name="l1")
result = tf.sigmoid(tf.add(tf.matmul(l1, W2), b2), name="l2")  # Note: works much better without sigmoid

r_squared = tf.square(result - y)
loss = tf.reduce_mean(r_squared)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

train_x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]]).reshape((4, 2))
train_y = np.array([1, 1, 0, 0]).reshape((4, 1))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for itr in range(10000):
    _, loss_val = sess.run([train, loss], {x: train_x, y: train_y})
    if itr % 100 == 0:
      prediction = sess.run(result, {x: [[1, 0]]})
      print('Epoch %d done. Loss=%.6f Prediction=%.6f' % (itr, loss_val, prediction))
print 'TensorFlow'
