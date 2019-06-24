#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:26:32 2019

@author: lg
"""

#!/usr/bin/python

#这是一个很经典的cnn 入门教程了


import tensorflow as tf
import sys

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#定以权重变量，初始状态是一个随机数
#https://blog.csdn.net/u013713117/article/details/65446361/
#tf.truncated_normal 截断正太分布，下面函数中随机数取自（0-0.1×标准差，0+0.1×标准差）
#用截断正太分布的原因应该是避免有些奇异值导致某些神经元不工作

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#bias 采用0.1的常数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#二维卷积，x是输入，W是卷积核的参数，
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

(x_train, y_train),(x_test, y_test) = mnist.load_data()

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32])

#数据的输入是28*28的矩阵

x_image = tf.reshape(x, [-1, 28, 28, 1])

#在这个函数下,W_conv1 = weight_variable([5, 5, 1, 32])
#表示卷积核的大小是5*5，因为图像是灰度图只有一个通道，32表示有32个卷积核
#对于stride=1*1的卷积，并且padding=SAME,那么卷积后的图像和卷积前的图像，有相同的shape,conv1的shape是28*28*32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#由于池化层的stride 是2*2，padding=SAME,那么每池化一次，shape降低一半，pool1的shape是14*14*32
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])


#输入层pool1层，对pool1层进行一次卷积，pool1层 的shape是14*14*32，conv2的shape也是14*14*64
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Now image size is reduced to 7*7

#pool2的shape是7*7*64
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

print(h_pool2_flat.shape)
print(W_fc1.shape)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

print(h_fc1.shape)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.shape)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print(y_conv.shape)



cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

num=int(x_train.shape[0]/100)

for i in range(200):
    for j in range(num):
      train=x_train[j*100:j*100+100]   
      ty=y_train[j*100:j*100+100]   
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:train, y_: ty, keep_prob: 1.0})
        print ("step %d,  accuracy %.3f"%(i, train_accuracy))
      train_step.run(feed_dict={x: train, y_:ty , keep_prob: 0.5})

print ("Training finished")

print ("test accuracy %.3f" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
