#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:15:25 2018

@author: luogan
"""
'''
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
'''

import tensorflow as tf

import numpy as np
from pymongo import MongoClient
import tensorflow as tf
import pandas as pd
client=MongoClient('localhost',27017)
db=client.mnist.data
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mnist_data
sess=tf.InteractiveSession()



in_units=784
h1_units=500

h2_units=100




w1=tf.Variable(  tf.truncated_normal([in_units,h1_units],stddev=0.1 ) )
b1=tf.Variable(tf.zeros([h1_units]))


w2=tf.Variable(tf.truncated_normal([h1_units,h2_units],stddev=0.1 ))
b2=tf.Variable(tf.zeros([h2_units]))

print('*'*20)

w3=tf.Variable(tf.zeros([h2_units,10]))
b3=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,784])


#x=tf.placeholder(tf.float32,shape=[None,28,28])
#
#x2=tf.reshape(x,[None,784])

keep_prob=tf.placeholder(tf.float32)


hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)




hidden2=tf.nn.relu(tf.matmul(hidden1_drop,w2)+b2)

hidden2_drop=tf.nn.dropout(hidden2,keep_prob)

y2=tf.nn.softmax(tf.matmul(hidden2_drop,w3)+b3)



y_=tf.placeholder(tf.float32,[None,10])

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y2),
                                            reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)





pre_num=tf.argmax(y2,1,output_type='int64',name="output")#输出节点名：output
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(pre_num, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


tf.global_variables_initializer().run()


X,Y=mnist_data.mnist_soft()

import time 
import random
t1=time.time()
#for i in range(1000):
#    cut=100
#    rand=random.randint(cut, 60000)
##    rand=10000
#    down=rand-cut
##    batch_xs,batch_ys=mnist.train.next_batch(1000)
#    batch_xs,batch_ys=X[down:rand],Y[down:rand]
#    
#    batch_xs1=batch_xs.reshape([100,784])
#    train_step.run({x:batch_xs1,y_:batch_ys,keep_prob:1.0})
    
for i in range(2000):
  cut=100
  rand=random.randint(cut, 60000)
#    rand=10000
  down=rand-cut
#    batch_xs,batch_ys=mnist.train.next_batch(1000)
  batch_xs,batch_ys=X[down:rand],Y[down:rand]
  batch_xs1=batch_xs.reshape([100,784])
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch_xs1, y_: batch_ys, keep_prob: 1.0})
    print ("step %d, training accuracy %.3f"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs1, y_: batch_ys, keep_prob: 0.5})

print(time.time()-t1)



print ("test accuracy %.3f" % accuracy.eval(feed_dict={
    x: X[50000:].reshape(10000,784), y_: Y[50000:], keep_prob: 1.0}))




#correct_prediction=tf.equal(tf.argmax(y2,1),tf.argmax(y_,1))
#
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#print(time.time()-t1)
#print(accuracy.eval({x:X[50000:].reshape([10000,784]),y_:Y[50000:],keep_prob:1}))




