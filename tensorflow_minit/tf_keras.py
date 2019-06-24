
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:41:23 2019

@author: lg
"""

#!/usr/bin/python

#这是一个很经典的cnn 入门教程了
####################

import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

mnist = input_data.read_data_sets("minist_data/", one_hot=True)

with tf.name_scope('input'):

    x = tf.placeholder("float", shape=[None, 784],name='x_input')
    y_ = tf.placeholder("float", shape=[None, 10],name='y_input')



x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1=tf.keras.layers.Conv2D(32,5,strides=1,padding='same')(x_image)
#由于池化层的stride 是2*2，padding=SAME,那么每池化一次，shape降低一半，pool1的shape是14*14*32
h_pool1 =tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')

#输入层pool1层，对pool1层进行一次卷积，pool1层 的shape是14*14*32，conv2的shape也是14*14*64
h_conv2 = tf.layers.conv2d(h_pool1,64,5,strides=1,padding='same')
h_pool2 = tf.layers.max_pooling2d(h_conv2,2,2,padding='same')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#print(h_pool2_flat.shape)
#print(W_fc1.shape)
##h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1 = tf.layers.dense(h_pool2_flat,1024,activation='relu')
print(h_fc1.shape)

#
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#print(h_fc1_drop.shape)
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_conv=tf.layers.dense(h_fc1,10,activation='softmax')

print(y_conv.shape)



cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

pre_num=tf.argmax(y_conv,1,output_type='int64',name="output")#输出节点名：output
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(pre_num, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
keep_prob = tf.placeholder("float")
#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %.3f"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



#
#print ("test accuracy %.3f" % accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    
    

# 保存训练好的模型
#形参output_node_names用于指定输出的节点名称,output_node_names=['output']对应pre_num=tf.argmax(y,1,name="output"),
output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['output'])
with tf.gfile.FastGFile('mnist.pb', mode='wb') as f:#’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
sess.close()
