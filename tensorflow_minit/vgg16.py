#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:02:22 2019

@author: lg
"""


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
#定以权重变量，初始状态是一个随机数
#https://blog.csdn.net/u013713117/article/details/65446361/
#tf.truncated_normal 截断正太分布，下面函数中随机数取自（0-0.1×标准差，0+0.1×标准差）
#用截断正太分布的原因应该是避免有些奇异值导致某些神经元不工作


mnist = input_data.read_data_sets("minist_data/", one_hot=True)

with tf.name_scope('input'):

    x = tf.placeholder("float", shape=[None, 784],name='x_input')
    y_ = tf.placeholder("float", shape=[None, 10],name='y_input')





#数据的输入是28*28的矩阵

x_image = tf.reshape(x, [-1, 28, 28, 1])

#在这个函数下,W_conv1 = weight_variable([5, 5, 1, 32])
#表示卷积核的大小是5*5，因为图像是灰度图只有一个通道，32表示有32个卷积核
#对于stride=1*1的卷积，并且padding=SAME,那么卷积后的图像和卷积前的图像，有相同的shape,conv1的shape是28*28*32
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_conv1=tf.layers.conv2d(x_image,64,5,strides=1,padding='same')
#由于池化层的stride 是2*2，padding=SAME,那么每池化一次，shape降低一半，pool1的shape是14*14*32
h_conv1=tf.layers.conv2d(h_conv1,64,5,strides=1,padding='same')
h_pool1 =tf.layers.max_pooling2d(h_conv1,2,2,padding='same')



#输入层pool1层，对pool1层进行一次卷积，pool1层 的shape是14*14*32，conv2的shape也是14*14*64
h_conv2 = tf.layers.conv2d(h_pool1,128,5,strides=1,padding='same')

h_pool2 = tf.layers.max_pooling2d(h_conv2,2,2,padding='same')


h_conv3 = tf.layers.conv2d(h_pool2,256,5,strides=1,padding='same')

h_pool3 = tf.layers.max_pooling2d(h_conv3,2,2,padding='same')

h_conv3 = tf.layers.conv2d(h_pool3,512,5,strides=1,padding='same')

h_pool3 = tf.layers.max_pooling2d(h_conv3,2,2,padding='same')

#h_pool2_flat = tf.reshape(h_pool3, [-1, 256])

h_pool2_flat =tf.layers.flatten(h_pool3)



h_fc1 = tf.layers.dense(h_pool2_flat,512,activation='relu')
print(h_fc1.shape)



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
  if i%10 == 0:
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
