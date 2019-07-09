#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:03:33 2019


这是一个fm算法的简单实现
@author: lg
"""

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# number of latent factors
from tensorflow.python.framework import graph_util

#读取数据
import pandas as pd

#file_path = './data/tiny_train_input.csv'

file_path = './df.csv'

#读取文件为DataFrame 格式
data2= pd.read_csv(file_path)
#对df的列重新命名
#data2.columns = ['c' + str(i) for i in range(data2.shape[1])]

#考虑到数据比较大，我只用了20000条数据
data1=data2.iloc[:20000]

#c0是label 把它drop掉
train=data1.drop(['c0'],axis=1)

#同样仅仅为了展示，我只用了数据集合的四个特征，特别是在nlp和推荐方面,数据是非常稀疏的
ohe_x1= OneHotEncoder(handle_unknown='ignore')
X1 = ohe_x1.fit_transform(train[['c1']])

ohe_x2= OneHotEncoder(handle_unknown='ignore')
X2 = ohe_x2.fit_transform(train[['c2']])

ohe_x3= OneHotEncoder(handle_unknown='ignore')
X3 = ohe_x3.fit_transform(train[['c3']])

ohe_x4= OneHotEncoder(handle_unknown='ignore')
X4 = ohe_x4.fit_transform(train[['c4']])

import numpy as np
X_TRAIN=np.hstack((X1.toarray(),X2.toarray(),X3.toarray(),X4.toarray()))



k = 100
lr = 0.1
batch_size = 1
reg_l1 = 0.1
reg_l2 = 1
# num of features
p = X_TRAIN.shape[1]
#提取训练的label
label = data1.c0.values
#reshanpe 
label = label.reshape(len(label), 1)

ohe_period = OneHotEncoder(handle_unknown='ignore')

#    X_train_period = ohe_period.fit_transform(y1[['label']])
yy =  ohe_period.fit_transform(data1[['c0']])



#xx=data.values
#
#xx1=xx[:,1:11]

yy1=yy.toarray()




#开始建立模型，P是输入向量的长度，y是softmax二分类
X = tf.placeholder('float32', [None, p])
y = tf.placeholder('float32', [None,2])
keep_prob = tf.placeholder('float32')



#这是一个线性层y1=w1*X+b，可以理解为一个多元线性回归
with tf.variable_scope('linear_layer'):
    b = tf.get_variable('bias', shape=[2],
                        initializer=tf.zeros_initializer())
    w1 = tf.get_variable('w1', shape=[p, 2],
                         initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
    # shape of [None, 2]
    linear_terms = tf.add(tf.matmul(X, w1), b)#y1=w1*X+b


#这是fm的二阶项，v是p行k列的权重矩阵
with tf.variable_scope('interaction_layer'):
    v = tf.get_variable('v', shape=[p, k],
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    # shape of [None, 1]
    #二阶=0.5*((x*v)**2-(x**2)(v**2))
    interaction_terms = tf.multiply(0.5,
                                         tf.reduce_mean(
                                             tf.subtract(
                                                 tf.pow(tf.matmul(X, v), 2),
                                                 tf.matmul(tf.pow(X, 2), tf.pow(v, 2))),
                                             1, keep_dims=True))
                                             
    # shape of [None, 2]

    #y_out=一阶项＋二阶项=y1+interaction_terms
    y_out = tf.add(linear_terms, interaction_terms)
    y_out_prob = tf.nn.softmax(y_out)


cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out_prob),
                                            reduction_indices=[1]))

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out)
mean_loss = tf.reduce_mean(cross_entropy)
loss = mean_loss
tf.summary.scalar('loss', loss)






# Applies exponential decay to learning rate
global_step = tf.Variable(0, trainable=False)
# define optimizer
optimizer = tf.train.FtrlOptimizer(lr, l1_regularization_strength=reg_l1,
                                   l2_regularization_strength=reg_l2)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)




sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(200):

  if i%10 == 0:
      
      print(i)
#    train_accuracy = accuracy.eval(feed_dict={
#        x:batch_xs1, y_: batch_ys, keep_prob: 1.0})
#    print ("step %d, training accuracy %.3f"%(i, train_accuracy))
  train_op.run(feed_dict={X:X_TRAIN , y:yy1, keep_prob: 0.5})

pre_num=tf.argmax(y_out_prob,1,output_type='int32',name="output")#输出节点名：output


output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['output'])
with tf.gfile.FastGFile('fm.pb', mode='wb') as f:#’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
    f.write(output_graph_def.SerializeToString())
sess.close()


'''
    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)



    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()
'''
