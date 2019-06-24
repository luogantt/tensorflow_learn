#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:55:41 2019

@author: lg
"""
import tensorflow as tf

import numpy as np
from pymongo import MongoClient
import tensorflow as tf
import pandas as pd
import random
client=MongoClient('localhost',27017)
db=client.mnist.data

from mnist_data import take_mnist



model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
#    tf.keras.layers.Dense(784, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


x,y=take_mnist()


#@tf.function
import time

t1=time.time()




def train(model, optimizer):
#  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for p in range(2000):
    step += 1
    
    cut=100
    rand=random.randint(cut, 50000)
#    rand=10000
    down=rand-cut
    loss = train_one_step(model, optimizer, x[down:rand], y[down:rand])
    if tf.equal(step % 100, 0):
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
  return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)

print(time.time()-t1)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())




    

    
    
    

    
    


