#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:17:53 2019

@author: lg
"""

import numpy as np
from pymongo import MongoClient
import tensorflow as tf
import pandas as pd
client=MongoClient('localhost',27017)
db=client.mnist.data
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def save_mnist_mongodb():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    
    x1=x.tolist()
    y1=y.tolist()
    for  p in range(60000): 
        db.save({'data':x1[p],'label':y1[p]})
        if p%100==0:
            print(p)
    return None



def take_mnist():
    ll=list(db.find({}))
    ll1=pd.DataFrame(ll)
    ll2=ll1[['data','label']]
    ll3=ll2.values
    
    bb=[]
    cc=[]
    for n in ll3:
        bb.append(np.array(n[0]))
        cc.append(n[1])
        
    bb1=np.array(bb)
    cc1=np.array(cc)
    return bb1,cc1






def mnist_soft():
    x,y=take_mnist()
    ohe_period = OneHotEncoder(handle_unknown='ignore')

    y1=pd.DataFrame()
    y1['label']=y
    
#    X_train_period = ohe_period.fit_transform(y1[['label']])
    yy =  ohe_period.fit_transform(y1[['label']])
    return x,yy.toarray()
    
    

