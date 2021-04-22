# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:34:33 2021

@author: jeffa


To Do: try adding output at t-2

"""




from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from numpy import array, sqrt, array
from numpy.random import uniform
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import time 
import numpy as np
import pandas as pd

start = time.time()


with tf.device('/CPU:0'):
    dat = pd.read_csv('data_set2_013121_1100.txt')
    Q1 = dat[' Heater 1 (%)'].values
    Q2 = dat[' Heater 2 (%)'].values
    T1 = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
    T2 = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]
    
    xtrain = np.hstack((Q1.reshape(len(Q1),1),Q2.reshape(len(Q2),1)))
    ytrain = np.hstack((T1.reshape(len(T1),1),T2.reshape(len(T2),1)))
    
    dat = pd.read_csv('data_set1_013021_1930.txt')
    Time = dat['Time (sec)'].values
    U1 = dat[' Heater 1 (%)'].values
    U2 = dat[' Heater 2 (%)'].values
    T1 = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
    T2 = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]
    
    xtest = np.hstack((U1.reshape(len(U1),1),U2.reshape(len(U2),1)))
    ytest = np.hstack((T1.reshape(len(T1),1),T2.reshape(len(T2),1)))
    
    print("xtrain:", xtrain.shape, "ytrain:", ytrain.shape)
    
    def convertData(datax,datay,step):
        X, Y = [], []
        for i in range(len(datax)-step):
            d = i+step  
            X.append(np.hstack((datay[i:d], datax[i:d,])))
            Y.append(datay[d])
    
        return array(X), array(Y)
    
    step=1
    testx,testy = convertData(xtest,ytest, step)
    trainx,trainy = convertData(xtrain,ytrain, step)
    
    #%%
    
    print("test-x:", testx.shape, "test-y:", testy.shape)
    print("train-x:", trainx.shape, "trian-y:", trainy.shape)
    
    in_dim = trainx.shape[1:3]
    out_dim = trainy.shape[1]
    
    model = Sequential()
    model.add(SimpleRNN(units=64, input_shape=in_dim, activation="relu"))
    model.add(Dense(16, activation="relu")) 
    model.add(Dense(out_dim))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    
    model.fit(trainx,trainy, epochs=300, verbose=2)
    trainScore = model.evaluate(trainx, trainy, verbose=0)
    print(trainScore)
    
    predtest= model.predict(testx)
    
    rmse_y1 = sqrt(mean_squared_error(testy[:,0], predtest[:,0]))
    rmse_y2 = sqrt(mean_squared_error(testy[:,1], predtest[:,1]))
    print("RMSE y1: %.2f y2: %.2f" % (rmse_y1, rmse_y2))

#%%
# plt.figure()

# plt.plot(Time[:-step], testy[:,0],  label="y1-test",color="c")
# plt.plot(Time[:-step], predtest[:,0], label="y1-pred")
# plt.plot(Time[:-step], testy[:,1],  label="y2-test",color="m")
# plt.plot(Time[:-step], predtest[:,1], label="y2-pred")
#plt.legend()

#%%
val = trainx
with tf.device('/CPU:0'):
    U = val[:1,0:1,:]
    predtest2=np.ndarray(shape=(len(val),2))
    for k in range(0, len(val)-1):            
        predtest2[k, :] = model.predict(U)
        U = val[k+1:k+2,0:1,:]
        U[0,0,0]= predtest2[k, 0]
        U[0,0,1]= predtest2[k, 1]
    
#%%
plt.figure()
plt.subplot(211)
plt.title("NARX model Training Simulation Result")
plt.plot(predtest2[:-2,0])
plt.plot(trainy[:,0],  label="y1-test",color="c")
plt.subplot(212)
plt.plot(predtest2[:-2,1])
plt.plot(trainy[:,1],  label="y2-test",color="m")

#%%
val = testx
with tf.device('/CPU:0'):
    U = val[:1,0:1,:]
    predtest2=np.ndarray(shape=(len(val),2))
    for k in range(0, len(val)-1):            
        predtest2[k, :] = model.predict(U)
        U = val[k+1:k+2,0:1,:]
        U[0,0,0]= predtest2[k, 0]
        U[0,0,1]= predtest2[k, 1]
    
#%%
plt.figure()
plt.subplot(211)
plt.title("NARX model Validation Result")
plt.plot(predtest2[:-2,0])
plt.plot(testy[:,0],  label="y1-test",color="c")
plt.subplot(212)
plt.plot(predtest2[:-2,1])
plt.plot(testy[:,1],  label="y2-test",color="m")