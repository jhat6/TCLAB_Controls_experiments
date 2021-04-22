# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 20:31:07 2021

@author: jeffa
"""



import numpy as np
from sippy.functionsetSIM import SS_lsim_process_form as SS_lsim
import sippy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr

dat = pd.read_csv('data_set2_013121_1100.txt')
Time = dat['Time (sec)'].values
Q1 = dat[' Heater 1 (%)'].values
Q2 = dat[' Heater 2 (%)'].values
T1 = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
T2 = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]
U = np.vstack([[Q1],[Q2]])
Y = np.vstack([[T1],[T2]])

##System identification
method = 'N4SID'
sys_id = sp.system_identification(Y, U, method)
xid, yid = SS_lsim(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0)

#%%

plt.close("all")

plt.figure()
plt.subplot(2,1,1)
plt.plot(Time, Y[0])
plt.plot(Time, yid[0])
plt.ylabel("[C]")
plt.grid()
plt.title("Temperature 1")
plt.legend(['Exper. Data', 'Identified system, ' + method])
plt.subplot(2,1,2)
plt.plot(Time, U[0])
plt.ylabel("Heater 1 [Power %]")
plt.xlabel("Time [s]")
plt.legend(['PRBS Signal'])
plt.grid()
fitErr1 = yid[0]-Y[0]
print(np.mean(np.abs(fitErr1)))
print(np.mean(fitErr1**2))

plt.figure()
plt.subplot(2,1,1)
plt.plot(Time, Y[1])
plt.plot(Time, yid[1])
plt.ylabel("[C]")
plt.grid()
plt.title("Temperature 2")
plt.legend(['Exper. Data', 'Identified system, ' + method])
plt.subplot(2,1,2)
plt.plot(Time, U[1])
plt.ylabel("Heater 2 [Power %]")
plt.xlabel("Time [s]")
plt.legend(['PRBS Signal'])
plt.grid()
fitErr2 = yid[1]-Y[1]
print(np.mean(np.abs(fitErr2)))
plt.show()




#%%

dat = pd.read_csv('data_set1_013021_1930.txt')
Time = dat['Time (sec)'].values
U1 = dat[' Heater 1 (%)'].values
U2 = dat[' Heater 2 (%)'].values
T1 = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
T2 = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]
U = np.vstack([[U1],[U2]])
Y = np.vstack([[T1],[T2]])
xid, yid = SS_lsim(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0)

#%%

b = T1[1:,] - T1[0:-1,]
Ta = T1[0,]
a1 = np.asmatrix(Ta - T1[0:-1,]).T
a2 = np.asmatrix(Ta**4 - T1[0:-1,]**4).T
a3 = np.asmatrix(T2[0:-1,] - T1[0:-1,]).T
a4 = np.asmatrix(T2[0:-1,]**4 - T1[0:-1,]**4).T
a5 = np.asmatrix(U1[0:-1,]).T
A = np.concatenate((a1, a2, a3, a4, a5), axis=1)

x = lsqr(A, b)

T1est = np.matmul(A,x[0]) - b

Ta = 23 + 273.15   # K
Ua = 10.0           # W/m^2-K
m = 4.0/1000.0     # kg
Cp = 0.5 * 1000.0  # J/kg-K    
Area1 = 10.0 / 100.0**2 # Area in m^2
Areas = 2.0 / 100.0**2 # Area in m^2
alpha1 = 0.0100     # W / % heater 1
alpha2 = 0.0075     # W / % heater 2
eps = 0.9          # Emissivity
sigma = 5.67e-8    # Stefan-Boltzman


a1 = Ua*Area1/(m*Cp)
a2 = eps*sigma*Area1
a3 = Ua*Areas
a4 = eps*sigma*Areas
a5 = alpha1

x0 = np.array([a1, a2, a3, a4, a5])


#%%

plt.figure()
plt.subplot(2,1,1)
plt.plot(Time, Y[0])
plt.plot(Time, yid[0])
plt.ylabel("[C]")
plt.grid()
plt.title("Temperature 1")
plt.legend(['Validation Data', 'Identified system, ' + method])
plt.subplot(2,1,2)
plt.plot(Time, U[0])
plt.ylabel("Heater 1 [Power %]")
plt.xlabel("Time [s]")
plt.legend(['PRBS Signal'])
plt.grid()
fitErr1 = yid[0]-Y[0]
print(np.mean(np.abs(fitErr1)))
print(np.mean(fitErr1**2))

plt.figure()
plt.subplot(2,1,1)
plt.plot(Time, Y[1])
plt.plot(Time, yid[1])
plt.ylabel("[C]")
plt.grid()
plt.title("Temperature 2")
plt.legend(['Validation Data', 'Identified system, ' + method])
plt.subplot(2,1,2)
plt.plot(Time, U[1])
plt.ylabel("Heater 2 [Power %]")
plt.xlabel("Time [s]")
plt.legend(['PRBS Signal'])
plt.grid()
fitErr2 = yid[1]-Y[1]
print(np.mean(np.abs(fitErr2)))
plt.show()

#%%


from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, SimpleRNN, LSTM
from keras import regularizers
from numpy import array, sqrt, array
from numpy.random import uniform
from numpy import hstack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import time 

start = time.time()


with tf.device('/CPU:0'):
    #dat = pd.read_csv('data_set2_013121_1100.txt')
    #dat = pd.read_csv('date_030120210942.txt')
    
    dat1 = pd.read_csv('date_030120211601.txt')
    dat2 = pd.read_csv('date_030120211644.txt')
    dat3 = pd.read_csv('date_030120210942.txt')
    dat = pd.concat((dat1, dat2, dat3))
    
    
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
    model.add(SimpleRNN(units=64, input_shape=in_dim, activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)))
    model.add(Dense(16, activation="relu", 
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)))
    
    model.add(Dense(out_dim))
    
    #opt = optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='mse', optimizer=opt)
    
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    
    model.fit(trainx,trainy, batch_size=32, epochs=250, verbose=1)
    trainScore = model.evaluate(trainx, trainy, verbose=0)
    print(trainScore)
    
    predtest= model.predict(testx)
    
    rmse_y1 = sqrt(mean_squared_error(testy[:,0], predtest[:,0]))
    rmse_y2 = sqrt(mean_squared_error(testy[:,1], predtest[:,1]))
    print("RMSE y1: %.2f y2: %.2f" % (rmse_y1, rmse_y2))

#%%
plt.figure()

plt.plot(Time[:-step], testy[:,0],  label="y1-test",color="c")
plt.plot(Time[:-step], predtest[:,0], label="y1-pred")
plt.plot(Time[:-step], testy[:,1],  label="y2-test",color="m")
plt.plot(Time[:-step], predtest[:,1], label="y2-pred")
plt.legend()

#%%
val = trainx
with tf.device('/GPU:0'):
    U = val[:1,0:1,:]
    predtest2=np.ndarray(shape=(len(val),2))
    for k in range(0, len(val)-1):            
        predtest2[k, :] = model.predict_on_batch(U)
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

with tf.device('/GPU:0'):
    val = testx
    with tf.device('/GPU:0'):
        U = val[:1,0:1,:]
        predtest2=np.ndarray(shape=(len(val),2))
        for k in range(0, len(val)-1):            
            predtest2[k, :] = model.predict_on_batch(U)
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
