# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:24:59 2021

@author: jeffa
"""



from keras.models import Sequential
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, SimpleRNN, LSTM
from keras import optimizers

start = time.time()

# Load TCLab data from system Identification experiments
# Data sets to use for training   
dat1 = pd.read_csv('date_030120211601.txt')
dat2 = pd.read_csv('date_030120211644.txt')
dat3 = pd.read_csv('date_030120210942.txt')
dat = pd.concat((dat1, dat2, dat3))


Q1train = dat[' Heater 1 (%)'].values
Q2train = dat[' Heater 2 (%)'].values
T1train = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
T2train = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]

Qtrain = np.hstack((Q1train.reshape(len(Q1train),1),
                    Q2train.reshape(len(Q2train),1)))
Ttrain = np.hstack((T1train.reshape(len(T1train),1), 
                    T2train.reshape(len(T2train),1)))

# Load data set to use for testing
dat = pd.read_csv('data_set1_013021_1930.txt')
Time = dat['Time (sec)'].values
Q1test = dat[' Heater 1 (%)'].values
Q2test = dat[' Heater 2 (%)'].values
T1test = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
T2test = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]

Qtest = np.hstack((Q1test.reshape(len(Q1test),1), 
                   Q2test.reshape(len(Q2test),1)))
Ttest = np.hstack((T1test.reshape(len(T1test),1), 
                   T2test.reshape(len(T2test),1)))


# Normalize data 
scaler = MinMaxScaler(feature_range=(0, 1))
QtrainScaled = scaler.fit_transform(Qtrain)
TtrainScaled = scaler.fit_transform(Ttrain)
QtestScaled = scaler.fit_transform(Qtest)
TtestScaled = scaler.fit_transform(Ttest)

# Add time lags for the outputs 
def shift_data(Qdat, Tdat, lags):    
    Y = pd.DataFrame(Tdat)
    Y.columns = ('T1(t)', 'T2(t)')
    X = list([pd.DataFrame(Qdat)])
    XcolNames = ('Q1(t)', 'Q2(t)')
    for i in range(1, lags+1):
        X.append(Y.shift(i)) 
        XcolNames += ('T1(t-%d)' %(i), 'T2(t-%d)' %(i))    
    X = pd.concat(X, axis=1)
    X.columns = XcolNames
    # Replace X will 
    X=X.fillna(0)    
    return X, Y

lags = 10
trainX, trainY = shift_data(QtrainScaled, TtrainScaled, lags)
testX, testY = shift_data(QtestScaled, TtestScaled, lags)


# # Reshape for 3D input to RNN
trainX = trainX.values.reshape((trainX.shape[0], 1, 2+2*lags))
testX = testX.values.reshape((testX.shape[0], 1, 2+2*lags))

# # Build network arch.
in_dim = (trainX.shape[1], trainX.shape[2])
model = Sequential()
model.add(LSTM(units=50, input_shape=in_dim, activation="sigmoid"))
#model.add(Dense(8, activation="tanh"))
model.add(Dense(2))
#opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='mae', optimizer='adam')
model.summary()

# # Train model
with tf.device('/CPU:0'):
    history = model.fit(trainX, trainY, batch_size=32, epochs=50, verbose=1, shuffle=False)

#%%

predtest= model.predict(trainX)
predtest_inv = scaler.inverse_transform(predtest)
plt.figure()
plt.plot(scaler.inverse_transform(trainY))
plt.plot(predtest_inv)