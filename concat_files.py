# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:54:27 2021

@author: jeffa
"""

import pandas as pd

dat1 = pd.read_csv('date_030120211601.txt')
dat2 = pd.read_csv('date_030120211644.txt')
dat3 = pd.read_csv('date_030120210942.txt')
dat = pd.concat((dat1, dat2, dat3))


# Q1 = dat[' Heater 1 (%)'].values
# Q2 = dat[' Heater 2 (%)'].values
# T1 = dat[' Temperature 1 (degC)'].values - dat[' Temperature 1 (degC)'].values[0]
# T2 = dat[' Temperature 2 (degC)'].values - dat[' Temperature 2 (degC)'].values[0]