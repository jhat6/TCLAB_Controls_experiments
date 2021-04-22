# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:49:43 2021

@author: jeffa
"""
import tclab  # pip install tclab
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# define energy balance model
def heat(x,t,Q1,Q2):
    # Parameters
    Ta = 23 + 273.15   # K
    U = 10.0           # W/m^2-K
    m = 4.0/1000.0     # kg
    Cp = 0.5 * 1000.0  # J/kg-K    
    A = 10.0 / 100.0**2 # Area in m^2
    As = 2.0 / 100.0**2 # Area in m^2
    alpha1 = 0.0100     # W / % heater 1
    alpha2 = 0.0075     # W / % heater 2
    eps = 0.9          # Emissivity
    sigma = 5.67e-8    # Stefan-Boltzman

    # Temperature States 
    T1 = x[0]
    T2 = x[1]

    # Heat Transfer Exchange Between 1 and 2
    conv12 = U*As*(T2-T1)
    rad12  = eps*sigma*As * (T2**4 - T1**4)

    # Nonlinear Energy Balances
    dT1dt = (1.0/(m*Cp))*(U*A*(Ta-T1) \
            + eps * sigma * A * (Ta**4 - T1**4) \
            + conv12 + rad12 \
            + alpha1*Q1)
    dT2dt = (1.0/(m*Cp))*(U*A*(Ta-T2) \
            + eps * sigma * A * (Ta**4 - T2**4) \
            - conv12 - rad12 \
            + alpha2*Q2)

    return [dT1dt,dT2dt]

# save txt file
def save_txt(t,u1,u2,y1,y2,sp1,sp2):
    data = np.vstack((t,u1,u2,y1,y2,sp1,sp2))  # vertical stack
    data = data.T                 # transpose data
    top = 'Time (sec), Heater 1 (%), Heater 2 (%), ' \
        + 'Temperature 1 (degC), Temperature 2 (degC), ' \
        + 'Set Point 1 (degC), Set Point 2 (degC)' 
    
    now = datetime.now()
    currentTime = now.strftime('%m') + now.strftime('%d') + now.strftime('%Y') + \
                  now.strftime('%H') + now.strftime('%M')  
    fileName = "date_" + currentTime + ".txt"
    np.savetxt(fileName,data,delimiter=',',header=top,comments='')

def input_signals(loops):
    nstep = loops

    # random signal generation

    a_range = [0, 100]
    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
    
    #b_range = [60, 600]
    b_range = [300, 720]
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
    b = np.round(b)
    b = b.astype(int)
    
    b[0] = 0
    
    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]
    
    # Random Signal
    i=0
    random_signal = np.zeros(nstep)
    while b[i]<np.size(random_signal):
        k = b[i]
        random_signal[k:] = a[i]
        i=i+1
    
    # PRBS
    # a = np.zeros(nstep)
    # j = 0
    # while j < nstep:
    #     a[j] = 100
    #     a[j+1] = 0
    #     j = j+1
    
    i=0
    prbs = np.zeros(nstep)
    while b[i]<np.size(prbs):
        k = b[i]
        prbs[k:] = a[i]
        i=i+1
    
    return prbs


# Connect to Arduino
a = tclab.TCLab()

# Turn LED on
print('LED On')
a.LED(100)

# Run time in minutes
#run_time = 240.0
run_time = 15.0

# Number of cycles
loops = int(60.0*run_time)
tm = np.zeros(loops)

# Temperature (K)
Tsp1 = np.ones(loops) * 23.0 # set point (degC)
T1 = np.ones(loops) * a.T1 # measured T (degC)

Tsp2 = np.ones(loops) * 23.0 # set point (degC)
T2 = np.ones(loops) * a.T2 # measured T (degC)

# Predictions
Tp1 = np.ones(loops) * a.T1
Tp2 = np.ones(loops) * a.T2
error_eb = np.zeros(loops)


# # step response
# Q1_step = 100*np.zeros((60*105))
# Q2_step = 100*np.zeros((60*105))

# Q1_step[0:900]=100
# Q1_step[3600:4500]=100
# Q2_step[1800:2700]=100
# Q2_step[3600:5400]=100

# # impulse tests (0 - 100%)
# Q1_prbs = input_signals(loops-len(Q1_step))
# Q2_prbs = input_signals(loops-len(Q1_step))

# Q1 = np.concatenate((Q1_step, Q1_prbs))
# Q2 = np.concatenate((Q2_step, Q2_prbs))


# # impulse tests (0 - 100%)
Q1 = input_signals(loops)
Q2 = input_signals(loops)

Q1 = np.zeros(loops)
Q2 = np.zeros(loops)

print('Running Main Loop. Ctrl-C to end.')
print('  Time   Q1     Q2    T1     T2')
print('{:6.1f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(tm[0], \
                                                       Q1[0], \
                                                       Q2[0], \
                                                       T1[0], \
                                                       T2[0]))

# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Main Loop
start_time = time.time()
prev_time = start_time
try:
    for i in range(1,loops):
        # Sleep time
        sleep_max = 1.0
        sleep = sleep_max - (time.time() - prev_time)
        if sleep>=0.01:
            time.sleep(sleep-0.01)
        else:
            time.sleep(0.01)

        # Record time and change in time
        t = time.time()
        dt = t - prev_time
        prev_time = t
        tm[i] = t - start_time

        # Read temperatures in Kelvin 
        T1[i] = a.T1
        T2[i] = a.T2

        # Simulate one time step with Energy Balance
        Tinit = [Tp1[i-1]+273.15,Tp2[i-1]+273.15]
        Tnext = odeint(heat,Tinit, \
                       [0,dt],args=(Q1[i-1],Q2[i-1]))
        Tp1[i] = Tnext[1,0]-273.15
        Tp2[i] = Tnext[1,1]-273.15
        error_eb[i] = error_eb[i-1] \
                      + (abs(Tp1[i]-T1[i]) \
                      +  abs(Tp2[i]-T2[i]))*dt

        # Write output (0-100)
        a.Q1(Q1[i])
        a.Q2(Q2[i])

        # Print line of data
        print('{:6.1f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(tm[i], \
                                                               Q1[i], \
                                                               Q2[i], \
                                                               T1[i], \
                                                               T2[i]))

        # Plot
        plt.clf()
        ax=plt.subplot(3,1,1)
        ax.grid()
        plt.plot(tm[0:i],T1[0:i],'ro',label=r'$T_1$ measured')
        plt.plot(tm[0:i],Tp1[0:i],'k-',label=r'$T_1$ energy balance')
        plt.plot(tm[0:i],T2[0:i],'bx',label=r'$T_2$ measured')
        plt.plot(tm[0:i],Tp2[0:i],'k--',label=r'$T_2$ energy balance')
        plt.ylabel('Temperature (degC)')
        plt.legend(loc=2)
        ax=plt.subplot(3,1,2)
        ax.grid()
        plt.plot(tm[0:i],error_eb[0:i],'k-',label='Energy Balance Error')
        plt.ylabel('Cumulative Error')
        plt.legend(loc='best')
        ax=plt.subplot(3,1,3)
        ax.grid()
        plt.plot(tm[0:i],Q1[0:i],'r-',label=r'$Q_1$')
        plt.plot(tm[0:i],Q2[0:i],'b:',label=r'$Q_2$')
        plt.ylabel('Heaters')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')
        plt.draw()
        plt.pause(0.05)

    # Turn off heaters
    a.Q1(0)
    a.Q2(0)
    # Save text file and plot at end
    save_txt(tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    # Save figure
    plt.savefig('test_Models.png')

# Allow user to end loop with Ctrl-C           
except KeyboardInterrupt:
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    print('Shutting down')
    a.close()
    save_txt(tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    plt.savefig('test_Models.png')

# Make sure serial connection still closes when there's an error
except:           
    # Disconnect from Arduino
    a.Q1(0)
    a.Q2(0)
    print('Error: Shutting down')
    a.close()
    save_txt(tm[0:i],Q1[0:i],Q2[0:i],T1[0:i],T2[0:i],Tsp1[0:i],Tsp2[0:i])
    plt.savefig('test_Models.png')
    raise