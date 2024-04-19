"""
python script for ESN grid search.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

from math import isclose, log
import os
sys.path.append(os.getcwd())
print(sys.path)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

import h5py

from docopt import docopt
args = docopt(__doc__)

import nolds

input_path = args['--input_path']
output_path = args['--output_path']

#load in data
q = np.load(input_path+'/q.npy')
ke = np.load(input_path+'/KE.npy')
time_vals = np.load(input_path+'/time_vals.npy')
print(len(q), len(ke), len(time_vals))

q = q[:8000]


lyap = nolds.lyap_r(q)
print('largest lyapunov exponent q', lyap)
lyap_time = 1/ np.abs(lyap)
print('lyapunov time q', lyap_time)
lyap = nolds.lyap_r(ke)
print('largest lyapunov exponent ke', lyap)
lyap_time = 1/ np.abs(lyap)
print('lyapunov time ke', lyap_time)

#fig,ax = plt.subplots(1)
#x = phase_space[:, 0]
#y = phase_space[:, 1]
#ax.plot(x, y, color='b', label='Reconstructed Phase Space Trajectory')
#ax.set_xlabel('Dimension 1')
#ax.set_ylabel('Dimension 2')
#plt.show()

# Plot the trajectory of the reconstructed phase space (1D visualization)
#plt.figure(figsize=(10, 6))
#plt.plot(q[:-2 * lag], q[lag:-lag], 'b-', label='Reconstructed Phase Space Trajectory')
#plt.xlabel('x(t)')
#plt.ylabel('x(t + lag)')
#plt.title(f'Reconstructed Phase Space Trajectory (lag = {lag})')
#plt.legend()
#plt.grid(True)
#plt.show()

'''
lyap_lags_q = np.zeros(10)
lyap_lags_ke = np.zeros(10)
for i in range(10):
    lyap = nolds.lyap_r(q, emb_dim=10, lag=i)
    print('largest lyapunov exponent q', lyap)
    lyap_time = 1/ np.abs(lyap)
    print('lyapunov time q', lyap_time)
    lyap_lags_q[i] = lyap_time
    lyap = nolds.lyap_r(ke, emb_dim=10, lag=i)
    print('largest lyapunov exponent ke', lyap)
    lyap_time = 1/ np.abs(lyap)
    print('lyapunov time ke', lyap_time)
    lyap_lags_ke[i] = lyap_time

plt.plot(np.linspace(1,10,10), lyap_lags_q)
plt.plot(np.linspace(1,10,10), lyap_lags_ke)
plt.show()
'''
