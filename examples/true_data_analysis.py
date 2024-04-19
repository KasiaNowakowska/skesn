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

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

#load in data
q = np.load(input_path+'/q.npy')
ke = np.load(input_path+'/KE.npy')
time_vals = np.load(input_path+'/time_vals.npy')
print(len(q), len(ke), len(time_vals))

# Reshape the arrays into column vectors
ke_column = ke.reshape(len(ke), 1)
q_column = q.reshape(len(q), 1)

#ke_column = np.sqrt(ke_column)

# Concatenate the column vectors horizontally
data = np.hstack((ke_column, q_column))

# Print the shape of the combined array
print(data.shape)

def calculate_distance(traj1, traj2):
  # Ensure trajectories have the same length
  assert traj1.shape == traj2.shape
  distances = np.linalg.norm(traj2-traj1, axis=1)
  return distances

#### ANALYSIS ####
prediction_index = 6010
prediction_point = data[prediction_index, :]
print('initial point', prediction_point)

epsilon_q =  1e-3
epsilon_ke = 1e-6
print('eps values: ke', epsilon_ke, 'q', epsilon_q)
points = 0
inds = []
for i in range(len(data[:,0])):
  if (prediction_point[0] - epsilon_ke < abs(data[i,0]) < prediction_point[0] + epsilon_ke) and (prediction_point[1] - epsilon_q < abs(data[i,1]) < prediction_point[1] + epsilon_q):
    #print(data[i,0], data[i,1], time_vals[i]) 
    points += 1
    inds.append(i)
print('number of points', points)
print('inidices of points', inds)

t_value = 1000
inds0 = inds[0]
inds1 = inds[1]

IC0 = data[inds0:inds0+t_value, :]
IC1 = data[inds1:inds1+t_value, :]


for t in range(t_value):
    fig, ax = plt.subplots(1,2, figsize=(6,6), tight_layout=True)
    scatter0 = ax[0].scatter(IC0[:t,1], IC0[:t,0], c='red', marker='.', alpha=0.8)
    scatter1 = ax[0].scatter(IC1[:t,1], IC1[:t,0], c='blue', marker='.', alpha=0.8)

    d = calculate_distance(IC0[:t], IC1[:t])
    ax[1].plot(np.arange(0, t, 1), d)

    ax[0].set_xlabel('q')
    ax[0].set_ylabel('KE')
    ax[0].set_xlim(0.265, 0.300)
    ax[0].set_ylim(0, 3e-4)
    ax[1].set_xlabel('time')
    ax[1].set_xlim(0,t_value)
    ax[1].set_ylabel('euclidean distance')
    ax[1].set_ylim(0,0.020)
    fig.savefig(output_path+'/phase_space_distance%i.png' % t)
    plt.close()

