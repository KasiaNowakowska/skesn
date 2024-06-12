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

output_path1 = output_path + '/onsets'
if not os.path.exists(output_path1):
    os.makedirs(output_path1)
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
KE_threshold = 0.00015

tvals = np.arange(0,8000,1)
wait_times = []
all_times = []

ke_range = (0.00000, 0.00008)
q_range = (0.285,0.290)
indices = []
prev_index = -364
# Iterate over the indices of the arrays
for i, (ke_val, q_val) in enumerate(zip(ke, q)):
    # Check if the current index satisfies the conditions
    if ke_range[0] <= ke_val <= ke_range[1] and q_range[0] <= q_val <= q_range[1]:
        # Check if the difference between the current index and the previous index is at least 100
        if i - prev_index >= 363:
            indices.append(i)
            prev_index = i

for t in range(1000):
    fig,ax = plt.subplots(1, figsize=(8,6))
    for i in indices:
        ax.scatter(q[i+t], ke[i+t], marker='.')
    ax.set_ylim(-0.0001, 0.00035)
    ax.set_xlim(0.260,0.30)
    ax.grid()
    ax.set_xlabel('q')
    ax.set_ylabel('KE')
    plt.savefig(output_path+"/phasespace{:05d}.png".format(t))
    plt.close()



'''
fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)

def phase_space(test_data_ke, test_data_q, times, ax=ax, fig=fig):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 6), tight_layout=True)
    plt.grid()
    s_true = ax.scatter(test_data_q, test_data_ke,
                            cmap='viridis', marker='.', c=times,
                            vmin=times[0], vmax=times[-1], label='True')
    ax.set_xlabel('q')
    ax.set_ylabel('KE')
    ax.set_ylim(-0.0001, 0.00035)
    ax.set_xlim(0.260,0.30)
    #cbar_true = fig.colorbar(s_true, ax=ax, label='time')
    #plt.legend()


for i in range(1,3000):
    fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)
    phase_space(ke[:i], q[:i], time_vals[:i], ax=ax, fig=fig)
    fig.savefig(output_path+"/phasespace{:05d}.png".format(i))
    plt.close()
'''





'''
for t_i in tvals:
    
    # creating grid for subplots
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(6)
    fig.set_figwidth(6)

    ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4, rowspan=3)
    ax2 = plt.subplot2grid(shape=(4, 4), loc=(3, 0), colspan=4, rowspan=1)
 
    onset_true = np.zeros((len(time_vals[:t_i])))
    offset_true = np.zeros((len(time_vals[:t_i])))

    for t in range(len(time_vals[:t_i])):
        if (ke[t] >= KE_threshold and ke[t-1] < KE_threshold):
            onset_true[t] = 1
        elif (ke[t] <= KE_threshold and ke[t-1] > KE_threshold):
            offset_true[t] = 1

  
    scatter = ax1.scatter(q[:t_i], ke[:t_i], cmap='viridis', marker='.', alpha=0.8, c=time_vals[:t_i], vmin=time_vals[0], vmax=time_vals[t_i])
    ax1.axhline(y=0.00015)
    ax1.set_xlabel('q')
    ax1.set_ylabel('KE')
    ax1.set_xlim(np.min(q), np.max(q))
    ax1.set_ylim(np.min(ke), np.max(ke))
    cbar = plt.colorbar(scatter, ax=ax1, ticks=[time_vals[0], time_vals[t_i]], label='time')
    ax2.plot(time_vals[:t_i], onset_true, color='green', label='true onset')
    ax2.plot(time_vals[:t_i], offset_true, color='red', label='true cessation')
    plt.legend()
    ax2.set_xlim(5000,7000)
    ax2.set_ylim(-0.1,1.1)
    ax2.set_xlabel('time')
    ax2.set_ylabel('threshold passed') 
    fig.savefig(output_path1+'/onsets{:05d}.png'.format(t_i))
    plt.close()
'''
'''
for t in range(tvals[-1]):
        if (ke[t] >= KE_threshold and ke[t-1] < KE_threshold):
            all_times.append(time_vals[t])
        #elif (ke[t] <= KE_threshold and ke[t-1] > KE_threshold):
            #all_times.append(time_vals[t])

for i in range(1, len(all_times)):
  range = all_times[i] - all_times[i-1]
  wait_times.append(range)
  
fig, ax = plt.subplots(1)
ax.hist(wait_times, bins=20)
ax.set_xlabel('wait time')
ax.set_ylabel('frequency')
fig.savefig(output_path+'/histogram_wait_times_onset.png')
'''

'''
#phase space
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
    fig.savefig(output_path+'/phase_space_distance{:03d}.png'.format(t))
    plt.close()
'''

