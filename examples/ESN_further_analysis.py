"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

from math import isclose
import os
sys.path.append(os.getcwd())
#print(sys.path)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from cross_validation import ValidationBasedOnRollingForecastingOrigin
from esn_old_adaptations import EsnForecaster
import evaluation_metrics as em

from scipy.signal import find_peaks

import h5py

from docopt import docopt
args = docopt(__doc__)

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

input_path = args['--input_path']
output_path = args['--output_path']

output_path2 = output_path+'/boxplots'
print(output_path2)
if not os.path.exists(output_path2):
    os.makedirs(output_path2)
    print('made directory')
    
input_path2 = '../data'

#load in data
pred_keIC  = np.load(input_path+'/ensemble_all_vals_ke.npy')
pred_qIC = np.load(input_path+'/ensemble_all_vals_q.npy')
true_keIC = np.load(input_path+'/ensemble_test_data_ke.npy')
true_qIC = np.load(input_path+'/ensemble_test_data_q.npy')
time_IC = np.load(input_path+'/ensemble_prediction_times.npy')

allq = np.load(input_path2+'/q.npy')
allke = np.load(input_path2+'/KE.npy')

q_training = allq[:6000]
ke_training = allke[:6000]
time_training = np.arange(5000,11000)

'''
IC = np.arange(0, 750, 25)
gradientske = []
gradientsq = []
valueske = []
valuesq = []
for i in IC:
    #print(i)
    KE_diff = np.diff(allke, 1)
    q_diff = np.diff(allq, 1)
    KE_grad = KE_diff[6010+i]
    q_grad = q_diff[6010+i]
    valueske.append(allke[6010+i])
    valuesq.append(allq[6010+i])
    gradientske.append(KE_grad)
    gradientsq.append(q_grad)

valueske = np.array(valueske)
valuesq = np.array(valuesq)

tests, ensembles = 30, 100
peak_matches = []
peak_missing_one = []
within100 = []
true_peak_time = []
true_peak_val = []
mean_peak_time_errors = []
median_peak_value_errors = []
ub_error, lb_error = [], []
for i in range(tests):
    peak_time_error = []
    tp = em.true_next_peak(true_keIC[:,i], true_qIC[:,i], time_IC[:,i])
    true_peak_time.append(tp[1])
    true_peak_val.append(tp[2])
    num_peaks_forIC = []
    next_peak_time_forIC = []
    next_peak_value_forIC = []
    for r in range(ensembles):
        pp = em.pred_next_peak(pred_keIC[:,i,r], pred_qIC[:,i,r], time_IC[:,i])
        num_peaks_forIC.append(pp[0])
        next_peak_time_forIC.append(pp[1])
        next_peak_value_forIC.append(pp[2])
    counter_all = 0
    counter_missing_one = 0
    counter_next_time_within100 = 0
    for n_p in num_peaks_forIC:
        if n_p == tp[0]:
            counter_all += 1
        elif n_p == tp[0]-1:
            counter_missing_one += 1
    fraction_peak_matches = counter_all/ensembles
    fraction_peaks_missing_one = counter_missing_one/ensembles
    peak_matches.append(fraction_peak_matches)
    peak_missing_one.append(fraction_peaks_missing_one)
    for n_vp in next_peak_time_forIC:
        er = n_vp - tp[1]
        peak_time_error.append(er)
        if abs(n_vp - tp[1]) < 50:
            counter_next_time_within100 += 1
    fraction_within100 = counter_next_time_within100/ensembles
    within100.append(fraction_within100)

    median_peak_value = np.median(next_peak_value_forIC)
    print(median_peak_value)
    median_peak_value_errors.append(np.abs(tp[2]-median_peak_value))
    peak_time_error = np.array(peak_time_error)
    peak_time_error = peak_time_error[np.isfinite(peak_time_error)]
    mean_peak_time_error = np.median(peak_time_error)
    lower_bound_time_error = np.percentile(peak_time_error, 10)
    upper_bound_time_error = np.percentile(peak_time_error, 90)
    mean_peak_time_errors.append(mean_peak_time_error)
    ub_error.append(upper_bound_time_error)
    lb_error.append(lower_bound_time_error)

tests = 30
for i in range(tests):
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))

    # Define the grid for subplots
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=4)
    ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2, sharex=ax2)

    em.ensemble_timeseries_plustraining(pred_keIC[:,i,:], pred_qIC[:,i,:], true_keIC[:,i], true_qIC[:,i], ke_training, q_training, time_training, time_IC[:,i], ax=np.array((ax2,ax3)))
    ax2.scatter(true_peak_time[i], true_peak_val[i])
    ax2.set_xlim(10000, 14000)
    ax3.set_xlim(10000, 14000)
    ax2.set_ylim(-0.0001, 0.00035)
    ax3.set_ylim(0.260,0.30)
    ax2.plot(np.arange(11000,14000), allke[6000:9000], color='blue', alpha=0.2, linestyle='--')
    ax3.plot(np.arange(11000,14000), allq[6000:9000], color='blue', alpha=0.2, linestyle='--')
    #plt.suptitle('test={:.1f}, gradient ke={:.2e}, gradient q={:.2e}'.format(i+1, gradientske[i], gradientsq[i]))
    ax1.plot(allq[6010:6010+790],allke[6010:6010+790], linestyle='--', color='red', alpha=0.5)
    scatter = ax1.scatter(valuesq[:i+1], valueske[:i+1], c=within100[:i+1], cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax1)
    ax1.grid()
    ax1.set_xlabel('q')
    ax1.set_ylabel('KE')
    fig.savefig(output_path2+'/image{:03d}.png'.format(i))
    plt.close()
    '''
    
'''
ax.plot(q_timeseries[6010:6010+790],ke_timeseries[6010:6010+790], linestyle='--', color='red', alpha=0.5)
scatter = ax.scatter(valuesq, valueske, c=median_peak_value_errors, cmap='viridis')
cbar = fig.colorbar(scatter, label='error')
ax.grid()
#'ax.set_ylim(-5e-6,5e-6)
#ax.set_xlim(-0.0003,0.0003)
ax.set_xlabel('value of q')
ax.set_ylabel('value of KE')
'''

FT = np.linspace(0,600,75)
for i in FT:
    fig, ax = plt.subplots(1)
    phase_diagram_boxplot(pred_keIC, pred_qIC, true_keIC, true_qIC, prediction_times, forecast_time=i, ax=ax)
    ax.scatter(test_data_q[:i], test_data_ke[:i], marker='.', color='tab:blue', label='True')
    ax.scatter(median_q[:i], median_ke[:i], marker='.', color='orange', label='median')
    plt.legend()
    plt.grid()
    plt.savefig(output_path+'/boxplots%i.png' % i)
