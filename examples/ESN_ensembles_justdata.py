"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --ensembles=<ensembles>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_train=<n_train>                number of training data points [default: 6000]
    --n_sync=<n_sync>                  number of data to synchronise with [default: 10]
    --n_prediction=<n_prediction>      number of timesteps for prediction [default: 2000]
    --synchronisation=<synchronisation>  turn on synchronisation [default: True]
    --n_reservoir=<n_reservoir>        size of reservoir [default: 2000]
    --spectral_radius=<spectral radius> spectral radius of reservoir [default: 0.80]
    --sparsity=<sparsity>              sparsity of reservoir [default: 0.80]
    --lambda_r=<lambda_r>              noise added to update equation, only when use_noise is on [default: 1e-2]
    --beta=<beta>                      tikhonov regularisation parameter [default: 1e-3]
    --use_noise=<use_noise>            turn noise on in update equation [default: True]
    --use_bias=<use_bias>              turn bias term on in u adds nother dimension to input [default: True]
    --use_b=<use_b>                    turn on extra bias term in actoiivation function [default: True] 
    --ensembles=<ensembles>            number of ensembles/how many times to rerun the ESN with the same IC and no retraining
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

from math import isclose
import os
sys.path.append(os.getcwd())
print(sys.path)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from cross_validation import ValidationBasedOnRollingForecastingOrigin
from esn_old_adaptations import EsnForecaster

from scipy.signal import find_peaks

import h5py

from docopt import docopt
args = docopt(__doc__)

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

#Define functions
def calculate_nrmse(predictions, true_values):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between predicted and true values using the range for normalisation.
  
    Args:
    predictions: numpy array or list containing predicted values
    true_values: numpy array or list containing true values
  
    Returns:
    nrmse: Normalized Root Mean Square Error (float)
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
  
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
  
    # Calculate range of true values
    value_range = np.max(true_values) - np.min(true_values)
  
    # Calculate NRMSE
    nrmse = rmse / value_range
  
    return nrmse
    
def calculate_mse(predictions, true_values):
    """
    Calculate the Mean Square Error (MSE) between predicted and true values.
  
    Args:
    predictions: numpy array or list containing predicted values
    true_values: numpy array or list containing true values
  
    Returns:
    mse: Mean Square Error (float)
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
  
    # Calculate MSE
    mse = np.mean((predictions - true_values) ** 2)
  
    return mse

def fftinx(variable, x1):
    variable = variable - np.mean(variable)
    window = np.hanning(len(variable))
    windowed_signal = variable * window
    fs = len(x1)/(x1[-1]-x1[0])
    end = x1[-1]
    start = x1[0]
    fft = np.fft.fft(windowed_signal, len(x1))
    fft = np.fft.fftshift(fft)
    freq = np.fft.fftfreq(len(fft), d=(end-start)/len(x1))
    freq = np.fft.fftshift(freq)
    freq = 2*np.pi*freq
    magnitude = np.abs(fft)
    psd = magnitude**2
    #print(psd)
    return psd, freq, fft

def timedelay(y_true, y_pred):
    cross_corr = np.correlate(y_true, y_pred, mode='full')
    lag_max_corr = np.argmax(cross_corr) - len(y_true) + 1
    return lag_max_corr
    
input_path = args['--input_path']
output_path1 = args['--output_path']
n_train = int(args['--n_train'])
n_sync = int(args['--n_sync'])
n_prediction = int(args['--n_prediction'])
synchronisation = args['--synchronisation']
n_reservoir = int(args['--n_reservoir'])
spectral_radius = float(args['--spectral_radius'])
sparsity = float(args['--sparsity'])
lambda_r = float(args['--lambda_r'])
beta = float(args['--beta'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']
ensembles = int(args['--ensembles'])

data_dir = '/validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_beta{4:}_noise{5:}_bias{6:}_b{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), float(args['--beta']), args['--use_noise'], args['--use_bias'], args['--use_b'])

output_path = output_path1+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

images_output_path = output_path+'/images'
if not os.path.exists(images_output_path):
    os.makedirs(images_output_path)
    print('made directory')

images_output_path2 = output_path+'/phasespace'
if not os.path.exists(images_output_path2):
    os.makedirs(images_output_path2)
    print('made directory')

images_output_path3 = output_path+'/peakspath'
if not os.path.exists(images_output_path3):
    os.makedirs(images_output_path3)
    print('made directory')

output_path_2 = output_path+'/withtime'
if not os.path.exists(output_path_2):
    os.makedirs(output_path_2)
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

ss = StandardScaler()

data = ss.fit_transform(data)
print(np.shape(data))

#generate model
modelsync = EsnForecaster(
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            regularization='l2',
            lambda_r=lambda_r,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=use_additive_noise_when_forecasting,
            random_state=42,
            use_bias=use_bias,
            use_b=use_b,
            beta=beta)

# Plot long-term prediction
synclen = n_sync
trainlen = n_train - synclen
ts = data[0:trainlen,:]
train_times = time_vals[0:trainlen]
dt = time_vals[1] - time_vals[0]
predictionlen = n_prediction
future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
sync_times = time_vals[trainlen:trainlen+synclen]
prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
sync_data = data[trainlen:trainlen+synclen]
future_data = data[trainlen:trainlen+synclen+predictionlen, :]

#generate model
modelsync = EsnForecaster(
            n_reservoir=n_reservoir,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            regularization='l2',
            lambda_r=lambda_r,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=use_additive_noise_when_forecasting,
            random_state=42,
            use_bias=use_bias,
            use_b=use_b,
            beta=beta)
modelsync.fit(ts)

ensemble_all_vals_ke = np.zeros((len(prediction_times), ensembles))
ensemble_all_vals_q = np.zeros((len(prediction_times), ensembles))
ensemble_all_vals = np.zeros((len(prediction_times), 2, ensembles))

# run through ensembles
for i in range(ensembles):
    #synchronise data
    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
    #predict
    future_predictionsync = modelsync.predict(predictionlen)

    #inverse scalar
    inverse_training_data = ss.inverse_transform(ts) 
    inverse_prediction = ss.inverse_transform(future_predictionsync)
    inverse_test_data = ss.inverse_transform(test_data)
    inverse_future_data = ss.inverse_transform(future_data) #this include the sync data

    ### ensemble mean ###
    ensemble_all_vals_ke[:,i] = inverse_prediction[:,0]
    ensemble_all_vals_q[:,i] = inverse_prediction[:,1]

print(np.shape(ensemble_all_vals))
np.save(output_path+'/ensemble%i_all_vals.npy' % ensembles, ensemble_all_vals)

ensemble_mean_ke = np.mean(ensemble_all_vals_ke, axis=1)
ensemble_mean_q = np.mean(ensemble_all_vals_q, axis=1)
ensemble_var_ke = np.var(ensemble_all_vals_ke, axis=1)
ensemble_var_q = np.var(ensemble_all_vals_q, axis=1)
max_var_ke = np.max(ensemble_var_ke)
max_var_q = np.max(ensemble_var_q)

median_ke = np.median(ensemble_all_vals_ke, axis=1) 
median_q = np.median(ensemble_all_vals_q, axis=1) 
lower_bound_ke = np.percentile(ensemble_all_vals_ke, 10, axis=1)
upper_bound_ke = np.percentile(ensemble_all_vals_ke, 90, axis=1)
lower_bound_q = np.percentile(ensemble_all_vals_q, 10, axis=1)
upper_bound_q = np.percentile(ensemble_all_vals_q, 90, axis=1)

# creating grid for subplots
fig = plt.figure(constrained_layout=True)
fig.set_figheight(12)
fig.set_figwidth(12)

ax1 = plt.subplot2grid(shape=(8, 8), loc=(0, 0), colspan=4, rowspan=4)
ax21 = plt.subplot2grid(shape=(8, 8), loc=(4, 0), colspan=4, rowspan=2)
ax22 = plt.subplot2grid(shape=(8, 8), loc=(6, 0), colspan=4, rowspan=2, sharex=ax21)
ax2 = plt.subplot2grid(shape=(8, 8), loc=(0, 4), colspan=4, rowspan=2)
ax3 = plt.subplot2grid(shape=(8, 8), loc=(2, 4), colspan=4, rowspan=2, sharex=ax2)
ax4 = plt.subplot2grid(shape=(8, 8), loc=(4, 4), colspan=4, rowspan=2, sharex=ax2)
ax5 = plt.subplot2grid(shape=(8, 8), loc=(6, 4), colspan=4, rowspan=2, sharex=ax2)

for tvalue in range(len(prediction_times)):
    if tvalue != 0:
        if tvalue % 50 == 0:
            ax21.plot(train_times, inverse_training_data[:, 0], color='blue', alpha=0.5)
            ax21.plot(future_times, inverse_future_data[:, 0], color='blue', alpha=0.5)
            ax22.plot(train_times, inverse_training_data[:, 1], color='blue', alpha=0.5)
            ax22.plot(future_times, inverse_future_data[:, 1], color='blue', alpha=0.5)
            for member in range(ensembles):
                if member % 10 == 0:
                    inverse_prediction = ensemble_all_vals[:tvalue,:,member]
                    ax1.scatter(inverse_prediction[:,1], inverse_prediction[:,0])
                    ax21.plot(prediction_times[:tvalue], inverse_prediction[:,0], linewidth=2, label='Prediction')
                    ax22.plot(prediction_times[:tvalue], inverse_prediction[:,1], linewidth=2, label='Prediction')
            ax2.plot(train_times, inverse_training_data[:, 0], color='blue', alpha=0.5)
            ax2.plot(future_times, inverse_future_data[:, 0], color='blue', alpha=0.5)
            ax3.plot(train_times, inverse_training_data[:, 1], color='blue', alpha=0.5) 
            ax3.plot(future_times, inverse_future_data[:, 1], color='blue', alpha=0.5)
            ax2.plot(prediction_times[:tvalue], median_ke[:tvalue], linewidth=2, label='Median Prediction', color='green')
            ax3.plot(prediction_times[:tvalue], median_q[:tvalue], linewidth=2, label='Median Prediction', color='green')
            ax2.fill_between(prediction_times[:tvalue], lower_bound_ke[:tvalue], upper_bound_ke[:tvalue], color='green', alpha=0.3, label='80% Confidence Interval')
            ax3.fill_between(prediction_times[:tvalue], lower_bound_q[:tvalue], upper_bound_q[:tvalue], color='green', alpha=0.3, label='80% Confidence Interval')
            ax4.plot(prediction_times[:tvalue], ensemble_var_ke[:tvalue])
            ax5.plot(prediction_times[:tvalue], ensemble_var_q[:tvalue])
            ax1.set_xlabel('q')
            ax1.set_ylabel('KE')
            ax2.set_ylabel('median KE')
            ax3.set_ylabel('median q')
            ax4.set_ylabel('variation KE')
            ax5.set_ylabel('variation q')
            ax21.set_ylabel('KE')
            ax22.set_ylabel('q')
            ax22.set_xlabel('time')
            ax1.set_xlim(0.265, 0.300)
            ax1.set_ylim(0, 3e-4)
            ax2.set_xlim(4000,12000)
            ax3.set_xlim(4000,12000)
            ax4.set_xlim(4000,12000)
            ax4.set_ylim(0,max_var_ke)
            ax5.set_xlim(4000,12000)
            ax5.set_ylim(0,max_var_q)
            fig.savefig(output_path_2+'/image%i.png' % tvalue)
        plt.close()
