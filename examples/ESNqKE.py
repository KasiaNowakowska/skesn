"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_train=<n_train>                number of training data points [default: 6000]
    --n_sync=<n_sync>                  number of data to synchronise with [default: 10]
    --n_prediction=<n_predcition>      number of timesteps for prediction [default: 2000]
    --synchronisation=<synchronisation>  turn on synchronisation [default: True]
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
    
input_path = args['--input_path']
output_path = args['--output_path']
n_train = args['--n_train']
n_sync = args['--n_sync']
n_prediction = args['--n_prediction']
synchronisation = args['--synchronisation']

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
n_reservoir=2000,
spectral_radius=0.80,
sparsity=0.80,
regularization='l2',
lambda_r=1e-2,
in_activation='tanh',
out_activation='identity',
use_additive_noise_when_forecasting=True,
random_state=42,
use_bias=False)

# Plot long-term prediction
trainlen = n_train
synclen = n_sync
predictionlen = 2000 - synclen
train_times = time_vals[0:trainlen]
dt = time_vals[1] - time_vals[0]
ts = data[0:trainlen,:]
future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
sync_times = time_vals[trainlen:trainlen+synclen]
prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
sync_data = data[trainlen:trainlen+synclen]
modelsync.fit(ts)

# synchronise
if synchronisation==True:
    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
    
# predict
future_predictionsync = modelsync.predict(predictionlen)


#### plots #####
fig, axes = plt.subplots(2, figsize=(12, 6), sharex=True, tight_layout=True)
for i in range(data[0:trainlen].shape[1]):
  axes[i].plot(train_times, ss.inverse_transform(ts)[:, i],
                  linewidth=2,
                  label='Training data')
  axes[i].plot(prediction_times, ss.inverse_transform(future_predictionsync)[:,i],
                  linewidth=2,
                  label='Prediction')
  axes[i].plot(future_times, ss.inverse_transform(data[trainlen:trainlen+n_prediction+synclen])[:, i],
                  linewidth=2,
                  label='True')
  plt.legend()
axes[0].set_ylabel('KE')
axes[1].set_ylabel('q')
axes[0].set_xlabel('time')
axes[1].set_xlabel('time')
plt.savefig(output_path+'/timeseries_predictions.png')

fig2, ax = plt.subplots(1, 2, figsize = (12,6), tight_layout=True)
ax[0].scatter(ss.inverse_transform(future_predictionsync)[:,1], ss.inverse_transform(future_predictionsync)[:,0], cmap='viridis', marker='.', alpha=0.8, c=prediction_times, vmin=prediction_times[0], vmax=prediction_times[-1])
ax[1].scatter(ss.inverse_transform(data[trainlen:trainlen+n_prediction+synclen])[:, 1], ss.inverse_transform(data[trainlen:trainlen+n_prediction+synclen])[:, 0], cmap='viridis', marker='.', alpha=0.8, c=future_times, vmin=future_times[0], vmax=future_times[-1])
for i in range(2):
  ax[i].set_xlabel('q')
  ax[i].set_ylabel('KE')
  ax[i].set_ylim(-0.0001, 0.00035)
  ax[i].set_xlim(0.260,0.30)
ax[0].set_title('prediction')
ax[1].set_title('true')
plt.savefig(output_path+'/phase_space_predictions.png')

datanames = ['KE', 'q']
for i in range(2):
    print('\n', datanames[i])
    prediction_reshape = ss.inverse_transform(future_predictionsync)[:,i].reshape(len(future_predictionsync), 1)
    test_data_reshape = ss.inverse_transform(test_data)[:,i].reshape(len(future_predictionsync), 1)
    print('\n NRMSE for first 500 timesteps =', calculate_nrmse(prediction_reshape[:500-synclen] , test_data_reshape[:500-synclen]))
    print('\n NRMSE for 2000 timesteps=', calculate_nrmse(prediction_reshape[:] , test_data_reshape[:]))
    print('\n MSE for first 500 timesteps =', calculate_mse(prediction_reshape[:500-synclen] , test_data_reshape[:500-synclen]))
    print('\n MSE for 2000 timesteps=', calculate_mse(prediction_reshape[:] , test_data_reshape[:]))

v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=4000,
                                              n_test_timesteps=500,
                                              n_splits=4,
                                              metric=mean_squared_error)
for i in range(2):
    print('\n', datanames[i])
    prediction_reshape = ss.inverse_transform(future_predictionsync)[:,i]
    test_data_reshape = ss.inverse_transform(test_data)[:,i]
    print('\n delay for first 500 timesteps =', v.time_delay(test_data_reshape[:500-synclen], prediction_reshape[:500-synclen]))
    print('\n delay for 2000 timesteps=', v.time_delay(test_data_reshape[:], prediction_reshape[:]))