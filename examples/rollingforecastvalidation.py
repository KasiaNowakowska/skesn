"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b>] 

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_reservoir=<n_reservoir>        size of reservoir [default: 2000]
    --spectral_radius=<spectral radius> spectral radius of reservoir [default: 0.80]
    --sparsity=<sparsity>              sparsity of reservoir [default: 0.80]
    --lambda_r=<lambda_r>              noise added to update equation, only when use_noise is on [default: 1e-2]
    --beta=<beta>                      tikhonov regularisation parameter [default: 1e-3]
    --use_noise=<use_noise>            turn noise on in update equation [default: True]
    --use_bias=<use_bias>              turn bias term on in u adds nother dimension to input [default: True]
    --use_b=<use_b>                    turn on extra bias term in actoiivation function [default: True] 
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
n_reservoir = int(args['--n_reservoir'])
spectral_radius = float(args['--spectral_radius'])
sparsity = float(args['--sparsity'])
lambda_r = float(args['--lambda_r'])
beta = float(args['--beta'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']

data_dir = '/validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_beta{4:}_noise{5:}_bias{6:}_b{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), float(args['--beta']), args['--use_noise'], args['--use_bias'], args['--use_b'])

output_path = output_path+data_dir
print(output_path)
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

ss = StandardScaler()

data = ss.fit_transform(data)
print(np.shape(data))

# Plot short-term forecasting skill assessment based on rolling forecasting origin, both KE and q
trainlen = 14000
train_times = time_vals[0:trainlen]
ts = data[0:trainlen,:]
v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=6000,
                                              n_test_timesteps=2000,
                                              n_splits=4,
                                              metric=mean_squared_error,
                                              overlap=0,
                                              sync = False)
fig, axs = plt.subplots(3,2, figsize=(12, 6), sharex=True)
initial_test_times = []
MSE0, MSE1 = [], []
lag0, lag1 = [], []
lag_vals0 = []
for val in range(2):
  axs[0,val].plot(train_times, ss.inverse_transform(ts)[:, val], linewidth=2)
  axs[2,val].set_xlabel(r'$t$', fontsize=12)

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

for test_index, y_pred, y_true in v.prediction_generator(modelsync,
                                                          y=ts,
                                                          X=None):
    y_pred = ss.inverse_transform(y_pred)
    y_true = ss.inverse_transform(y_true)
    for val in range(2):
      axs[0,val].plot(time_vals[test_index], y_pred[:, val], color='tab:orange', linewidth=2)
      axs[0,val].plot([time_vals[test_index][0]], [y_pred[0, val]], 'o', color='tab:red')
    MSE0.append(v.metric(y_true[:, 0], y_pred[:, 0]))
    initial_test_times.append(time_vals[test_index][0])
    MSE1.append(v.metric(y_true[:, 1], y_pred[:, 1]))
    lag0.append(v.time_delay(y_true[:,0], y_pred[:,0]))
    lag1.append(v.time_delay(y_true[:,1], y_pred[:,1]))
axs[0,0].set_ylabel(r'KE', fontsize=12)
axs[0,1].set_ylabel(r'$q$', fontsize=12)
axs[1,0].semilogy(initial_test_times, MSE0, 'o--')
axs[1,1].semilogy(initial_test_times, MSE1, 'o--')
axs[2,0].plot(initial_test_times, lag0, 'o--')
axs[2,1].plot(initial_test_times, lag1, 'o--')
axs[1,0].set_ylabel(r'MSE KE', fontsize=12)
axs[2,0].set_ylabel(r'lag KE', fontsize=12)
axs[1,1].set_ylabel(r'MSE q', fontsize=12)
axs[2,1].set_ylabel(r'lag q', fontsize=12)
for i in range(3):
  for j in range(2):
    axs[i,j].grid()
plt.tight_layout()
plt.savefig(output_path+'/errors.png')
print('avg MSE KE', np.mean(MSE0))
print('avg MSE q',np.mean(MSE1))
print('avg lag KE', np.mean(lag0))
print('avg lag q',np.mean(lag1))

model_parameters = {
    'n_reservoir': modelsync.n_reservoir,
    'spectral_radius': modelsync.spectral_radius,
    'sparsity': modelsync.sparsity,
    'lambda_r': modelsync.lambda_r,
    'use_additive_noise_when_forecasting': modelsync.use_additive_noise_when_forecasting,
    'use_bias': modelsync.use_bias,
    'use_b': modelsync.use_b,
    'beta': modelsync.beta
}

# Save hyperparameters to a text file
with open(output_path+'/esn_hyperparameters.txt', 'w') as f:
    for key, value in model_parameters.items():
        f.write(f"{key}: {value}\n")

# Save MSE value to a separate text file
with open(output_path+'/mse_value.txt', 'w') as f:
    f.write(f"avg MSE KE: {np.mean(MSE0)}\n")
    f.write(f"avg MSE q: {np.mean(MSE1)}\n")
    f.write(f"avg lag KE: {np.mean(lag0)}\n")
    f.write(f"avg lag q: {np.mean(lag1)}\n")
