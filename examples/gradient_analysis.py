"""
python script for ESN grid search.

Usage: grid_search.py [--input_path=<input_path> --output_path=<output_path>]

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

trainlen = 10000
train_times = time_vals[0:trainlen]
ts = data[0:trainlen,:]

def differentiate_time_series(series):
    # Compute the differences between adjacent elements
    differences = np.diff(series)
    
    # Compute the gradients
    gradients = differences / np.diff(range(len(series)))
    
    return gradients

def determine_trend(gradients):
    trend_array = []
    for gradient in gradients:
        if gradient > 0:
            trend_array.append(1)
        elif gradient < 0:
            trend_array.append(-1)
        else:
            trend_array.append(0)
    return trend_array

gradients_ke = np.array(differentiate_time_series(ts[:,0]))
gradients_q = np.array(differentiate_time_series(ts[:,1]))

# Determine the trend at each point
trend_array_ke = np.array(determine_trend(gradients_ke))
trend_array_q = np.array(determine_trend(gradients_q))

print(np.shape(trend_array_ke), len(trend_array_q))

# Reshape the arrays into column vectors
trend_array_ke_column = trend_array_ke.reshape(len(trend_array_ke), 1)
trend_array_q_column = trend_array_q.reshape(len(trend_array_q), 1)

gradients_ke_column = gradients_ke.reshape(len(gradients_ke), 1)
gradients_q_column = gradients_q.reshape(len(gradients_q), 1)

#ke_column = np.sqrt(ke_column)

# Concatenate the column vectors horizontally
data_trends = np.hstack((trend_array_ke_column, trend_array_q_column))
data_gradients = np.hstack((gradients_ke_column, gradients_q_column))
# Print the shape of the combined array
print(data_trends.shape)
print(data_gradients.shape)

# Plot short-term forecasting skill assessment based on rolling forecasting origin training data stays the same size, both KE and q
v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=6000,
                                              n_test_timesteps=1000,
                                              n_splits=4,
                                              metric=mean_squared_error,
                                              overlap=100)
fig, axs = plt.subplots(2,2, figsize=(12, 6), sharex=True, tight_layout=True)
initial_test_times = []
gradients0, gradients1 = [], []
MSE0, MSE1 = [], []
lag0, lag1 = [], []
lag_vals0 = []

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
                use_bias=True,
                use_b=True,
                beta=1e-2)

for test_index, y_pred, y_true in v.prediction_generator_overlap(modelsync,
                                                          y=ts,
                                                          X=None):
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
                use_bias=True,
                use_b=True,
                beta=1e-2)
    y_pred = ss.inverse_transform(y_pred)
    y_true = ss.inverse_transform(y_true)
    MSE0.append(v.metric(y_true[:, 0], y_pred[:, 0]))
    lag0.append(v.time_delay(y_true[:,0], y_pred[:,0]))
    initial_test_times.append(time_vals[test_index][0])
    MSE1.append(v.metric(y_true[:, 1], y_pred[:, 1]))
    lag1.append(v.time_delay(y_true[:,1], y_pred[:,1]))
    gradients_both=data_gradients[test_index[0],:]
    gradients0.append(gradients_both[0])
    gradients1.append(gradients_both[1])
    print(test_index[0])
axs[0,0].scatter(gradients0, MSE0)
axs[0,1].scatter(gradients1, MSE1)
axs[1,0].scatter(gradients0, lag0)
axs[1,1].scatter(gradients1, lag1)
axs[0,0].set_ylabel(r'MSE KE', fontsize=12)
axs[1,0].set_ylabel(r'lag KE', fontsize=12)
axs[0,1].set_ylabel(r'MSE q', fontsize=12)
axs[1,1].set_ylabel(r'lag q', fontsize=12)
axs[0,0].set_xlim(-max(abs(min(gradients0)), abs(max(gradients0))), max(abs(min(gradients0)), abs(max(gradients0))))
axs[0,1].set_xlim(-max(abs(min(gradients1)), abs(max(gradients1))), max(abs(min(gradients1)), abs(max(gradients1))))
for i in range(2):
  for j in range(2):
    axs[i,j].set_xlabel('gradient')
    axs[i,j].grid()
plt.tight_layout()
plt.suptitle('errors over 1000 timesteps of test data')
plt.savefig(output_path+'/gradients.png')