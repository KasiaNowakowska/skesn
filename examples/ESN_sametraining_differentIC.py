"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_train=<n_train>                number of training data points [default: 6000]
    --n_sync=<n_sync>                  number of data to synchronise with [default: 10]
    --n_prediction=<n_predcition>      number of timesteps for prediction [default: 2000]
    --synchronisation=<synchronisation>  turn on synchronisation [default: True]
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

data_dir = '/validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_beta{4:}_noise{5:}_bias{6:}_b{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), float(args['--beta']), args['--use_noise'], args['--use_bias'], args['--use_b'])

output_path = output_path1+data_dir
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
trainlen = n_train
ts = data[0:trainlen,:]
IC = np.arange(0,2000,500)
print(IC)
train_times = time_vals[0:trainlen]
dt = time_vals[1] - time_vals[0]
synclen = n_sync
predictionlen = n_prediction-synclen

MSE_values_q = []
MSE_values_ke = []
lag_values_q = []
lag_values_ke = []

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
last_endo = modelsync.last_endo_state_
last_res = modelsync.last_reservoir_state_
print(last_endo, last_res)

for i in IC:
    future_times = time_vals[trainlen+i:trainlen+i+synclen+predictionlen]
    sync_times = time_vals[trainlen+i:trainlen+i+synclen]
    prediction_times = time_vals[trainlen+i+synclen:trainlen+i+synclen+predictionlen]
    test_data = data[trainlen+i+synclen:trainlen+i+synclen+predictionlen, :]
    sync_data = data[trainlen+i:trainlen+i+synclen]
    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
    future_predictionsync = modelsync.predict(predictionlen)
    fig, axes = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
    for j in range(data[0:trainlen].shape[1]):
      axes[j].plot(train_times, ss.inverse_transform(ts)[:, j],
                      linewidth=2,
                      label='Training data')
      axes[j].plot(prediction_times, ss.inverse_transform(future_predictionsync)[:,j],
                      linewidth=2,
                      label='Prediction')
      axes[j].plot(future_times, ss.inverse_transform(data[trainlen+i:trainlen+i+predictionlen+synclen])[:, j],
                      linewidth=2,
                      label='True')
      plt.legend()
    axes[0].set_ylabel('KE')
    axes[1].set_ylabel('q')
    axes[1].set_xlabel('time')
    plt.xlim(4000,15000)
    axes[0].grid()
    axes[1].grid()
    fig.savefig(output_path+'/timeseries_predictions%i.png' % time_vals[i])
    plt.close()

    datanames = ['KE', 'q']
    for k in range(2):
        print('\n', datanames[k])
        prediction_reshape = ss.inverse_transform(future_predictionsync)[:,k].reshape(len(future_predictionsync), 1)
        test_data_reshape = ss.inverse_transform(test_data)[:,k].reshape(len(future_predictionsync), 1)
        print('\n NRMSE for first 500 timesteps =', calculate_nrmse(prediction_reshape[:500-synclen] , test_data_reshape[:500-synclen]))
        print('\n NRMSE for 2000 timesteps=', calculate_nrmse(prediction_reshape[:] , test_data_reshape[:]))
        print('\n MSE for first 500 timesteps =', calculate_mse(prediction_reshape[:500-synclen] , test_data_reshape[:500-synclen]))
        print('\n MSE for 2000 timesteps=', calculate_mse(prediction_reshape[:] , test_data_reshape[:]))
        if k == 0:
            MSE_values_ke.append(calculate_mse(prediction_reshape[:] , test_data_reshape[:]))
        elif k == 1:
            MSE_values_q.append(calculate_mse(prediction_reshape[:] , test_data_reshape[:]))
    for l in range(2):
        print('\n', datanames[l])
        prediction_reshape = ss.inverse_transform(future_predictionsync)[:,l]
        test_data_reshape = ss.inverse_transform(test_data)[:,l]
        print('\n delay for first 500 timesteps =', timedelay(test_data_reshape[:500-synclen], prediction_reshape[:500-synclen]))
        print('\n delay for 2000 timesteps=', timedelay(test_data_reshape[:], prediction_reshape[:]))
        if l == 0:
            lag_values_ke.append(timedelay(test_data_reshape[:], prediction_reshape[:]))
        if l == 1:
            lag_values_q.append(timedelay(test_data_reshape[:], prediction_reshape[:]))
            print(prediction_reshape, test_data_reshape)

avg_MSE_ke = np.mean(MSE_values_ke)
avg_MSE_q = np.mean(MSE_values_q)
avg_lag_ke = np.mean(lag_values_ke)
avg_lag_q = np.mean(lag_values_q)

simulation_details = {
            'n_reservoir': n_reservoir,
            'spectral_radius': spectral_radius,
            'sparsity': sparsity,
            'lambda_r': lambda_r,
            'use_additive_noise_when_forecasting': use_additive_noise_when_forecasting,
            'use_bias': use_bias,
            'use_b': use_b,
            'beta': beta,
            'avg_mse_ke': avg_MSE_ke,
            'avg_mse_q': avg_MSE_q,
            'avg_lag_ke': avg_lag_ke,
            'avg_lag_q': avg_lag_q
}

# Convert dictionary to DataFrame
df = pd.DataFrame([simulation_details])

# Save DataFrame to CSV
file_path = output_path1+'/simulation_results.csv'
with open(file_path, 'a', newline='') as f:
    df.to_csv(f, header=f.tell()==0, index=False) 

# Save hyperparameters to a text file
with open(output_path+'/esn_hyperparameters.txt', 'w') as f:
    for key, value in simulation_details.items():
        f.write(f"{key}: {value}\n")
