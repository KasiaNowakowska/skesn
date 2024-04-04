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

trainlen = 14000
train_times = time_vals[0:trainlen]
ts = data[0:trainlen,:]
v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=6000,
                                              n_test_timesteps=2000,
                                              n_splits=4,
                                              metric=mean_squared_error,
                                              overlap=0,
                                              sync = False)
                                              
summary, best_model = v.grid_search(modelsync,
                                    param_grid=dict(
                                    n_reservoir=[2000,4000,6000]),
                                    y=ts,
                                    X=None)
                                    
summary_df = pd.DataFrame(summary).sort_values('rank_test_score')
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
table_rows = []
param_names = list(summary_df.iloc[0]['params'].keys())
ranks = []
test_scores = []
for i in range(len(summary_df)):
    ranks.append(int(summary_df.iloc[i]['rank_test_score']))
    table_rows.append(list(summary_df.iloc[i]['params'].values()))
    test_scores.append(np.abs(np.array([float(summary_df.iloc[i][f'split{j}_test_score']) for j in range(5)])))
ax.boxplot(test_scores)
ax.set_yscale('log')
ax.set_xticks([])
ax.set_ylabel('MSE', fontsize=12)
ax.grid()
table_rows = [*zip(*table_rows)]
the_table = ax.table(cellText=table_rows,
                      rowLabels=param_names,
                      colLabels=ranks,
                      loc='bottom')
plt.tight_layout()
plt.savefig('summary.png')