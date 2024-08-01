"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --input_scaling=<input_scaling> --ensembles=<ensembles>]

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
    --use_noise=<use_noise>            turn noise on in update equation [default: True]
    --use_bias=<use_bias>              turn bias term on in u adds nother dimension to input [default: True]
    --use_b=<use_b>                    turn on extra bias term in actoiivation function [default: True] 
    --input_scaling=<input_sclaing>    input scaling, sigma, so input weights are randomly generated between [-sigma, sigma] [default: 1]
    --ensembles=<ensembles>            number of ensembles/how many times to rerun the ESN with the same IC and no retraining
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

import time

from math import isclose, prod
import os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from esn_old_adaptations import EsnForecaster
from cross_validation import ValidationBasedOnRollingForecastingOrigin

from scipy.signal import find_peaks

import h5py
from scipy import stats
from itertools import product

from docopt import docopt
args = docopt(__doc__)

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

from validation_stratergies import *

def parse_list_argument_float(arg):
    if ',' in arg:
        # If arg contains commas, split it and convert each substring to an integer
        return [float(x) for x in arg.split(',')]
    else:
        # If arg doesn't contain commas, convert the whole argument to an integer
        return [float(arg)]

def parse_list_argument_int(arg):
    if ',' in arg:
        # If arg contains commas, split it and convert each substring to an integer
        return [int(x) for x in arg.split(',')]
    else:
        # If arg doesn't contain commas, convert the whole argument to an integer
        return [int(arg)]
    
input_path = args['--input_path']
output_path1 = args['--output_path']
n_train = int(args['--n_train'])
n_sync = int(args['--n_sync'])
n_prediction = int(args['--n_prediction'])
synchronisation = args['--synchronisation']
n_reservoir = parse_list_argument_int(args['--n_reservoir'])
spectral_radius = parse_list_argument_float(args['--spectral_radius'])
sparsity = parse_list_argument_float(args['--sparsity'])
lambda_r = parse_list_argument_float(args['--lambda_r'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']
input_scaling = parse_list_argument_float(args['--input_scaling'])
ensembles = int(args['--ensembles'])

print(sparsity)

data_dir = '/SSV_retrained_validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_noise{4:}_bias{5:}_b{6:}_input_scaling{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], args['--lambda_r'], args['--use_noise'], args['--use_bias'], args['--use_b'], args['--input_scaling'])

output_path = output_path1+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

#load in data
q = np.load(input_path+'/q5000_30000.npy')
ke = np.load(input_path+'/KE5000_30000.npy')
evap = np.load(input_path+'/evap5000_30000.npy')
time_vals = np.load(input_path+'/total_time5000_30000.npy')
print(len(q), len(ke), len(time_vals))

ke_column = ke.reshape(len(ke), 1)
q_column = q.reshape(len(q), 1)
evap_column = evap.reshape(len(evap), 1)

data = np.hstack((ke_column, q_column, evap_column))

# Print the shape of the combined array
print(data.shape)

ss = StandardScaler()

data = ss.fit_transform(data)
print(np.shape(data))


param_grid = dict(n_reservoir=n_reservoir,
                  spectral_radius=spectral_radius,
                  sparsity=sparsity,
                  regularization=['l2'],
                  lambda_r=lambda_r,
                  in_activation=['tanh'],
                  out_activation=['identity'],
                  use_additive_noise_when_forecasting=[use_additive_noise_when_forecasting],
                  random_state=[42],
                  use_bias=[use_bias],
                  use_b=[use_b],
                  input_scaling=input_scaling)

print(param_grid)
trainlen = n_train
synclen = n_sync
predictionlen = n_prediction

# Generate all combinations of parameters
# Get the parameter labels and values
param_labels = list(param_grid.keys())
param_values = list(param_grid.values())

param_combinations = list(product(*param_values))

# Create a DataFrame where each row is a different set of parameters
df = pd.DataFrame(param_combinations, columns=param_labels)

print(df)

MSE_params, PH_params = grid_search_SSV_retrained(EsnForecaster, param_grid, data, time_vals, ensembles, n_prediction, trainlen, synclen)

#np.save(output_path+'/prediction_data.npy', prediction_data)

MSE_mean = np.zeros((MSE_params.shape[0]))
for i in range(MSE_params.shape[0]):
    MSE_mean[i] = np.mean(MSE_params[i,:])
    
PH_mean = np.zeros((PH_params.shape[0]))
for i in range(PH_params.shape[0]):
    PH_mean[i] = np.mean(PH_params[i,:])

fig, ax =plt.subplots(1, figsize=(12,10), tight_layout=True)
ax.boxplot(MSE_params.T, meanline=True, showmeans=True)
ax.set_ylabel('MSE')

the_table = ax.table(cellText=df.values.T,
                    rowLabels=df.columns,
                    loc='bottom',
                    bbox=[0, -0.75, 1, 0.65])

fig.savefig(output_path+'/boxplots.png')

fig2, ax2 = plt.subplots(1, figsize=(12,10), tight_layout=True)
ax2.boxplot(PH_params.T, meanline=True, showmeans=True)
ax2.set_ylabel('PH')

the_table = ax2.table(cellText=df.values.T,
                    rowLabels=df.columns,
                    loc='bottom',
                    bbox=[0, -0.75, 1, 0.65])

fig2.savefig(output_path+'/boxplotsPH.png')

np.save(output_path+'/MSE_means.npy', MSE_mean)
np.save(output_path+'/PH_means.npy', PH_mean)

df_copy = df.copy(deep=True)
df_copy['mean_MSE'] = MSE_mean
df_copy['mean_PH'] = PH_mean

# Save df_copy as CSV
df_copy.to_csv(output_path+'/gridsearch_dataframe.csv', index=False)
print('data frame saved')

