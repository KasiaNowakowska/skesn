"""
python script for ESN grid search with bayesian opt.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_trainLT=<n_train> --n_sync=<n_sync> --n_predictionLT=<n_prediction> --n_washoutLT=<n_washoutLT> --LT=<LT> --splits=<splits> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --ensembles=<ensembles>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_trainLT=<n_train>                number of training LTs [default: 10]
    --n_sync=<n_sync>                  number of data to synchronise with [default: 10]
    --n_predictionLT=<n_prediction>      number of prediction LTs [default: 3]
    --n_washoutLT=<n_washoutLT>             number of washout LTs [default: 1]
    --LT=<LT>                           Lyapunov time [default: 400]
    --splits=<splits>                   number of folds [default: 5]
    --synchronisation=<synchronisation>  turn on synchronisation [default: True]
    --n_reservoir=<n_reservoir>        size of reservoir [default: 2000]
    --use_noise=<use_noise>            turn noise on in update equation [default: True]
    --use_bias=<use_bias>              turn bias term on in u adds nother dimension to input [default: True]
    --use_b=<use_b>                    turn on extra bias term in actoiivation function [default: True] 
    --ensembles=<ensembles>            number of ensembles/how many times to rerun the ESN with the same IC and no retraining
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

import time
import os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from esn_old_adaptations import EsnForecaster

from scipy.signal import find_peaks

import h5py
from scipy import stats
from itertools import product

from docopt import docopt
args = docopt(__doc__)

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

from validation_stratergies import *

from sklearn.decomposition import PCA

import wandb
wandb.login()
    
input_path = args['--input_path']
output_path1 = args['--output_path']
n_trainLT = int(args['--n_trainLT'])
n_sync = int(args['--n_sync'])
n_predictionLT = int(args['--n_predictionLT'])
n_washoutLT = int(args['--n_washoutLT'])
LT = int(args['--LT'])
Splits = int(args['--splits'])
synchronisation = args['--synchronisation']
#n_reservoir = int(args['--n_reservoir'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']
ensembles = int(args['--ensembles'])

data_dir = '/WFV_wandb_ntrain{0:}_nprediction{1:}_nwashout{2:}_folds{3:}'.format(args['--n_trainLT'], args['--n_predictionLT'], args['--n_washoutLT'], args['--splits'])

output_path = output_path1+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')
    
def NRMSE(predictions, true_values):
    "input: predictions, true_values as (time, variables)"
    variables = predictions.shape[1]
    mse = np.mean((true_values-predictions) ** 2, axis = 1)
    #print(np.shape(mse))
    rmse = np.sqrt(mse)

    std_squared = np.std(true_values, axis = 0) **2
    print(np.shape(std_squared))
    sum_std = np.mean(std_squared)
    print(sum_std)
    sqrt_std = np.sqrt(sum_std)

    nrmse = rmse/sqrt_std
    #print(np.shape(nrmse))

    return nrmse

#### Hyperparmas #####
# WandB hyperparams settings
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "MSE_geom"},
    "parameters": {
        "n_reservoir":{"values": [2000, 4000, 6000]},
        "spectral_radius": {"max": 0.99, "min": 0.2},
        "sparsity": {"max": 0.99, "min": 0.3},
        "lambda_r": {"max": 0.5, "min": 0.01},
        "input_scaling": {"max": 1.5, "min": 0.5},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="WFV_2D")

##### load in data ######
q = np.load(input_path+'/q_vertical_allz.npy')
w = np.load(input_path+'/w_vertical_allz.npy')
total_time = np.load(input_path+'/total_time.npy')
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
print(len(q), len(w), len(total_time))

variables = num_variables = 2
variable_names = ['q', 'w']

data = np.zeros((len(total_time), len(x), 1, len(z), variables))
data[:,:,:,:,0] = q
data[:,:,:,:,1] = w
data = np.squeeze(data, axis=2)
    
# Print the shape of the combined array
print(data.shape)

num_snapshots = 5000
data_across_x = data[:num_snapshots, :, :, :]
data_matrix = data_across_x.reshape(num_snapshots, -1)
time_vals = total_time[:num_snapshots]

del q, w
del data

#### RUN PCA ####
pca = PCA(n_components=100)
# Fit and transform the data
data_reduced = pca.fit_transform(data_matrix)  # (5000, n_modes)
# Reconstruct the data
data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
#data_reconstructed_reshaped = ss.inverse_transform(data_reconstructed_reshaped)
data_reconstructed = data_reconstructed_reshaped.reshape(num_snapshots, len(x), len(z), num_variables)  # (5000, 256, 1, 2)
components = pca.components_
# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print('cumulative explained variance', cumulative_explained_variance[-1])

nrmse = NRMSE(data_reconstructed_reshaped, data_matrix)
avg_nrmse = np.mean(nrmse)
sq_avg_nrmse = avg_nrmse**2
Energy = 1-sq_avg_nrmse
curve = 1-Energy
print('Energy Error (curve value):', curve)

del data_matrix

def main():

    run = wandb.init()
    
    #### ESN ####
    print('starting ESN ...')
    dt = int(time_vals[1]-time_vals[0])
    print('dt=', dt)

    splits = Splits
    n_lyap = int(LT/dt)
    print('N_lyap=', n_lyap)
    synclen = n_sync
    n_prediction_inLT = n_predictionLT
    n_train_inLT = n_trainLT + n_washoutLT #current set up is the washout happens at start so need to add this washout into train time
    n_washout = n_washoutLT*n_lyap


    #### set hyperparameter values #####
    spectral_radius = wandb.config.spectral_radius
    sparsity = wandb.config.sparsity
    lambda_r = wandb.config.lambda_r
    input_scaling = wandb.config.input_scaling
    n_reservoir = wandb.config.n_reservoir

    param_grid = dict(n_reservoir=n_reservoir,
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
                      input_scaling=input_scaling,
                      n_washout=n_washout)

    print(param_grid)

    MSE_ens, PH_ens = grid_search_WFV_wandb(EsnForecaster, param_grid, data_reduced, time_vals, ensembles, splits, n_lyap, n_prediction_inLT, n_train_inLT, n_sync)



    MSE_mean = np.mean(MSE_ens)
    
    PH_mean_02 = np.mean(PH_ens[:,0])
    PH_mean_05 = np.mean(PH_ens[:,1])
    PH_mean_10 = np.mean(PH_ens[:,2])
    
    MSE_mean_geom = np.exp(np.mean(np.log(MSE_ens)))
    
    wandb.log({"MSE": MSE_mean,
              "PH_02": PH_mean_02,
              "PH_05": PH_mean_05,
              "PH_10": PH_mean_10,
              "MSE_geom": MSE_mean_geom
              })
              
wandb.agent(sweep_id, function=main, count=100)
    

