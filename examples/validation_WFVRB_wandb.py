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

#### Hyperparmas #####
# WandB hyperparams settings
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "PH_10"},
    "parameters": {
        "n_reservoir":{"values": [200, 500, 1000, 2500]},
        "spectral_radius": {"max": 0.99, "min": 0.01},
        "sparsity": {"max": 0.99, "min": 0.01},
        "lambda_r": {"max": 1.00, "min": 0.001},
        "input_scaling": {"max": 2.0, "min": 0.2},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="WFV_global")

def main():

    run = wandb.init()

    ##### load in data ######
    q = np.load(input_path+'/q5000_30000.npy')
    ke = np.load(input_path+'/KE5000_30000.npy')
    evap = np.load(input_path+'/evap5000_30000.npy')
    time_vals = np.load(input_path+'/total_time5000_30000.npy')
    print(len(q), len(ke), len(time_vals))

    ke_column = ke.reshape(len(ke), 1)
    q_column = q.reshape(len(q), 1)
    evap_column = evap.reshape(len(evap), 1)
    
    data = np.hstack((ke_column, q_column))
    
    # Print the shape of the combined array
    print(data.shape)
    
    ss = StandardScaler()
    
    data = ss.fit_transform(data)
    #data = data[::2]
    #time_vals = time_vals[::2]
    dt = int(time_vals[1]-time_vals[0])
    print('dt=', dt)
    print(np.shape(data))

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

    MSE_ens, PH_ens = grid_search_WFV_wandb(EsnForecaster, param_grid, data, time_vals, ensembles, splits, n_lyap, n_prediction_inLT, n_train_inLT, n_sync)



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
    

