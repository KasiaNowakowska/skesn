"""
python script for ESN grid search with bayesian opt.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_trainLT=<n_train> --n_sync=<n_sync> --n_predictionLT=<n_prediction> --n_washoutLT=<n_washoutLT> --LT=<LT> --splits=<splits> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --modes=<modes> --ensembles=<ensembles>]

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
    --modes=<modes>                    number of modes in POD [default: 100]
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
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

from sklearn.decomposition import PCA, IncrementalPCA

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
modes = int(args['--modes'])
ensembles = int(args['--ensembles'])

data_dir = '/WFV_wandb_ntrain{0:}_nprediction{1:}_nwashout{2:}_folds{3:}_modes{4:}'.format(args['--n_trainLT'], args['--n_predictionLT'], args['--n_washoutLT'], args['--splits'], args['--modes'])

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
    "metric": {"goal": "maximize", "name": "PH_02"},
    "parameters": {
        "n_reservoir":{"values": [4000, 6000, 8000]},
        "spectral_radius": {"max": 0.99, "min": 0.2},
        "sparsity": {"max": 0.99, "min": 0.5},
        "lambda_r": {"max": 0.5, "min": 0.05},
        "input_scaling": {"max": 1.5, "min": 0.5},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="WFV_2D")

##### load in data ######
'''
#### smaller data 5000-15000 ####
q = np.load(input_path+'/q_vertical_allz.npy')
w = np.load(input_path+'/w_vertical_allz.npy')
total_time = np.load(input_path+'/total_time.npy')
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
print(len(q), len(w), len(total_time))

data = np.zeros((len(total_time), len(x), 1, len(z), variables))
data[:,:,:,:,0] = q - np.mean(q, axis=0)
data[:,:,:,:,1] = w - np.mean(w, axis=0)
data = np.squeeze(data, axis=2)

num_snapshots = 5000
data_across_x = data[:num_snapshots, :, :, :]
data_matrix = data_across_x.reshape(num_snapshots, -1)
time_vals = total_time[:num_snapshots]
'''

#### larger data 5000-30000 ####
'''
total_time = np.load(input_path+'/total_time5000_30000.npy')
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
q = np.load(input_path+'/q5000_30000.npy')
w = np.load(input_path+'/w5000_30000.npy')
print(len(q), len(w), len(total_time))

variables = num_variables = 2
variable_names = ['q', 'w']
data = np.zeros((len(total_time), len(x), 1, len(z), variables))
data[:,:,:,:,0] = q - np.mean(q, axis=0)
data[:,:,:,:,1] = w - np.mean(w, axis=0)
data = np.squeeze(data, axis=2)
# Print the shape of the combined array
print(data.shape)

num_snapshots = 12500
data_across_x = data[:num_snapshots, :, :, :]
data_matrix = data_across_x.reshape(num_snapshots, -1)
time_vals = total_time[:num_snapshots]


del q, w
del data
del total_time
'''
#### larger data 5000-30000 hf ####
num_snapshots = 10000 #12500
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']

with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
    time_vals = np.array(df['total_time_all'][:num_snapshots])
    q = np.array(df['q_all'][:num_snapshots])
    w = np.array(df['w_all'][:num_snapshots])

    q = np.squeeze(q, axis=2)
    w = np.squeeze(w, axis=2)

    q_mean = np.mean(q, axis=0)
    w_mean = np.mean(w, axis=0)
    print(q_mean, w_mean)

    q_array = q - q_mean
    w_array = w - w_mean
    print(np.shape(q_array))

print('shape of time_vals', np.shape(time_vals))

q_array = q_array.reshape(len(time_vals), len(x), len(z), 1)
w_array = w_array.reshape(len(time_vals), len(x), len(z), 1)

data_full = np.concatenate((q_array, w_array), axis=-1)
print('shape of data_full:', np.shape(data_full))

del q_array
del w_array

data_across_x = data_full[:num_snapshots]
time_vals = time_vals[:num_snapshots]
print('shape of data:', np.shape(data_across_x))

del data_full

data_matrix = data_across_x.reshape(num_snapshots, -1)
print('shape of data_matrix:', np.shape(data_matrix))



#### RUN PCA ####
pca = PCA(n_components=modes)
# Fit and transform the data
data_reduced = pca.fit_transform(data_matrix)  # (5000, n_modes)
# Reconstruct the data
data_reconstructed_reshaped = pca.inverse_transform(data_reduced)  # (5000, 256 * 2)
#data_reconstructed_reshaped = ss.inverse_transform(data_reconstructed_reshaped)
data_reconstructed = data_reconstructed_reshaped.reshape(num_snapshots, len(x), len(z), num_variables)  # (5000, 256, 1, 2)
components = pca.components_
# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_


'''
#### RUN IPCA ####
print('run ipca')
batch_size=500
ipca = IncrementalPCA(n_components=modes)
with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
    data_reduced = np.zeros((num_snapshots, modes), dtype='float64')
    nrmse = np.zeros((num_snapshots))
    for j in range(0, num_snapshots, batch_size):
        #load batch
        batch_q = df['q_all'][j:j+batch_size] - q_mean
        batch_w = df['w_all'][j:j+batch_size] - w_mean
        # reshape array
        batch_q = batch_q.reshape(batch_size, len(x), 1, len(z), 1)
        batch_w = batch_w.reshape(batch_size, len(x), 1, len(z), 1)
        # concatenate and flatten
        data = np.concatenate((batch_q, batch_w), axis=-1)
        data_across_x = np.squeeze(data, axis=2)
        data_matrix = data_across_x.reshape(batch_size, -1)
        ipca.partial_fit(data_matrix)
        data_reduced[j:j + batch_size] = ipca.transform(data_matrix)

        #### reconstruction error for batch ####
        data_reconstructed_reshaped_batch = ipca.inverse_transform(data_reduced[j:j + batch_size])
        nrmse_batch = NRMSE(data_reconstructed_reshaped_batch, data_matrix)
        nrmse[j: j + batch_size] = nrmse_batch

data_reconstructed_reshaped = ipca.inverse_transform(data_reduced)
data_reconstructed = data_reconstructed_reshaped.reshape(num_snapshots, len(x), len(z), variables)
# Get the explained variance ratio
explained_variance_ratio = ipca.explained_variance_ratio_
'''

inspect_plots = True
if inspect_plots == True:
    # Plot original vs reconstructed snapshot for a height
    fig, ax =plt.subplots(3,variables, figsize=(12,9), tight_layout=True, sharex=True)

    z_index=32
    error = error = data_across_x[:, :, z_index, :] - data_reconstructed[:, :, z_index, :]
    for v in range(variables):
        minm = min(np.min(data_across_x[:, :, z_index, v]), np.min(data_reconstructed[:, :, z_index, v]))
        maxm = max(np.max(data_across_x[:, :, z_index, v]), np.max(data_reconstructed[:, :, z_index, v]))
        print(minm, maxm)
        c1 = ax[0,v].pcolormesh(time_vals[:], x, data_across_x[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c2 = ax[1,v].pcolormesh(time_vals[:], x, data_reconstructed[:, :, z_index, v].T, vmin=minm, vmax=maxm)
        c3 = ax[2,v].pcolormesh(time_vals[:], x, error[:, :, v].T, cmap='RdBu', norm=TwoSlopeNorm(vmin=-np.max(np.abs(error)), vcenter=0, vmax=np.max(np.abs(error))))
        ax[0,v].set_title(variable_names[v])

        fig.colorbar(c1, label='True')
        fig.colorbar(c2, label='reconstructed')
        fig.colorbar(c3, label='error')

    for i in range(2):
        ax[i,0].set_ylabel('x')
    for i in range(variables):
        ax[0,i].set_xlabel('time')

    ax[-1,-1].set_xlim(6000,8000)
    fig.savefig(output_path+'/reconstruction.png')

    ##### GLOBAL #####
    groundtruth_avgq = np.mean(data_across_x[:,:,:,0], axis=(1,2))
    groundtruth_ke = 0.5*data_across_x[:,:,:,1]*data_across_x[:,:,:,1]
    groundtruth_avgke = np.mean(groundtruth_ke, axis=(1,2))
    groundtruth_global = np.zeros((num_snapshots,2))
    groundtruth_global[:,0] = groundtruth_avgke
    groundtruth_global[:,1] = groundtruth_avgq

    reconstructed_groundtruth_avgq = np.mean(data_reconstructed[:,:,:,0], axis=(1,2))
    reconstructed_groundtruth_ke = 0.5*data_reconstructed[:,:,:,1]*data_reconstructed[:,:,:,1]
    reconstructed_groundtruth_avgke = np.mean(reconstructed_groundtruth_ke, axis=(1,2))

    reconstructed_groundtruth_global = np.zeros((num_snapshots,2))
    reconstructed_groundtruth_global[:,0] = reconstructed_groundtruth_avgke
    reconstructed_groundtruth_global[:,1] = reconstructed_groundtruth_avgq

    fig, ax = plt.subplots(2,1, figsize=(12,6), tight_layout=True)
    for i in range(2):
        ax[i].plot(time_vals, groundtruth_global[:,i], color='tab:blue', label='truth')
        ax[i].plot(time_vals, reconstructed_groundtruth_global[:,i], color='tab:orange', label='reconstruction')
    ax[1].legend()
    #ax[1].set_xlim()


    print('shape of data for ESN', np.shape(data_reduced))

#### varaince, energy and errors in reconstruction ####
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print('cumulative explained variance', cumulative_explained_variance[-1])

nrmse_batch = NRMSE(data_reconstructed_reshaped, data_matrix)
avg_nrmse = np.mean(nrmse)
sq_avg_nrmse = avg_nrmse**2
Energy = 1-sq_avg_nrmse
curve = 1-Energy
print('Energy Error (curve value):', curve)

del data_reconstructed_reshaped 
del data_reconstructed
'''
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
    
'''
