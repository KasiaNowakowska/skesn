"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_trainLT=<n_train> --n_sync=<n_sync> --n_predictionLT=<n_prediction> --n_washoutLT=<n_washoutLT> --LT=<LT> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --input_scaling=<input_scaling>  --ensembles=<ensembles> --index_start=<index_start>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
    --n_trainLT=<n_train>                number of training LTs [default: 10]
    --n_sync=<n_sync>                  number of data to synchronise with [default: 10]
    --n_predictionLT=<n_prediction>      number of prediction LTs [default: 3]
    --n_washoutLT=<n_washoutLT>             number of washout LTs [default: 1]
    --LT=<LT>                           Lyapunov time [default: 400]
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
    --index_start=<index_start>        starting index for timeseries
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

from esn_old_adaptations import EsnForecaster

from scipy.signal import find_peaks

from validation_stratergies import *
import evaluation_metrics as evalm

import h5py

from docopt import docopt
args = docopt(__doc__)

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')
    
input_path = args['--input_path']
output_path1 = args['--output_path']
n_trainLT = int(args['--n_trainLT'])
n_sync = int(args['--n_sync'])
n_predictionLT = int(args['--n_predictionLT'])
n_washoutLT = int(args['--n_washoutLT'])
LT = int(args['--LT'])
synchronisation = args['--synchronisation']
n_reservoir = int(args['--n_reservoir'])
spectral_radius = float(args['--spectral_radius'])
sparsity = float(args['--sparsity'])
lambda_r = float(args['--lambda_r'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']
ensembles = int(args['--ensembles'])
index_start = int(args['--index_start'])
input_scaling = float(args['--input_scaling'])

data_dir = '/testing_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_noise{4:}_bias{5:}_b{6:}_start_index{7:}_input_scaling{8:}__n_trainLT{9:}_n_predictionLT{10:}_n_washoutLT{11:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), args['--use_noise'], args['--use_bias'], args['--use_b'], args['--index_start'], float(args['--input_scaling']), args['--n_trainLT'], args['--n_predictionLT'], args['--n_washoutLT'])

output_path = output_path1+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

output_path_images = output_path+'/Images'
print(output_path_images)
if not os.path.exists(output_path_images):
    os.makedirs(output_path_images)
    print('made directory')

#load in data
q = np.load(input_path+'/q5000_30000.npy')
ke = np.load(input_path+'/KE5000_30000.npy')
time_vals = np.load(input_path+'/total_time5000_30000.npy')
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
dt = int(time_vals[1]-time_vals[0])
print('dt=', dt)

n_lyap = int(LT/dt)
print('N_lyap=', n_lyap)
synclen = n_sync
n_prediction_inLT = n_predictionLT
n_train_inLT = n_trainLT + n_washoutLT #current set up is the washout happens at start so need to add this washout into train time
n_washout = n_washoutLT*n_lyap

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
            input_scaling=input_scaling)

# Plot long-term prediction


N_train = n_train_inLT*n_lyap
N_val = N_test = n_prediction_inLT*n_lyap
test_index_start = index_start - n_train_inLT*n_lyap

U_train = data[test_index_start:test_index_start+N_train-synclen]
times_train = time_vals[test_index_start:test_index_start+N_train-synclen]

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
            input_scaling=input_scaling)
modelsync.fit(U_train)

test_index = test_index_start
#test_indices = np.arange(test_index, test_index+15*n_lyap, 50)
test_indices = np.arange(test_index, test_index+3*n_lyap, 100)
IC_number = len(test_indices)
print(test_indices)
print('IC_number:', IC_number)

variables = len(data[1])

ensemble_all_vals = np.zeros((N_test, variables, IC_number, ensembles))
ensemble_all_vals_unscaled = np.zeros((N_test, variables, IC_number, ensembles))

inspect_graphs = True

for index, test_index in enumerate(test_indices):
    print(test_index)
    y_test = data[test_index:test_index+N_test, :]
    print('y_test=', np.shape(y_test))
    U_test_sync = data[test_index-synclen:test_index,:]
    times_test = time_vals[test_index:test_index+N_test]
    
    # run through ensembles
    for i in range(ensembles):
        print('ensemble number =', i)
        #synchronise data
        modelsync._update(U_test_sync, UpdateModes = UpdateModes.synchronization)
        #predict
        future_predictionsync = modelsync.predict(N_test)

        #inverse scalar
        inverse_prediction = ss.inverse_transform(future_predictionsync)
        inverse_test_data = ss.inverse_transform(y_test)

        for v in range(variables):
            ensemble_all_vals_unscaled[:, v, index, i] = future_predictionsync[:,v]
            ensemble_all_vals[:, v, index, i] = inverse_prediction[:,v]

print(np.shape(ensemble_all_vals))
np.save(output_path+'/ensemble%i_all_vals.npy' % ensembles, ensemble_all_vals)
np.save(output_path+'/ensemble%i_all_vals_unscaled.npy' % ensembles, ensemble_all_vals_unscaled)


variable_names = ['KE', 'q']
if inspect_graphs == True:
    for index, test_index in enumerate(test_indices):
        if index % 4 == 0:
            fig, ax = plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
            evalm.ensemble_timeseries(ensemble_all_vals[:,:,index,:], ss.inverse_transform(data[test_index:test_index+N_test,:]), time_vals[test_index:test_index+N_test], variables, variable_names, ax=ax)
            ax[1].legend()
            fig.savefig(output_path_images+'/timeseries_ens%i.png' % test_index)
            plt.close()

    fig, ax =plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    for index, index_start in enumerate(test_indices):
        if index % 5 == 0:
            evalm.ensemble_timeseries(ensemble_all_vals_IC[:,:,index,:], ss.inverse_transform(data[index_start:index_start+N_test, :]), time_vals[index_start:index_start+N_test], variables, variable_names, ax=ax)
    ax[0].plot(time_vals[test_indices[0]:test_indices[-1]+3*n_lyap], ss.inverse_transform(data[test_indices[0]:test_indices[-1]+3*n_lyap, 0]), alpha=0.2)
    ax[1].plot(time_vals[test_indices[0]:test_indices[-1]+3*n_lyap], ss.inerse_transfrom(data[test_indices[0]:test_indices[-1]+3*n_lyap, 1]), alpha=0.2)
    ax[1].set_xlim(time_vals[test_indices[0]], time_vals[test_indices[-1]+3*n_lyap])
    fig.savefig(output_path_images+'/timeseries_ens_sample.png')
    plt.close()






