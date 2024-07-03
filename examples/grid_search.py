"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --input_scaling=<input_scaling> --ensembles=<ensembles>]

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
    --beta=<beta>                      tikhonov regularisation parameter [default: 1e-3]
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

#Define functions
def PH_error(predictions, true_values):
    "input predictions, true_values as (time, variables)"
    variables = predictions.shape[1]
    numerator = np.zeros((predictions.shape[0]))
    norm = np.zeros((predictions.shape[0]))
    for i in range(true_values.shape[0]):
        numerator[i] = np.linalg.norm(true_values[i,:] - predictions[i,:])
        norm[i] = np.linalg.norm(true_values[i,:])
    #denominator = np.max(true_values) - np.min(true_values)

    norm_squared = np.mean(norm**2)
    denominator = np.sqrt(norm_squared)

    PH = numerator/denominator

    return PH

def prediction_horizon(predictions, true_values, threshold=1e-2):
    error = PH_error(predictions, true_values)
    #print(np.shape(np.where(error > threshold)))
    shape = np.shape(np.where(error > threshold))
    if shape[1] == 0:
        PredictabilityHorizon = predictions.shape[0]
    else:
        PredictabilityHorizon = np.where(error > threshold)[0][0]

    return PredictabilityHorizon

def MSE(predictions, true_values):
    Nu = predictions.shape[1]
    norm = np.zeros((predictions.shape[0]))
    for i in range(true_values.shape[0]):
        norm[i] = np.linalg.norm(true_values[i,:] - predictions[i,:])

    norm_squared = np.mean(norm**2)
    MSE = 1/Nu * norm_squared

    return MSE

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
beta = parse_list_argument_float(args['--beta'])
use_additive_noise_when_forecasting = args['--use_noise']
use_bias = args['--use_bias']
use_b = args['--use_b']
input_scaling = parse_list_argument_float(args['--input_scaling'])
ensembles = int(args['--ensembles'])

print(sparsity)

data_dir = '/validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_beta{4:}_noise{5:}_bias{6:}_b{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], args['--lambda_r'], args['--beta'], args['--use_noise'], args['--use_bias'], args['--use_b'])

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
                  input_scaling=input_scaling,
                  beta=beta)

print(param_grid)
trainlen = n_train
synclen = n_sync
predictionlen = n_prediction

def grid_search_SSV_retrained(forecaster, param_grid, data, time_vals, ensembles, n_prediction, n_train, n_sync):
    print(param_grid)

    synclen = n_sync
    trainlen = n_train - synclen
    predictionlen = n_prediction
    ts = data[0:trainlen,:]
    train_times = time_vals[0:trainlen]
    dt = time_vals[1] - time_vals[0]
    future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
    sync_times = time_vals[trainlen:trainlen+synclen]
    prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
    test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
    sync_data = data[trainlen:trainlen+synclen]
    future_data = data[trainlen:trainlen+synclen+predictionlen, :]

    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())

    PH_params = np.zeros((num_combinations, ensembles))
    MSE_params = np.zeros((num_combinations, ensembles))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        #print(param_dict)
        modelsync = forecaster(**param_dict)
        print(modelsync)
        # Now you can use `modelsync` with the current combination of parameters

        ensemble_all_vals_ke = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals_q = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals = np.zeros((len(prediction_times), 2, ensembles))
        MSE_ens = np.zeros((ensembles))
        PH_ens = np.zeros((ensembles))

        for i in range(ensembles):
            print('ensemble member =', i)
            modelsync.fit(ts)
            #synchronise data
            modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
            #predict
            future_predictionsync = modelsync.predict(predictionlen)

            #inverse scalar
            inverse_training_data = ss.inverse_transform(ts)
            inverse_prediction = ss.inverse_transform(future_predictionsync)
            inverse_test_data = ss.inverse_transform(test_data)
            inverse_future_data = ss.inverse_transform(future_data) #this include the sync data

            ### ensemble mean ###
            ensemble_all_vals_ke[:,i] = future_predictionsync[:,0]
            ensemble_all_vals_q[:,i] = future_predictionsync[:,1]
            ensemble_all_vals[:, 0, i] = future_predictionsync[:,0]
            ensemble_all_vals[:, 1, i] = future_predictionsync[:,1]

            mse = MSE(ensemble_all_vals[:,:,i], test_data[:,:])
            MSE_ens[i] = mse
            ph = prediction_horizon(ensemble_all_vals[:,:,i], test_data[:,:], threshold = 0.1)
            PH_ens[i] = ph

        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:] = PH_ens
        counter += 1
    return MSE_params, PH_params

def grid_search_SSV(forecaster, param_grid, data, time_vals, ensembles, n_prediction, trainlen, synclen):
    print(param_grid)

    synclen = n_sync
    trainlen = n_train - synclen
    predictionlen = n_prediction
    ts = data[0:trainlen,:]
    train_times = time_vals[0:trainlen]
    dt = time_vals[1] - time_vals[0]
    future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
    sync_times = time_vals[trainlen:trainlen+synclen]
    prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
    test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
    sync_data = data[trainlen:trainlen+synclen]
    future_data = data[trainlen:trainlen+synclen+predictionlen, :]
    variables = data.shape[-1]
    print('number of var =', variables)

    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())

    PH_params = np.zeros((num_combinations, ensembles))
    MSE_params = np.zeros((num_combinations, ensembles))
    prediction_data = np.zeros((num_combinations, len(prediction_times), variables, ensembles))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        #print(param_dict)
        modelsync = forecaster(**param_dict)
        print(modelsync)
        # Now you can use `modelsync` with the current combination of parameters

        modelsync.fit(ts)

        ensemble_all_vals_ke = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals_q = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals = np.zeros((len(prediction_times), variables, ensembles))
        ensemble_all_vals_unscaled = np.zeros((len(prediction_times), variables, ensembles))
        MSE_ens = np.zeros((ensembles))
        PH_ens = np.zeros((ensembles))

        for i in range(ensembles):
            print('ensemble member =', i)
            #synchronise data
            modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
            #predict
            future_predictionsync = modelsync.predict(predictionlen)
            if i==0:
                print(np.shape(future_predictionsync))

            #inverse scalar
            inverse_training_data = ss.inverse_transform(ts) 
            inverse_prediction = ss.inverse_transform(future_predictionsync)
            inverse_test_data = ss.inverse_transform(test_data)
            inverse_future_data = ss.inverse_transform(future_data) #this include the sync data

            ### ensemble mean ###
            for v in range(variables):
                ensemble_all_vals_unscaled[:, v, i] = future_predictionsync[:,v]
                ensemble_all_vals[:, v, i] = inverse_prediction[:,v]

            mse = MSE(ensemble_all_vals_unscaled[:,:,i], test_data[:,:])
            MSE_ens[i] = mse
            ph = prediction_horizon(ensemble_all_vals_unscaled[:,:,i], test_data[:,:], threshold = 0.2)
            PH_ens[i] = ph

        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:] = PH_ens
        prediction_data[counter, :, :, :] = ensemble_all_vals[:, :, :]
        counter += 1
    return MSE_params, PH_params, prediction_data

# Generate all combinations of parameters
# Get the parameter labels and values
param_labels = list(param_grid.keys())
param_values = list(param_grid.values())

param_combinations = list(product(*param_values))

# Create a DataFrame where each row is a different set of parameters
df = pd.DataFrame(param_combinations, columns=param_labels)

print(df)

MSE_params, PH_params, prediction_data = grid_search_SSV(EsnForecaster, param_grid, data, time_vals, ensembles, n_prediction, trainlen, synclen)

np.save(output_path+'/prediction_data.npy', prediction_data)

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
