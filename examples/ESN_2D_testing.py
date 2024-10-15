"""
python script for ESN grid search with bayesian opt.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_trainLT=<n_train> --n_sync=<n_sync> --n_predictionLT=<n_prediction> --n_washoutLT=<n_washoutLT> --LT=<LT> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --input_scaling=<input_scaling> --data_type=<data_type> --ensembles=<ensembles> --modes=<modes> --index_start=<index_start>]

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
    --data_type=<data_type>            data type [default: minus_mean]
    --ensembles=<ensembles>            number of ensembles/how many times to rerun the ESN with the same IC and no retraining                       
    --modes=<modes>                    number of modes in PCA [default: 150]  
    --index_start=<index_start>        starting index for timeseries
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

import validation_stratergies as vs
import evaluation_metrics as evalm

from sklearn.decomposition import PCA, IncrementalPCA

import wandb
wandb.login()
    
input_path = args['--input_path']
output_path1 = args['--output_path']
data_type = args['--data_type']
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
modes = int(args['--modes'])

data_dir = '/POD_testing_ntrain{0:}_nprediction{1:}_nwashout{2:}_modes{3:}_n_reservoir{4:}_spectral_radius{5:}_sparsity{6:}_lambda_r{7:}_input_scaling{8:}_data_type{9:}_start_index{10:}'.format(args['--n_trainLT'], args['--n_predictionLT'], args['--n_washoutLT'], args['--modes'], args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), float(args['--input_scaling']), args['--data_type'], args['--index_start'])

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

'''
##### load in data ######
q = np.load(input_path+'/q_vertical_allz.npy')
w = np.load(input_path+'/w_vertical_allz.npy')
total_time = np.load(input_path+'/total_time.npy')
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
q = q[50:-50]
w = w[50:-50]
total_time = total_time[50:-50]
print(total_time[0], total_time[-1])
print(len(q), len(w), len(total_time))

variables = num_variables = 2
variable_names = ['q', 'w']
global_variable_names = ['KE', 'q']

data = np.zeros((len(total_time), len(x), 1, len(z), variables))
if data_type == 'full_data':
    data[:,:,:,:,0] = q 
    data[:,:,:,:,1] = w 
elif data_type == 'minus_mean':
    data[:,:,:,:,0] = q - np.mean(q, axis=0)
    data[:,:,:,:,1] = w - np.mean(w, axis=0)

data = np.squeeze(data, axis=2)

num_snapshots = 5000
data_across_x = data[:num_snapshots, :, :, :]
data_matrix = data_across_x.reshape(num_snapshots, -1)
time_vals = total_time[:num_snapshots]

''' 

#### larger data 5000-30000 hf ####
num_snapshots = 12500
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')
variables = num_variables = 2
variable_names = ['q', 'w']

with h5py.File(input_path+'/data_5000_30000.h5', 'r') as df:
    time_vals = np.array(df['total_time_all'])
    q = np.array(df['q_all'])
    w = np.array(df['w_all'])

    q = np.squeeze(q, axis=2)
    w = np.squeeze(w, axis=2)

    q_mean = np.mean(q, axis=0)
    w_mean = np.mean(w, axis=0)

    q_array = q - q_mean
    w_array = w - w_mean
    print(np.shape(q_array))

del q
del w

q_array = q_array.reshape(len(time_vals), len(x), len(z), 1)
w_array = w_array.reshape(len(time_vals), len(x), len(z), 1)

data_full = np.concatenate((q_array, w_array), axis=-1)
print('shape of data_full:', np.shape(data_full))

del q_array
del w_array

print('shape of time_vals', np.shape(time_vals))
global_variable_names = ['KE', 'q']

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

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print('cumulative explained variance', cumulative_explained_variance[-1])



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

print('shape of data for ESN', np.shape(data_reduced))

#### varaince, energy and errors in reconstruction ####
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print('cumulative explained variance', cumulative_explained_variance[-1])

avg_nrmse = np.mean(nrmse)
sq_avg_nrmse = avg_nrmse**2
Energy = 1-sq_avg_nrmse
curve = 1-Energy
print('Energy Error (curve value):', curve)
'''

#### ESN ####
print('starting ESN ...')
dt = int(time_vals[1]-time_vals[0])
print('dt=', dt)

n_lyap = int(LT/dt)
print('N_lyap=', n_lyap)
synclen = n_sync
n_prediction_inLT = n_predictionLT
n_train_inLT = n_trainLT + n_washoutLT #current set up is the washout happens at start so need to add this washout into train time
n_washout = n_washoutLT*n_lyap


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

N_train = n_train_inLT*n_lyap
N_val = N_test = n_prediction_inLT*n_lyap
test_index_start = index_start - n_train_inLT*n_lyap


U_train = data_reduced[test_index_start:test_index_start+N_train-synclen,:]
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


modes = len(data_reduced[1])

ensemble_all_modes = np.zeros((N_test, modes, ensembles))
ensemble_global = np.zeros((N_test, variables, ensembles))
ensemble_PH_error = np.zeros((N_test, ensembles))
ensemble_PH = np.zeros((ensembles))
ensemble_global_PH = np.zeros((ensembles))
ensmeble_FFT = np.zeros((N_test, modes, ensembles))

test_index = index_start
y_test = data_reduced[test_index:test_index+N_test, :]
print('y_test=', np.shape(y_test))
U_test_sync = data_reduced[test_index-synclen:test_index,:]
times_test = times_val = time_vals[test_index:test_index+N_test]

#### ground truth reconstructions #####
reconstructed_groundtruth = data_reconstructed[test_index:test_index+N_test, :, : ,:]
reconstructed_groundtruth_avgq = np.mean(reconstructed_groundtruth[:,:,:,0], axis=(1,2))
reconstructed_groundtruth_ke = 0.5*reconstructed_groundtruth[:,:,:,1]*reconstructed_groundtruth[:,:,:,1]
reconstructed_groundtruth_avgke = np.mean(reconstructed_groundtruth_ke, axis=(1,2))
reconstructed_global_truth = np.zeros((N_test, variables))
reconstructed_global_truth[:,0] = reconstructed_groundtruth_avgke
reconstructed_global_truth[:,1] = reconstructed_groundtruth_avgq

inspect_graphs = True

# run through ensembles
for i in range(ensembles):
    print('ensemble number =', i)
    #synchronise data
    modelsync._update(U_test_sync, UpdateModes = UpdateModes.synchronization)
    #predict
    future_predictionsync = modelsync.predict(N_test)

    for m in range(modes):
        ensemble_all_modes[:, m, i] = future_predictionsync[:, m]

    #### Visualise mode errors ####
    indexes_to_plot = np.array([1, 2, 10, 50, 100] ) -1
    if inspect_graphs == True:
        fig, ax = plt.subplots(5, figsize=(12,9),sharex=True, tight_layout=True)
        for l in range(5):
            mode = indexes_to_plot[l]
            ax[l].plot(times_train, U_train[:,mode], color='tab:blue', alpha=0.2, label='training')
            #ax[l].fill_between(U_train[:,mode], times_val[0], times_val[1])
            ax[l].plot(times_val, y_test[:,mode], color='tab:blue', label='mode truth')
            ax[l].plot(times_val, future_predictionsync[:,mode], color='tab:orange', label='mode prediction')
            ax[l].set_title('mode %i' % (mode+1))
            for l in range(5):
                ax[l].legend()
                ax[l].grid()
            ax[-1].set_xlabel('time')
            ax[-1].set_xlim(times_val[0]-400, times_val[-1])
        fig.suptitle('ensemble member=%03d' % i)
        fig.savefig(output_path_images+'/modes_%03d.png' % i)
        plt.close()

    #### PH ####
    PH_E = vs.PH_error(future_predictionsync, y_test)
    ensemble_PH_error[:, i] = PH_E
    PH = vs.prediction_horizon(future_predictionsync, y_test, threshold=0.2)
    ensemble_PH[i] = PH
    if inspect_graphs == True:
        fig, ax =plt.subplots(1, figsize=(12,3), tight_layout=True)
        ax.plot(times_test, PH_E, color='tab:blue')
        ax.set_xlabel('time')
        ax.set_ylabel('E')
        ax.grid()
        fig.suptitle('ensemble member=%03d' % i)
        fig.savefig(output_path_images+'/PH_error_%03d.png' % i)
        plt.close()

    #### Reconstruct 2D space ####
    reconstructed_predictions = pca.inverse_transform(future_predictionsync)
    reconstructed_predictions = reconstructed_predictions.reshape(N_val, len(x), len(z), variables) 

    # plot the result
    if inspect_graphs == True:
        print('plotting reconstructions')
        fig, ax = plt.subplots(3,variables, figsize=(15,9), sharex=True, tight_layout=True)

        z_index=32 
        error = reconstructed_groundtruth[:,:,:,:]-reconstructed_predictions[:,:,:,:]
        for v in range(variables):
            cs1 = ax[0,v].pcolormesh(times_val, x, reconstructed_groundtruth[:,:,z_index,v].T)
            cs2 = ax[1,v].pcolormesh(times_val, x, reconstructed_predictions[:,:,z_index,v].T)
            cs3 = ax[2,v].pcolormesh(times_val, x, error[:,:,z_index,v].T, cmap='RdBu', norm=TwoSlopeNorm(vmin=-np.max(np.abs(error)), vcenter=0, vmax=np.max(np.abs(error))))

            # Find global min and max for color scale
            vmin_data = min(cs1.get_clim()[0], cs2.get_clim()[0])
            vmax_data = max(cs1.get_clim()[1], cs2.get_clim()[1])

            # Set the color scale for both plots
            cs1.set_clim(vmin_data, vmax_data)
            cs2.set_clim(vmin_data, vmax_data)

            fig.colorbar(cs1, ax=ax[0,v])
            fig.colorbar(cs2, ax=ax[1,v])
            fig.colorbar(cs3, ax=ax[2,v])

            ax[2,v].set_xlabel('time')
            for k1 in range(variables):
                ax[1,k1].set_ylabel('x')
                #ax[1].set_xlim(150,160)

            ax[0,v].set_title(variable_names[v])
        print('ensemble member=%03d' % i)
        fig.suptitle('ensemble member=%03d' % i)
        fig.savefig(output_path_images+'/reconstruction_em_%03d.png' % i)
        plt.close()
    

    #### take global averages of predictions ####
    reconstructed_prediction_avgq = np.mean(reconstructed_predictions[:,:,:,0], axis=(1,2))
    reconstructed_prediction_ke = 0.5*reconstructed_predictions[:,:,:,1]*reconstructed_predictions[:,:,:,1]
    reconstructed_prediction_avgke = np.mean(reconstructed_prediction_ke, axis=(1,2))
    print('shape of global ke', np.shape(reconstructed_prediction_avgke))

    ensemble_global[:,0,i] = reconstructed_prediction_avgke
    ensemble_global[:,1,i] = reconstructed_prediction_avgq
    
    PH_g = vs.prediction_horizon(ensemble_global[:,:,i] , reconstructed_global_truth, threshold=0.2)
    ensemble_global_PH[i] = PH_g

    if inspect_graphs == True:
        fig, ax= plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
        for v in range(2):
            ax[v].plot(times_val, reconstructed_global_truth[:,v], color='tab:orange', label='truth')
            ax[v].plot(times_val, ensemble_global[:,v,i], color='tab:blue', label='prediction')
            ax[v].legend()
            ax[v].grid()
        ax[1].set_xlabel('time')
        ax[0].set_ylabel('KE')
        ax[1].set_ylabel('q')
        fig.suptitle('ensemble member=%03d' % i)
        fig.savefig(output_path_images+'/global_pred_%03d.png' % i)
        plt.close()

    #### FFT ####
    if inspect_graphs == True:
        fig, ax = plt.subplots(5, figsize=(12,9), sharex=True)
        for l in range(5):
            mode = indexes_to_plot[l]
            om_p, psd_p = evalm.FFT(future_predictionsync[:,mode],times_val)
            om_t, psd_t = evalm.FFT(y_test[:,mode], times_val)
            ax[l].plot(om_p[len(om_p)//2:], psd_p[len(om_p)//2:], label='ESN')
            ax[l].plot(om_t[len(om_p)//2:], psd_t[len(om_p)//2:], label='POD')
            ax[l].set_ylabel('Mode %i' % (mode+1))
            ax[l].legend()
            ax[l].grid()
        ax[-1].set_xlabel('Frequency f')
        fig.suptitle('ensemble member=%03d' % i)
        fig.savefig(output_path_images+'/FFT_%03d.png' % i)
        plt.close()

np.save(output_path+'/reconstructed_global_truth.npy', reconstructed_global_truth)
np.save(output_path+'/ensemble_global.npy', ensemble_global)

PH_global_mean = np.mean(ensemble_global_PH)
PH_mean = np.mean(ensemble_PH)
print('Mean Prediciton Horizon 2D data:', PH_mean)
print('Mean Prediciton Horizon global data:', PH_global_mean)

PH_error_local = np.mean(ensemble_PH_error, axis=1)

print(ensemble_global[:10, 0 ,0])

if inspect_graphs == True:
    fig, ax= plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
    for v in range(variables):
        means = np.mean(ensemble_global[:,v,:], axis=1)
        lower_bounds = np.percentile(ensemble_global[:,v,:], 5, axis=1)
        upper_bounds = np.percentile(ensemble_global[:,v,:], 95, axis=1)
        ax[v].plot(times_val, means, label='mean prediction', color='tab:orange')
        ax[v].plot(times_val, reconstructed_global_truth[:,v], color='tab:blue', label='true')
        ax[v].fill_between(times_val, lower_bounds, upper_bounds, color='tab:orange', alpha=0.3, label='90% confidence interval')
        ax[v].grid()
        ax[v].set_ylabel(global_variable_names[v])
    ax[1].legend()
    fig.savefig(output_path_images+'/timeseries_ens.png')
    plt.close()

    fig, ax =plt.subplots(1, figsize=(12,3), tight_layout=True)
    ax.plot(times_test, PH_error_local, color='tab:blue')
    ax.set_xlabel('time')
    ax.set_ylabel('E')
    ax.grid()
    fig.suptitle('mean error')
    fig.savefig(output_path_images+'/PH_mean_error_%03d.png' % i)
    plt.close()

