"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path> --n_train=<n_train> --n_sync=<n_sync> --n_prediction=<n_prediction> --synchronisation=<synchronisation> --n_reservoir=<n_reservoir> --spectral_radius=<spectral_radius> --sparsity=<sparsity> --lambda_r=<lambda_r> --beta=<beta> --use_noise=<use_noise> --use_bias=<use_bias> --use_b=<use_b> --ensembles=<ensembles>]

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
    --ensembles=<ensembles>            number of ensembles/how many times to rerun the ESN with the same IC and no retraining
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

def calculate_inst_nrmse(predictions, true_values):
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
    rmse = np.sqrt((predictions - true_values) ** 2)
  
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
ensembles = int(args['--ensembles'])

data_dir = '/validation_n_reservoir{0:}_spectral_radius{1:}_sparsity{2:}_lambda_r{3:}_beta{4:}_noise{5:}_bias{6:}_b{7:}'.format(args['--n_reservoir'], args['--spectral_radius'], args['--sparsity'], float(args['--lambda_r']), float(args['--beta']), args['--use_noise'], args['--use_bias'], args['--use_b'])

output_path = output_path1+data_dir
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print('made directory')

images_output_path = output_path+'/images'
if not os.path.exists(images_output_path):
    os.makedirs(images_output_path)
    print('made directory')

images_output_path2 = output_path+'/phasespace'
if not os.path.exists(images_output_path2):
    os.makedirs(images_output_path2)
    print('made directory')
images_output_path2b = output_path+'/phasespace/together'
if not os.path.exists(images_output_path2b):
    os.makedirs(images_output_path2b)
    print('made directory')

images_output_path3 = output_path+'/peakspath'
if not os.path.exists(images_output_path3):
    os.makedirs(images_output_path3)
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
synclen = n_sync
trainlen = n_train - synclen
ts = data[0:trainlen,:]
train_times = time_vals[0:trainlen]
dt = time_vals[1] - time_vals[0]
predictionlen = n_prediction
future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
sync_times = time_vals[trainlen:trainlen+synclen]
prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
sync_data = data[trainlen:trainlen+synclen]
future_data = data[trainlen:trainlen+synclen+predictionlen, :]

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

figA, axesA = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
figC, axesC = plt.subplots(1, figsize=(12, 6), tight_layout=True)

predicted_peaks = []
next_peak_time = []
next_peak_value = []

# Define number of bins and bin edges
num_bins = 60 # Adjust the number of bins as needed
bin_edges = np.linspace(prediction_times.min(), prediction_times.max(), num_bins + 1)

onset_true = np.zeros((len(prediction_times)))
offset_true = np.zeros((len(prediction_times)))
onset_pred_binned = np.zeros(num_bins)
offset_pred_binned = np.zeros(num_bins)

forecast_interval = 75 #150
no_of_forecasts = int(len(prediction_times)/forecast_interval) + 1
q_forecast = np.zeros((no_of_forecasts, ensembles))
ke_forecast = np.zeros((no_of_forecasts, ensembles))
forecast_times = np.zeros((no_of_forecasts))
q_true = np.zeros(no_of_forecasts)
ke_true = np.zeros(no_of_forecasts)
print(no_of_forecasts)

ensemble_all_vals_ke = np.zeros((len(prediction_times), ensembles))
ensemble_all_vals_q = np.zeros((len(prediction_times), ensembles))

PH_threshold_q = np.array((0.1, 0.2, 0.3, 0.4))
PH_threshold_ke = np.array((0.1, 0.2, 0.3, 0.4))
PH_vals_q = np.zeros((len(PH_threshold_q), ensembles))
PH_vals_ke = np.zeros((len(PH_threshold_ke), ensembles))


wait_times = []
# run through ensembles
for i in range(ensembles):
    #synchronise data
    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
    #predict
    future_predictionsync = modelsync.predict(predictionlen)

    #inverse scalar
    inverse_training_data = ss.inverse_transform(ts) 
    inverse_prediction = ss.inverse_transform(future_predictionsync)
    inverse_test_data = ss.inverse_transform(test_data)
    inverse_future_data = ss.inverse_transform(future_data) #this include the sync data

    #plot prediction for this ensemble member
    fig, axes = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
    for j in range(data[0:trainlen].shape[1]):
        axes[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data')
        axes[j].plot(prediction_times, inverse_prediction[:,j],
                      linewidth=2,
                      label='Prediction')
        axes[j].plot(future_times, inverse_future_data[:, j],
                      linewidth=2,
                      label='True')
        plt.legend()
    axes[0].set_ylabel('KE')
    axes[1].set_ylabel('q')
    axes[1].set_xlabel('time')
    axes[1].set_xlim(4000,15000)
    axes[0].grid()
    axes[1].grid()
    fig.savefig(images_output_path+'/timeseries_predictions%i.png' % i)
    plt.close()


    #add prediction to plot of all ensembles
    for j in range(data[0:trainlen].shape[1]):
        if i == 0:
            axesA[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data',
                      alpha=0.5, color='blue')
            axesA[j].plot(future_times, inverse_future_data[:, j],
                      linewidth=2,
                      label='True',
                      alpha=0.5, color='blue')
        if i % 10 == 0:
            axesA[j].plot(prediction_times, inverse_prediction[:,j],
                      linewidth=2,
                      label='Prediction')
    axesA[0].set_ylabel('KE')
    axesA[1].set_ylabel('q')
    axesA[1].set_xlabel('time')
    axesA[1].set_xlim(4500,12000)
    axesA[0].grid()
    axesA[1].grid()

    #plot phase space diagram for ensemble member
    figB, axB = plt.subplots(1, 2, figsize = (12,6), tight_layout=True)
    s_pred = axB[0].scatter(inverse_prediction[:,1], inverse_prediction[:,0], 
                            cmap='viridis', marker='.', alpha=0.8,
                            c=prediction_times, vmin=prediction_times[0], vmax=prediction_times[-1])
    s_true = axB[1].scatter(inverse_test_data[:, 1], inverse_test_data[:, 0],
                            cmap='viridis', marker='.', c=prediction_times,
                            vmin=prediction_times[0], vmax=prediction_times[-1])
    cbar_true = figB.colorbar(s_true, ax=axB[1], label='time')
    for m in range(2):
        axB[m].set_xlabel('q')
        axB[m].set_ylabel('KE')
        axB[m].set_ylim(-0.0001, 0.00035)
        axB[m].set_xlim(0.260,0.30)
    axB[0].set_title('prediction')
    axB[1].set_title('true')
    figB.savefig(images_output_path2+'/phasespace_predictions%i.png' % i)

    figB2, axB2 = plt.subplots(1, figsize = (12,12), tight_layout=True)
    s_pred = axB2.scatter(inverse_prediction[:,1], inverse_prediction[:,0], 
                            cmap='viridis', marker='x', alpha=0.8,
                            c=prediction_times, vmin=prediction_times[0], vmax=prediction_times[-1])
    s_true = axB2.scatter(inverse_test_data[:, 1], inverse_test_data[:, 0],
                            cmap='viridis', marker='.', c=prediction_times,
                            vmin=prediction_times[0], vmax=prediction_times[-1])
    axB2.set_xlabel('q')
    axB2.set_ylabel('KE')
    axB2.set_ylim(-0.0001, 0.00035)
    axB2.set_xlim(0.260,0.30)
    figB2.savefig(images_output_path2b+'/phasespace_predictions{:04d}.png'.format(i))
    
    #### peaks ####
    # set thresholds 
    threshold = 0.00010
    distance = 10
    prominence = 0.00005

    # find the peaks in the true test data (plot the peaks and record time of the next peak)
    if i == 0:
        peaks_true, _ = find_peaks(inverse_test_data[:, 0], height=threshold, distance=distance, prominence = prominence)
        num_true_peaks = len(peaks_true)
        axesC.scatter(prediction_times[peaks_true], np.ones_like(peaks_true) * 0, marker='x') #adds peaks in true test data to plot
        true_next_peak = prediction_times[peaks_true[0]] #records time of next peak
        true_next_peak_value = inverse_test_data[peaks_true[0], 0]

    # find the peaks in the prediction (plot the peaks of every 10th ensemble member and record time of next peak)
    peaks_pred, _ = find_peaks(inverse_prediction[:,0], height=threshold, distance=distance, prominence = prominence) 
    num_pred_peaks = len(peaks_pred)
    predicted_peak_times = prediction_times[peaks_pred]
    predicted_peaks.append(num_pred_peaks)
    if i % 10 == 0:
        axesC.scatter(predicted_peak_times, np.ones_like(peaks_pred) * (i+1))
    if len(predicted_peak_times) > 0:
        next_peak_time.append(predicted_peak_times[0])
        next_peak_value.append(inverse_prediction[peaks_pred[0],0])
    else:
        print('no peaks recorded')
        next_peak_time.append(np.inf) #no peak so set the next peak super far in advance so its not recorded

    axesC.set_xlim(prediction_times[0], prediction_times[-1])
    axesC.set_xlabel('time')
    axesC.set_ylabel('ensemble member')

    ### crossing dynamical system ###
    KE_threshold = 0.00015
    q_values_crossed_true = np.zeros((len(prediction_times)))
    time_values_crossed_true = np.zeros((len(prediction_times)))
    for t in range(len(prediction_times)):
        if (inverse_test_data[t,0] >= KE_threshold and inverse_test_data[t-1,0] < KE_threshold) or (inverse_test_data[t,0] <= KE_threshold and inverse_test_data[t-1,0] > KE_threshold):
            time_values_crossed_true[t] = prediction_times[t]
            q_values_crossed_true[t] = inverse_test_data[t,1]

    q_values_crossed_pred = np.zeros((len(prediction_times)))
    time_values_crossed_pred = np.zeros((len(prediction_times)))
    for t in range(len(prediction_times)):
        if (inverse_prediction[t,0] >= KE_threshold and inverse_prediction[t-1,0] < KE_threshold) or (inverse_prediction[t,0] <= KE_threshold and inverse_prediction[t-1,0] > KE_threshold):
            time_values_crossed_pred[t] = prediction_times[t]
            q_values_crossed_pred[t] = inverse_prediction[t,1]
    figD, axD = plt.subplots(1)
    axD.plot(prediction_times, q_values_crossed_true)
    axD.plot(prediction_times, q_values_crossed_pred, linestyle='--')
    axD.set_xlabel('time')
    axD.set_ylabel('q')
    figD.savefig(images_output_path3+'/crosssings%i.png' % i)

    ### onset and offset freq against time ###
    KE_threshold = 0.00015
    all_times = []
    for t in range(len(prediction_times)):
        bin_index = np.searchsorted(bin_edges, prediction_times[t]) - 1
        if i == 0:
            if (inverse_test_data[t,0] >= KE_threshold and inverse_test_data[t-1,0] < KE_threshold):
                onset_true[t] = 1
                all_times.append(prediction_times[t])
            elif (inverse_test_data[t,0] <= KE_threshold and inverse_test_data[t-1,0] > KE_threshold):
                offset_true[t] = 1

        if (inverse_prediction[t,0] >= KE_threshold and inverse_prediction[t-1,0] < KE_threshold):
            onset_pred_binned[bin_index] += 1
        elif (inverse_prediction[t,0] <= KE_threshold and inverse_prediction[t-1,0] > KE_threshold):
            offset_pred_binned[bin_index] += 1
 
    #determine MSE, NRMSE and lag
    NRMSE_ke = calculate_inst_nrmse(inverse_prediction[:,0] , inverse_test_data[:,0])
    NRMSE_q = calculate_inst_nrmse(inverse_prediction[:,1] , inverse_test_data[:,1])
    for thresh in range(len(PH_threshold_ke)):
        ind_ke = np.argmax(NRMSE_ke > PH_threshold_ke[thresh])
        ind_q = np.argmax(NRMSE_q > PH_threshold_q[thresh])
        PH_vals_ke[thresh, i] = prediction_times[ind_ke]
        PH_vals_q[thresh, i] = prediction_times[ind_q]
    
    datanames = ['KE', 'q']
    for k in range(2):
        print('\n', datanames[k])
        # reshape the data
        prediction_reshape = inverse_prediction[:,k].reshape(len(future_predictionsync), 1)
        test_data_reshape = inverse_test_data[:,k].reshape(len(future_predictionsync), 1)
        # calculate NRMSE
        if datanames[k] == 'KE':
            NRMSE_ke = calculate_nrmse(prediction_reshape , test_data_reshape)
        elif datanames[k] == 'q': 
            NRMSE_q = calculate_nrmse(prediction_reshape , test_data_reshape)
        # calculate MSE 
        if datanames[k] == 'KE':
            MSE_values_ke.append(calculate_mse(prediction_reshape, test_data_reshape))
        elif datanames[k] == 'q':
            MSE_values_q.append(calculate_mse(prediction_reshape, test_data_reshape))

    for l in range(2):
        print('\n', datanames[l])
        prediction_reshape = inverse_prediction[:,l]
        test_data_reshape = inverse_test_data[:,l]
        print('\n delay for 2000 timesteps=', timedelay(test_data_reshape[:], prediction_reshape[:]))
        if l == 0:
            lag_values_ke.append(timedelay(test_data_reshape[:], prediction_reshape[:]))
        if l == 1:
            lag_values_q.append(timedelay(test_data_reshape[:], prediction_reshape[:]))
            #print(prediction_reshape, test_data_reshape)

    ### metegorams ####
    forecast_time = 0
    for t in range(len(prediction_times)):
      if t % forecast_interval == 0:
        print(t)
        print(forecast_time)
        q_forecast[forecast_time, i] = inverse_prediction[t,1]
        ke_forecast[forecast_time, i] = inverse_prediction[t,0]
        forecast_times[forecast_time] = int(prediction_times[t])
        if i == 0:
          q_true[forecast_time] = inverse_test_data[t,1] 
          ke_true[forecast_time] = inverse_test_data[t,0] 
        forecast_time += 1

    ### ensemble mean ###
    ensemble_all_vals_ke[:,i] = inverse_prediction[:,0]
    ensemble_all_vals_q[:,i] = inverse_prediction[:,1]
    
    for t in range(1, len(all_times)):
        rangevals = all_times[t] - all_times[t-1]
        wait_times.append(rangevals)

np.save(output_path+'/ensemble_all_vals_ke.npy', ensemble_all_vals_ke)
np.save(output_path+'/ensemble_all_vals_q.npy', ensemble_all_vals_q)
np.save(output_path+'/test_data_ke.npy', inverse_test_data[:,0])
np.save(output_path+'/test_data_q.npy', inverse_test_data[:,1])


#find avg MSE across ensemble
avg_MSE_ke = np.mean(MSE_values_ke)
avg_MSE_q = np.mean(MSE_values_q)
avg_lag_ke = np.mean(lag_values_ke)
avg_lag_q = np.mean(lag_values_q)

#prediction horizon
figJ, axJ = plt.subplots(2, figsize=(12, 6), sharex=True, tight_layout=True)
print('lenght=', len(PH_vals_ke))
for thresh in range(len(PH_threshold_ke)):
    cumsum_ke = np.cumsum(PH_vals_ke[thresh, :])
    cumsum_q = np.cumsum(PH_vals_q[thresh, :])
    cdf_vals_PH_ke = cumsum_ke / np.max(cumsum_ke)
    cdf_vals_PH_q = cumsum_q / np.max(cumsum_q)
    axJ[0].plot(np.sort(PH_vals_ke[thresh, :]), cdf_vals_PH_ke, marker='o')
    axJ[1].plot(np.sort(PH_vals_q[thresh, :]), cdf_vals_PH_q, marker='o', label='threshold=%.3f' % PH_threshold_q[thresh])
    axJ[1].set_xlabel('Prediction Horizon')
    axJ[0].set_ylabel('Cumulative Probability KE')
    axJ[1].set_ylabel('Cumulative Probability q')
plt.legend()
figJ.savefig(output_path+'/cdf_rangethresholds.png')

#calculate fraction of ensembles which find all the peaks and all the peaks apart from 1
counter_all = 0
counter_missing_one = 0
for value in predicted_peaks:
    if value == num_true_peaks:
        counter_all +=1
    elif value == num_true_peaks-1:
        counter_missing_one += 1
fraction_peak_matches = counter_all/ensembles
fraction_peaks_missing_one = counter_missing_one/ensembles

avg_next_peak_value = np.mean(next_peak_value)
avg_amplitude_error = np.abs(true_next_peak_value - avg_next_peak_value)

#calculate the fraction of ensembles which predict the next peak within 100 timesteps of the true peak
#calculate the fraction of ensembles which predict the next peak within 50 timesteps and before the true peak
counter_next_time_within100 = 0
counter_next_time_within50andbefore = 0
for value in next_peak_time:
    if abs(value - true_next_peak) <= 100:
        counter_next_time_within100 += 1
    if 0 < (true_next_peak - value) <= 50:
        counter_next_time_within50andbefore += 1
fraction_within100 = counter_next_time_within100/ensembles
fraction_within50andbefore = counter_next_time_within50andbefore/ensembles

# plot for predicted onset/offset vs true
figE, axE = plt.subplots(1, figsize=(12, 6), tight_layout=True)
axE.plot(bin_edges[:-1], onset_pred_binned, color='green', linestyle='--', label='predicted onset')
axE.plot(bin_edges[:-1], offset_pred_binned, color='red', linestyle='--', label='predicted cessation')
axE.plot(prediction_times, onset_true, color='darkgreen', label='true onset')
axE.plot(prediction_times, offset_true, color='darkred', label='true cessation')
plt.legend()
plt.grid()
axE.set_xlabel('time')
axE.set_ylabel('Frequency')
figE.savefig(output_path+'/onsetoffset.png')

### meteogram plots ###
figF, axF = plt.subplots(2, figsize=(12, 6), sharex=True, tight_layout=True)
axF[0].boxplot(ke_forecast.T, positions=forecast_times, labels=forecast_times, widths=10)
axF[1].boxplot(q_forecast.T, positions=forecast_times, labels=forecast_times, widths=10)
axF[0].plot(prediction_times, inverse_test_data[:,0])
axF[1].plot(prediction_times, inverse_test_data[:,1])
axF[0].set_ylabel('KE')
axF[1].set_ylabel('q')
axF[1].set_xlabel('time')
axF[0].set_xlim(forecast_times[0]-10, forecast_times[-1]+10)
axF[1].set_xlim(forecast_times[0]-10, forecast_times[-1]+10)
figF.savefig(output_path+'/meteogram_lines.png')

### ensemble means and variance ###
figG, axG = plt.subplots(1, figsize=(6, 6), tight_layout=True)
ensemble_mean_ke = np.mean(ensemble_all_vals_ke, axis=1)
ensemble_mean_q = np.mean(ensemble_all_vals_q, axis=1)
ensemble_var_ke = np.var(ensemble_all_vals_ke, axis=1)
ensemble_var_q = np.var(ensemble_all_vals_q, axis=1)
axG.plot(ensemble_mean_q, ensemble_mean_ke)
axG.set_xlabel('q')
axG.set_ylabel('KE')
axG.set_xlim(0.265, 0.300)
axG.set_ylim(0, 3e-4)
figG.savefig(output_path+'/ensemble_mean_phase_diagram.png')

figH, axH = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
for j in range(data[0:trainlen].shape[1]):
  axH[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data',
                      alpha=0.5, color='blue')
  axH[j].plot(future_times, inverse_future_data[:, j],
                      linewidth=2,
                      label='True',
                      alpha=0.5, color='blue')
  if j == 0:
    axH[j].plot(prediction_times, ensemble_mean_ke,
                      linewidth=2,
                      label='Prediction')
  elif j == 1:
    axH[j].plot(prediction_times, ensemble_mean_q,
                      linewidth=2,
                      label='Prediction')
axH[0].set_ylabel('KE')
axH[1].set_ylabel('q')
axH[1].set_xlabel('time')
axH[1].set_xlim(4000,12000)
axH[0].grid()
axH[1].grid()
figH.savefig(output_path+'/ensemble_mean_time_series.png')

figI, axI = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
axI[0].plot(prediction_times, ensemble_var_ke)
axI[1].plot(prediction_times, ensemble_var_q)
axI[0].set_ylabel('var KE')
axI[1].set_ylabel('var q')
axI[1].set_xlabel('time')
axI[1].set_xlim(11000,12000)
figI.savefig(output_path+'/ensemble_var.png')

# median and confidence interval
median_ke = np.median(ensemble_all_vals_ke, axis=1) 
median_q = np.median(ensemble_all_vals_q, axis=1) 
lower_bound_ke = np.percentile(ensemble_all_vals_ke, 10, axis=1)
upper_bound_ke = np.percentile(ensemble_all_vals_ke, 90, axis=1)
lower_bound_q = np.percentile(ensemble_all_vals_q, 10, axis=1)
upper_bound_q = np.percentile(ensemble_all_vals_q, 90, axis=1)
figK, axK = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
for j in range(data[0:trainlen].shape[1]):
  axK[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data',
                      alpha=0.5, color='blue')
  axK[j].plot(future_times, inverse_future_data[:, j],
                      linewidth=2,
                      label='True',
                      alpha=0.5, color='blue')
  if j == 0:
    axK[j].plot(prediction_times, median_ke,
                      linewidth=2,
                      label='Median Prediction', color='green')
    axK[j].fill_between(prediction_times, lower_bound_ke, upper_bound_ke, color='green', alpha=0.3, label='80% Confidence Interval')
  elif j == 1:
    axK[j].plot(prediction_times, median_q,
                      linewidth=2,
                      label='Median Prediction')
    axK[j].fill_between(prediction_times, lower_bound_q, upper_bound_q, color='green', alpha=0.3, label='80% Confidence Interval')
plt.legend()
figK.savefig(output_path+'/median_conf_int.png')

# qtrue vs qpred 
figL, axL = plt.subplots(1,2 , figsize=(12, 6), tight_layout=True)
for r in range(ensembles):
    axL[0].scatter(inverse_test_data[:100,0], ensemble_all_vals_ke[:100,r], marker='.')
    axL[1].scatter(inverse_test_data[:100,1], ensemble_all_vals_q[:100,r], marker='.')
axL[0].set_xlabel('KE true')
axL[0].set_ylabel('KE ESN')
axL[1].set_xlabel('q true')
axL[1].set_ylabel('q ESN')
figL.savefig(output_path+'/scattertrue_v_pred.png')

figM, axM = plt.subplots(1,2 , figsize=(12, 6), tight_layout=True)
axM[0].scatter(inverse_test_data[:100,0], median_ke[:100], marker='.', color='green')
axM[1].scatter(inverse_test_data[:100,1], median_q[:100], marker='.', color='green')

figM.savefig(output_path+'/scatter_true_v_median.png')

# pdf of q and KE 
figN, axN = plt.subplots(2,2 , figsize=(12, 6), tight_layout=True, sharex='col')
flatten_ensembles_ke = ensemble_all_vals_ke.flatten()
flatten_ensembles_q = ensemble_all_vals_q.flatten()

max_pred_ke, min_pred_ke = np.max(flatten_ensembles_ke), np.min(flatten_ensembles_ke)
max_pred_q, min_pred_q = np.max(flatten_ensembles_q), np.min(flatten_ensembles_q)
max_true_ke, min_true_ke = np.max(inverse_test_data[:,0]), np.min(inverse_test_data[:,0])
max_true_q, min_true_q = np.max(inverse_test_data[:,1]), np.min(inverse_test_data[:,1])
min_ke = np.min((min_pred_ke, min_true_ke))
max_ke = np.max((max_pred_ke, max_true_ke))
min_q = np.min((min_pred_q, min_true_q))
max_q = np.max((max_pred_q, max_true_q))

bins_ke = np.linspace(-1e-4, 3e-4, 20)
bins_q = np.linspace(0.265, 0.300, 20)


axN[1,0].hist(flatten_ensembles_ke, bins=bins_ke, density=True)
axN[1,1].hist(flatten_ensembles_q, bins=bins_q, density=True)
axN[0,0].hist(inverse_test_data[:,0], bins=bins_ke, density=True)
axN[0,1].hist(inverse_test_data[:,1], bins=bins_q, density=True)
axN[0, 0].set_xlim(min_ke, max_ke)
axN[1, 0].set_xlim(min_ke, max_ke)
axN[0, 1].set_xlim(min_q, max_q)
axN[1, 1].set_xlim(min_q, max_q)
axN[1, 0].set_xlabel('KE')
axN[1, 1].set_xlabel('q')
for i in range(2):
  for j in range(2):
    axN[i, j].set_ylabel('density')
figN.savefig(output_path+'/pdfs.png')

### wait times ###
figP, axP = plt.subplots(1, figsize=(12, 6), tight_layout=True)
axP.hist(wait_times, bins=20)
axP.set_xlabel('wait time')
axP.set_ylabel('frequency')
figP.savefig(output_path+'/histogram_wait_times_onset.png')


#save simulation details and results
simulation_details = {
            'ensembles': ensembles,
            'n_train': n_train,
            'n_sync': n_sync,
            'n_prediction': n_prediction,
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
            'avg_lag_q': avg_lag_q,
            'fraction_peak_matches': fraction_peak_matches,
            'fraction_peaks_missing_one': fraction_peaks_missing_one,
            'avg_amplitude_error_for_peak': avg_amplitude_error,
            'fraction_within100': fraction_within100,
            'fraction_within50andbefore': fraction_within50andbefore
}


figA.savefig(output_path+'/timeseries_ensembles.png')
figC.savefig(output_path+'/peaks_ensembles.png')

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
