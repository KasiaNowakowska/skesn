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

def plot_prediction(forecast, inverse_training_data, inverse_future_data, train_times, prediction_times, future_times, member, output_path):
    """
    plots the training, prediciton and truth for an individual ensemble member
    inputs:

    outputs:
    plot
    """
    inverse_prediction = forecast[:,:,member]
    fig, axes = plt.subplots(2, figsize=(12, 6), tight_layout=True, sharex=True)
    for j in range(len(forecast[:,0,0])):
        axes[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data')
        axes[j].plot(prediction_times, inverse_prediction[:, j],
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
    fig.savefig(output_path+'/timeseries_predictions%i.png' % member)
    plt.close()

def plot_predction_ensemble(forecast, inverse_training_data, inverse_future_data, train_times, prediction_times, future_times, output_path, fig, axes):
    for k in range(len(forecast(0,0,:])):
        inverse_prediction = forecast[:,:,k]
        for j in range(len(forecast[:,0,0])):
            axes[j].plot(train_times, inverse_training_data[:, j],
                      linewidth=2,
                      label='Training data')
            axes[j].plot(prediction_times, inverse_prediction[:, j],
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
    fig.savefig(output_path+'/timeseries_ensembles.png')

def plot_phase_space(forecast, inverse_training_data, inverse_future_data, train_times, prediction_times, future_times, member, output_path):
        inverse_prediction = forecast[:,:,member]
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
    figB.savefig(images_output_path+'/phasespace_predictions%i.png' % i)




    
