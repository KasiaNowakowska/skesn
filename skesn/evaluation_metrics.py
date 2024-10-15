'''
To use these you must have your data as ensemble data variable by varaible and also true data
'''
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
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from cross_validation import ValidationBasedOnRollingForecastingOrigin
from esn_old_adaptations import EsnForecaster

from scipy.signal import find_peaks

import h5py

from docopt import docopt

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

fig, ax = plt.subplots(1)

def boxplot_2d(x,y, ax, whis=1.5, choose_color='black'):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = choose_color,
        fc = 'none',
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color=choose_color,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color=choose_color,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color=choose_color, marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom], 
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]], 
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top], 
        color = choose_color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors=choose_color,
    )
    
def timeseries(prediction, test_data, prediction_times, variables, variable_names, ax=ax):
    '''
    plots timeseries prediction for one single ensemble member
    inputs:
    prediction - prediction (array: (time, variables))
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    '''
    if ax.shape != (variables,):
        print('ax shape needs to be shape', variables)
        return
    for v in range(variables):
        ax[v].plot(prediction_times, prediction[:,v],
                                linewidth=2,
                                label='Prediction', color='orange')
        ax[v].plot(prediction_times, test_data[:,v], linewidth=2,
                                label='True', color='tab:blue', linestyle='--')
        ax[v].set_ylabel(variable_names[v])
        ax[v].grid()
    ax[variables-1].set_xlabel('time')
    print('added prediction')
    
def timeseries_plustraining(prediction, test_data, train_data, train_times, prediction_times, variables, variable_names, ax=ax):
    '''
    plots timeseries prediction with training for one single ensemble member
    inputs:
    prediction - prediction (array: (time, variables))
    test_data - true data (array: (time, variables))
    train_data - data used for training (array: (train_times, variables))
    train_times - array of times used for training (array: (train_times))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    '''
    if ax.shape != (variables,):
        print('ax shape needs to be shape', variables)
        return
    for v in range(variables):
        ax[v].plot(train_times, train_data[:,v], linewidth=2,
                                    color='tab:blue')
        ax[v].fill_between(train_times, train_data[:,v].min(), train_data[:,v].max(), alpha=0.2, color='tab:blue', label='Training')
        ax[v].plot(prediction_times, prediction[:,v],
                                linewidth=2,
                                label='Prediction', color='orange')
        ax[v].plot(prediction_times, test_data[:,v], linewidth=2,
                                label='True', color='tab:blue', linestyle='--')
        ax[v].set_ylabel(variable_names[v])
        ax[v].grid()
    ax[variables-1].set_xlabel('time')
    print('added prediction')
    
def phase_space_prediction(prediction, test_data, prediction_times, variables, variable_names, ax=ax, fig=fig):
    '''
    plots the phase space prediction for prediction and true for one single ensemble member
    inputs:
    prediction - prediction (array: (time, variables))
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    '''
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 6), tight_layout=True)
    plt.grid()
    s_pred = ax.scatter(prediction[:,1], prediction[:,0],
                            cmap='viridis', marker='x',
                            c=prediction_times, vmin=prediction_times[0], vmax=prediction_times[-1], label='Prediction')
    s_true = ax.scatter(test_data[:,1], test_data[:,0],
                            cmap='viridis', marker='.', c=prediction_times,
                            vmin=prediction_times[0], vmax=prediction_times[-1], label='True')
    ax.set_xlabel('q')
    ax.set_ylabel('KE')
    ax.set_ylim(-0.0001, 0.00035)
    ax.set_xlim(0.260,0.30)
    cbar_true = fig.colorbar(s_true, ax=ax, label='time')
    plt.legend()

def ensemble_timeseries_plustraining_mean(ensemble_all_vals, test_data, train_data, train_times, prediction_times, variables, variable_names, ax=ax):
    '''
    plots timeseries prediction with training for ensemble
    inputs:
    ensemble_all_vals - prediction for all ensembles (array: (time, variables, ensembles))
    test_data - true data (array: (time, variables))
    train_data - data used for training (array: (train_times, variables))
    train_times - array of times used for training (array: (train_times))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    '''
    #means
    means = np.zeros((len(prediction_times), variables))
    lower_bounds = np.zeros((len(prediction_times),variables))
    upper_bounds =  np.zeros((len(prediction_times),variables))
    for v in range(variables):
        means[:,v] = np.mean(ensemble_all_vals[:,v,:], axis=1)
        lower_bounds[:,v] = np.percentile(ensemble_all_vals[:,v,:], 5, axis=1)
        upper_bounds[:,v] = np.percentile(ensemble_all_vals[:,v,:], 95, axis=1)
        ax[v].plot(train_times, train_data[:,v], linewidth=2,
                                    color='tab:blue')
        ax[v].plot(prediction_times, means[:,v],
                                    linewidth=2,
                                    label='Mean Prediction', color='orange')
        ax[v].fill_between(prediction_times, lower_bounds[:,v], upper_bounds[:,v], color='orange', alpha=0.3, label='90% Confidence Interval')

        ax[v].plot(prediction_times, test_data[:,v], linewidth=2,
                                    label='True', color='tab:blue', linestyle ='--')
        ax[v].set_ylabel(varaible_names[v])
        ax[v].grid()
        #ax[v].set_xlim(prediction_times[0]-100, prediction_times[-1])
    ax[variables-1].set_xlabel('time')
    print('added prediction')


def ensemble_timeseries(ensemble_all_vals, test_data, prediction_times, variables, variable_names, ax=ax):
    '''
    plots timeseries prediction for ensemble
    inputs:
    ensemble_all_vals - prediction for all ensembles (array: (time, variables, ensembles))
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    '''
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    means = np.zeros((len(prediction_times), variables))
    lower_bounds = np.zeros((len(prediction_times),variables))
    upper_bounds =  np.zeros((len(prediction_times),variables))
    for v in range(variables):
        means[:,v] = np.mean(ensemble_all_vals[:,v,:], axis=1)
        lower_bounds[:,v] = np.percentile(ensemble_all_vals[:,v,:], 5, axis=1)
        upper_bounds[:,v] = np.percentile(ensemble_all_vals[:,v,:], 95, axis=1)
        ax[v].plot(prediction_times, means[:,v],
                                    linewidth=2,
                                    label='Mean Prediction', color='orange')
        ax[v].plot(prediction_times, test_data[:,v], linewidth=2,
                                    label='True', color='tab:blue')
        ax[v].fill_between(prediction_times, lower_bounds[:,v], upper_bounds[:,v], color='orange', alpha=0.3, label='90% Confidence Interval')
        ax[v].set_ylabel(variable_names[v])
        ax[v].grid()
        ax[v].set_xlim(prediction_times[0]-100, prediction_times[-1])
    ax[variables-1].set_xlabel('time')
    print('added prediction')

def meteogram(ensemble_all_vals, test_data, prediction_times, variables, variable_names, ax=ax, forecast_interval=200):
    '''
    plots metogram at forecast intervals
    inputs:
    ensemble_all_vals - prediction for all ensembles (array: (time, variables, ensembles))
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    variables - number of varaibles in data (integer)
    variable names - array of the labels of the variables (array: (variable names))
    ax - axis for figure of length variables
    forecast interval - length of time between forecasts 
    '''
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    ensembles = ensemble_all_vals.shape[-1]
    no_of_forecasts = int(len(prediction_times)/forecast_interval)
    forecast = np.zeros((no_of_forecasts, variables, ensembles))
    forecast_times = np.zeros((no_of_forecasts))
    print(no_of_forecasts)

    forecast_time = 0
    ensembles = len(ensemble_all_vals[0,0,:])
    for t in range(len(prediction_times)):
      if t % forecast_interval == 0:
        for v in range(variables):
            forecast[forecast_time, v, :] = ensemble_all_vals[t, v, :]
        forecast_times[forecast_time] = int(prediction_times[t])
        forecast_time += 1

    print(forecast_times)
    for v in range(variables):
        ax[v].boxplot(forecast[:,v,:].T, positions=forecast_times, labels=forecast_times, widths=20, showmeans=True)
        ax[v].plot(prediction_times, test_data[:,v])
        ax[v].set_xlim(forecast_times[0]-10, forecast_times[-1]+10)
        ax[v].grid()
        name = variable_names[v]
        ax[v].set_ylabel(name)
    ax[-1].set_xlabel('time')

def pdfs(ensemble_all_vals_ke, ensemble_all_vals_q, test_data_ke, test_data_q, prediction_times, ax=ax):
    flatten_ensembles_ke = ensemble_all_vals_ke.flatten()
    flatten_ensembles_q = ensemble_all_vals_q.flatten()

    max_pred_ke, min_pred_ke = np.max(flatten_ensembles_ke), np.min(flatten_ensembles_ke)
    max_pred_q, min_pred_q = np.max(flatten_ensembles_q), np.min(flatten_ensembles_q)
    max_true_ke, min_true_ke = np.max(test_data_ke), np.min(test_data_ke)
    max_true_q, min_true_q = np.max(test_data_q), np.min(test_data_q)
    min_ke = np.min((min_pred_ke, min_true_ke))
    max_ke = np.max((max_pred_ke, max_true_ke))
    min_q = np.min((min_pred_q, min_true_q))
    max_q = np.max((max_pred_q, max_true_q))

    bins_ke = np.linspace(-1e-4, 3e-4, 20)
    bins_q = np.linspace(0.265, 0.300, 20)

    ax[1,0].hist(flatten_ensembles_ke, bins=bins_ke, density=True)
    ax[1,1].hist(flatten_ensembles_q, bins=bins_q, density=True)
    ax[0,0].hist(test_data_ke, bins=bins_ke, density=True)
    ax[0,1].hist(test_data_q, bins=bins_q, density=True)
    ax[0, 0].set_xlim(min_ke, max_ke)
    ax[1, 0].set_xlim(min_ke, max_ke)
    ax[0, 1].set_xlim(min_q, max_q)
    ax[1, 1].set_xlim(min_q, max_q)
    ax[1, 0].set_xlabel('KE')
    ax[1, 1].set_xlabel('q')
    for i in range(2):
      for j in range(2):
        ax[i, j].set_ylabel('density')

def phase_diagram_boxplot(ensemble_all_vals_ke, ensemble_all_vals_q, test_data_ke, test_data_q, prediction_times, forecast_time=50, ax=ax):
    median_ke = np.median(ensemble_all_vals_ke, axis=1) 
    median_q = np.median(ensemble_all_vals_q, axis=1) 
    #ax.scatter(median_q[:forecast_time], median_ke[:forecast_time], marker='.', color='black')
    boxplot_2d(ensemble_all_vals_q[forecast_time,:],ensemble_all_vals_ke[forecast_time,:], ax, whis=1.5, choose_color='red')
    ax.set_xlim(0.265, 0.300)
    ax.set_ylim(0, 3e-4)
    ax.set_ylabel('KE')
    ax.set_xlabel('q')
    
#%% PEAKS 

def true_next_peak_multiple(test_data, prediction_times):
    '''
    finds peaks in true data
    inputs:
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    outputs:
    num_true_peaks - number of peaks in the prediction time (integer)
    true_next_peak_time - array of the times of the next true peaks (array)
    true_next_peak_value - array of the values of the amplitude of the next true peaks (array)
    '''
    # set thresholds
    threshold = 0.00005
    distance = 2 # 10 for same IC
    prominence = 0.000025 #0.000025

    peaks_true, _ = find_peaks(test_data[:,0], height=threshold, distance=distance, prominence = prominence)
    num_true_peaks = len(peaks_true)
    #print(prediction_times[peaks_true])
    true_next_peak_time = prediction_times[peaks_true] #records time of next peak
    true_next_peak_value = test_data[peaks_true,0]
    #print(num_true_peaks, true_next_peak_time, true_next_peak_value)

    return num_true_peaks, true_next_peak_time, true_next_peak_value

def pred_next_peak(pred, prediction_times):
    '''
    finds the next peak ONLY in predicted data
    inputs:
    pred - predicted data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    outputs:
    num_pred_peaks - number of peaks in the prediction time (integer)
    pred_next_peak_time - time of the next pred peak (float)
    pred_next_peak_value - value of the amplitude of the next predicted peak (float)
    '''
    # set thresholds
    threshold = 0.00005
    distance = 2 # 10 for same IC
    prominence = 0.000025 #0.000025

    peaks_pred, _ = find_peaks(pred[:,0], height=threshold, distance=distance, prominence = prominence)
    num_pred_peaks = len(peaks_pred)
    if num_pred_peaks > 0:
        pred_next_peak_time = prediction_times[peaks_pred[0]]
        pred_next_peak_value = pred[peaks_pred[0],0]
    else:
        #print('no peaks recorded')
        pred_next_peak_time = np.inf #no peak so set the next peak super far in advance so its not recorded
        pred_next_peak_value = 0

    return num_pred_peaks, pred_next_peak_time, pred_next_peak_value
    
def pred_next_peak_ensemble(ensemble_predictions, prediction_times):
    '''
    finds the next peak ONLY in predicted data ensemble
    inputs:
    ensemble_predictions - predicted data (array: (time, variables, ensembles))
    prediction_times - array of prediction times (array: (time))
    outputs:
    num_predicted_peaks - number of peaks in the prediction time for each ensemble member (list)
    pred_next_peak_time_all_members - time of the next pred peak for each ensemble member (list)
    pred_next_peak_value_all_members - value of the amplitude of the next predicted peak for each ensemble member (list)
    '''
    # set thresholds
    threshold = 0.00005
    distance = 2 # 10 for same IC
    prominence = 0.000025 #0.000025

    pred_next_peak_time_all_members = []
    pred_next_peak_value_all_members = []
    num_predicted_peaks = []

    ensembles = np.shape(ensemble_predictions)[-1]
    for r in range(ensembles):
        pp = pred_next_peak(ensemble_predictions[:,:,r], prediction_times)
        num_predicted_peaks.append(pp[0])
        pred_next_peak_time_all_members.append(pp[1])
        pred_next_peak_value_all_members.append(pp[2])

    return num_predicted_peaks, pred_next_peak_time_all_members, pred_next_peak_value_all_members
    
def fraction_of_peaks_within_t(true_next_peak_time, pred_next_peak_time_all_members, time_error=50):
    '''
    fraction of peaks in the ensemble that predict a peak within error of true peak (float, <= 1)
    inputs:
    true_next_peak_time - tru time of the next peak (float)
    pred_next_peak_time_all_member - predicted time of next peak in each ensemble member (array: (ensembles))
    '''
    counter = 0
    for value in pred_next_peak_time_all_members:
        #print(value)
        if abs(value - true_next_peak_time) <= time_error:
            counter += 1
    return counter/len(pred_next_peak_time_all_members)    
    
def peak_amplitude_error(true_next_peak_value, pred_next_peak_value_all_members):
    '''
    retuns the avg amplitude and a RMSE
    inputs:
    true_next_peak_time - tru time of the next peak (float)
    pred_next_peak_time_all_member - predicted time of next peak in each ensemble member (array: (ensembles))
    '''
    true_peak_values_array = np.ones((ensembles))*true_next_peak_value
    RMSE = np.sqrt(np.mean((pred_next_peak_value_all_members-true_next_peak_value)**2))
    avg_amplitude = np.mean(pred_next_peak_value_all_members)
    return avg_amplitude, RMSE

def peak_analysis(ensemble_predictions, test_data, prediction_times, time_error=50, inspect_plots=True):
    '''
    analysis on the peaks in the ensemble
    inputs:
    ensemble_predictions - predicted data (array: (time, variables, ensembles))
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    time_error - number of time steps for the peak to be within (integer)
    inspect_plots - shows plots (true/false)
    outputs:
    frac - fraction of peaks in the ensemble that predict a peak within error of true peak (float, <= 1)
    avg_amplitude - avg amplitude of the peak found in the predictions (float)
    RMSE - root mean square error between predicted peak and true peak across ensemble (float)
    peaks - ((number of esnemble memebrs with correct number of peaks, number missing one peak), mean number of peaks in ensemble, min number of peaks in ensemble, max number of peaks in ensemble)
    '''
    num_predicted_peaks, pred_next_peak_time, pred_next_peak_value = pred_next_peak_ensemble(ensemble_predictions, prediction_times)
    num_true_peaks, true_next_peak_time, true_next_peak_value = true_next_peak_multiple(test_data, prediction_times)

    frac = fraction_of_peaks_within_t(true_next_peak_time[0], pred_next_peak_time, time_error=time_error)
    avg_amplitude, RMSE = peak_amplitude_error(true_next_peak_value[0], pred_next_peak_value)

    number_peaks = number_of_peaks_captured(num_true_peaks, num_predicted_peaks)

    if inspect_plots == True:
        fig,ax = plt.subplots(1, figsize=(4,6))
        ax.boxplot(pred_next_peak_value[:], meanline=True, showmeans=True)
        ax.scatter(1, true_next_peak_value[0])
        ax.set_ylabel('KE peak value')
        ax.grid()
        print(np.mean(pred_next_peak_value[:]))
        print(np.abs(np.mean(pred_next_peak_value[:]) - true_next_peak_value[0])/((np.mean(pred_next_peak_value[:]) + true_next_peak_value[0])/2) * 100)

    return frac, avg_amplitude, RMSE, (number_peaks, np.mean(num_predicted_peaks), np.min(num_predicted_peaks), np.max(num_predicted_peaks))

def peaks_plot(test_data, prediction_times, ax=ax):
    '''
    plots the true peaks across a prediction interval
    test_data - true data (array: (time, variables))
    prediction_times - array of prediction times (array: (time))
    '''
    tp = true_next_peak_multiple(test_data, prediction_times)
    #print(tp[1], tp[2])
    ax.scatter(tp[1], tp[2], marker='x', color='red')

def FFT(variable, time):
    variable = variable - np.mean(variable)
    fs = len(time)/(time[-1]-time[0])
    end = time[-1]
    start = time[0]
    fft = np.fft.fft(variable)
    fft = np.fft.fftshift(fft)
    om = np.fft.fftfreq(len(time), d=(end-start)/len(time))
    om = np.fft.fftshift(om)
    #om = 2*np.pi*om
    magnitude_w = abs(fft)
    psd = magnitude_w
    return om, psd

