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
    
def timeseries_plustraining(ke_prediction, q_prediction, test_data_ke, test_data_q, train_data_ke, train_data_q, train_times, prediction_times, ax=ax):
    '''
    plots the training, prediction and truth timeseries for one single ensemble memeber
    '''
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    ax[0].plot(train_times, train_data_ke, linewidth=2,
                                color='tab:blue')
    ax[1].plot(train_times, train_data_q, linewidth=2,
                                color='tab:blue')
    ax[0].fill_between(train_times, train_data_ke.min(), train_data_ke.max(), alpha=0.2, color='tab:blue', label='Training')
    ax[1].fill_between(train_times, train_data_q.min(), train_data_q.max(), alpha=0.2, color='tab:blue', label='Training')
    ax[0].plot(prediction_times, ke_prediction,
                                linewidth=2,
                                label='Prediction', color='orange')
    ax[1].plot(prediction_times, q_prediction,
                                linewidth=2,
                                label='Prediction', color='orange')
    ax[0].plot(prediction_times, test_data_ke, linewidth=2,
                                label='True', color='tab:blue')    
    ax[1].plot(prediction_times, test_data_q, linewidth=2,
                                label='True', color='tab:blue')
    plt.legend()
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    ax[1].set_xlabel('time')
    print('added prediction')

def phase_space_prediction(ke_prediction, q_prediction, test_data_ke, test_data_q, train_data_ke, train_data_q, train_times, prediction_times, ax=ax, fig=fig):
    '''
    plots the phase space prediction for prediction and true for one single ensemble member
    '''
    plt.grid()
    s_pred = ax.scatter(q_prediction, ke_prediction, 
                            cmap='viridis', marker='x',
                            c=prediction_times, vmin=prediction_times[0], vmax=prediction_times[-1], label='Prediction')
    s_true = ax.scatter(test_data_q, test_data_ke,
                            cmap='viridis', marker='.', c=prediction_times,
                            vmin=prediction_times[0], vmax=prediction_times[-1], label='True')
    ax.set_xlabel('q')
    ax.set_ylabel('KE')
    ax.set_ylim(-0.0001, 0.00035)
    ax.set_xlim(0.260,0.30)
    cbar_true = fig.colorbar(s_true, ax=ax, label='time')
    plt.legend()

def ensemble_timeseries_plustraining(ensemble_all_vals_ke, ensemble_all_vals_q, test_data_ke, test_data_q, train_data_ke, train_data_q, train_times, prediction_times, ax=ax):
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    median_ke = np.median(ensemble_all_vals_ke, axis=1)
    median_q = np.median(ensemble_all_vals_q, axis=1)
    lower_bound_ke = np.percentile(ensemble_all_vals_ke, 10, axis=1)
    upper_bound_ke = np.percentile(ensemble_all_vals_ke, 90, axis=1)
    lower_bound_q = np.percentile(ensemble_all_vals_q, 10, axis=1)
    upper_bound_q = np.percentile(ensemble_all_vals_q, 90, axis=1)
    ax[0].plot(train_times, train_data_ke, linewidth=2,
                                color='tab:blue')
    ax[1].plot(train_times, train_data_q, linewidth=2,
                                color='tab:blue')
    ax[0].plot(prediction_times, median_ke,
                                linewidth=2,
                                label='Median Prediction', color='orange')
    ax[0].fill_between(prediction_times, lower_bound_ke, upper_bound_ke, color='orange', alpha=0.3, label='80% Confidence Interval')

    ax[1].plot(prediction_times, median_q,
                                linewidth=2,
                                label='Median Prediction', color='orange')
    ax[1].fill_between(prediction_times, lower_bound_q, upper_bound_q, color='orange', alpha=0.3, label='80% Confidence Interval')
    ax[0].plot(prediction_times, test_data_ke, linewidth=2,
                                label='True', color='tab:blue')
    ax[1].plot(prediction_times, test_data_q, linewidth=2,
                                label='True', color='tab:blue')
    plt.legend()
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    ax[1].set_xlabel('time')
    ax[0].set_xlim(10000, prediction_times[-1])
    ax[1].set_xlim(10000, prediction_times[-1])
    ax[0].grid()
    ax[1].grid()
    print('added prediction')

def ensemble_timeseries(ensemble_all_vals_ke, ensemble_all_vals_q, test_data_ke, test_data_q, prediction_times, ax=ax):
    '''
    plots the median prediction and 80% confidence interval
    '''
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    median_ke = np.median(ensemble_all_vals_ke, axis=1) 
    median_q = np.median(ensemble_all_vals_q, axis=1) 
    lower_bound_ke = np.percentile(ensemble_all_vals_ke, 10, axis=1)
    upper_bound_ke = np.percentile(ensemble_all_vals_ke, 90, axis=1)
    lower_bound_q = np.percentile(ensemble_all_vals_q, 10, axis=1)
    upper_bound_q = np.percentile(ensemble_all_vals_q, 90, axis=1)
    
    ax[0].plot(prediction_times, median_ke,
                                linewidth=2,
                                label='Median Prediction', color='green')
    ax[0].fill_between(prediction_times, lower_bound_ke, upper_bound_ke, color='green', alpha=0.3, label='80% Confidence Interval')

    ax[1].plot(prediction_times, median_q,
                                linewidth=2,
                                label='Median Prediction')
    ax[1].fill_between(prediction_times, lower_bound_q, upper_bound_q, color='green', alpha=0.3, label='80% Confidence Interval')
    plt.legend()
    print('added prediction')

def meteogram(ensemble_all_vals_ke, ensemble_all_vals_q, test_data_ke, test_data_q, prediction_times, forecast_interval=75, ax=ax):
    if ax.shape != (2,):
        print('ax shape needs to be 2')
        return
    ensembles = ensemble_all_vals_ke.shape[-1]
    no_of_forecasts = int(len(prediction_times)/forecast_interval) + 1
    q_forecast = np.zeros((no_of_forecasts, ensembles))
    ke_forecast = np.zeros((no_of_forecasts, ensembles))
    forecast_times = np.zeros((no_of_forecasts))
    print(no_of_forecasts)

    forecast_time = 0
    ensembles = len(ensemble_all_vals_ke[0,:])
    for t in range(len(prediction_times)):
      if t % forecast_interval == 0:
        q_forecast[forecast_time, :] = ensemble_all_vals_q[t,:]
        ke_forecast[forecast_time, :] = ensemble_all_vals_ke[t,:]
        forecast_times[forecast_time] = int(prediction_times[t])
        forecast_time += 1

    ax[0].boxplot(ke_forecast.T, positions=forecast_times, labels=forecast_times, widths=10)
    ax[1].boxplot(q_forecast.T, positions=forecast_times, labels=forecast_times, widths=10)
    ax[0].plot(prediction_times, test_data_ke)
    ax[1].plot(prediction_times, test_data_q)
    ax[0].set_ylabel('KE')
    ax[1].set_ylabel('q')
    ax[1].set_xlabel('time')
    ax[0].set_xlim(forecast_times[0]-10, forecast_times[-1]+10)
    ax[1].set_xlim(forecast_times[0]-10, forecast_times[-1]+10)
    ax[0].grid()
    ax[1].grid()

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

def true_next_peak(test_data_ke, test_data_q, prediction_times):
    # set thresholds
    threshold = 0.00005
    distance = 10
    prominence = 0.000025

    peaks_true, _ = find_peaks(test_data_ke, height=threshold, distance=distance, prominence = prominence)
    num_true_peaks = len(peaks_true)
    print(prediction_times[peaks_true])
    true_next_peak_time = prediction_times[peaks_true[0]] #records time of next peak
    true_next_peak_value = test_data_ke[peaks_true[0]]
    print(num_true_peaks, true_next_peak_time, true_next_peak_value)

    return num_true_peaks, true_next_peak_time, true_next_peak_value
    
def pred_next_peak(pred_ke, pred_q, prediction_times):
    # set thresholds
    threshold = 0.00005
    distance = 10
    prominence = 0.000025

    peaks_pred, _ = find_peaks(pred_ke, height=threshold, distance=distance, prominence = prominence)
    num_pred_peaks = len(peaks_pred)
    if num_pred_peaks > 0:
        pred_next_peak_time = prediction_times[peaks_pred[0]]
        pred_next_peak_value = pred_ke[peaks_pred[0]]
    else:
        #print('no peaks recorded')
        pred_next_peak_time = np.inf #no peak so set the next peak super far in advance so its not recorded
        pred_next_peak_value = np.NaN

    return num_pred_peaks, pred_next_peak_time, pred_next_peak_value

def peaks_plot(test_data_ke, test_data_q, prediction_times, ax=ax):
    tp = true_next_peak(test_data_ke, test_data_q, prediction_times)
    #print(tp[1], tp[2])
    ax[0].scatter(tp[1], tp[2])
