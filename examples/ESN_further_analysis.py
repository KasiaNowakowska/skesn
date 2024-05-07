"""
python script for ESN grid search.

Usage: ESN.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
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

input_path = args['--input_path']
output_path = args['--output_path']

output_path2 = output_path+'/2dboxplots'
if not os.path.exists(output_path2):
    os.makedirs(output_path2)
    print('made directory')

#load in data
ensemble_all_vals_ke  = np.load(input_path+'/ensemble_all_vals_ke.npy')
ensemble_all_vals_q = np.load(input_path+'/ensemble_all_vals_q.npy')
test_data_ke = np.load(input_path+'/test_data_ke.npy')
test_data_q = np.load(input_path+'/test_data_q.npy')

median_ke = np.median(ensemble_all_vals_ke, axis=1) 
median_q = np.median(ensemble_all_vals_q, axis=1) 


def boxplot_2d(x,y, ax, whis=1.5, choose_color='black'):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = choose_color,
        fc = choose_color,
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

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
fig, axs = plt.subplots(1, figsize=(6,6), tight_layout=True)
cols = ['blue', 'red', 'green', 'skyblue', 'salmon', 'purple']
time_slots = np.arange(50,600,50)
for i in range(len(time_slots)):
    fig, axs = plt.subplots(1, figsize=(6,6), tight_layout=True)
    time_slot = time_slots[i]
    axs.scatter(median_q[:time_slot], median_ke[:time_slot], marker='.', color='black')
    boxplot_2d(ensemble_all_vals_q[time_slot,:],ensemble_all_vals_ke[time_slot,:], axs, whis=1.5, choose_color=cols[i%len(cols)])
    axs.set_xlim(0.265, 0.300)
    axs.set_ylim(0, 3e-4)
    axs.set_ylabel('KE')
    axs.set_xlabel('q')
    fig.savefig(output_path2+'/phasespace_clears{:04d}.png'.format(time_slot))



