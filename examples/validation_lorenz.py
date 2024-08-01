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
from validation_stratergies import *

from scipy.signal import find_peaks

import h5py
from scipy import stats
from itertools import product

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

output_path='../data/Lorenz/'

def _lorenz(x_0, dt, t_final):
    sigma_ = 16.
    beta_ = 4
    rho_ = 45.92

    def rhs(x):
        f_ = np.zeros(3)
        f_[0] = sigma_ * (x[1] - x[0])
        f_[1] = rho_ * x[0] - x[0] * x[2] - x[1]
        f_[2] = x[0] * x[1] - beta_ * x[2]
        return f_

    times = np.arange(0, t_final, dt)
    ts = np.zeros((len(times), 3))
    ts[0, :] = x_0
    cur_x = x_0
    dt_integr = 10**(-3)
    n_timesteps = int(np.ceil(dt / dt_integr))
    dt_integr = dt / n_timesteps
    for i in range(1, n_timesteps*len(times)):
        cur_x = cur_x + dt_integr * rhs(cur_x)
        saved_time_i = i*dt_integr / dt
        if isclose(saved_time_i, np.round(saved_time_i)):
            saved_time_i = int(np.round(i*dt_integr / dt))
            ts[saved_time_i, :] = cur_x
    return ts, times

coord_names = [r'$X_t$', r'$Y_t$', r'$Z_t$']
esn_dt = 0.01
total_time=10
all_data, times = _lorenz([1.0, 1.0, 1.0], dt=esn_dt, t_final=total_time) #[0.1, 0.2, 25.],
print(np.shape(all_data))
all_data = all_data[:, 0:3:2]
ss = StandardScaler()
all_data = ss.fit_transform(all_data)
ts = all_data[:10000]
train_times = times[:10000]

param_grid = dict(n_reservoir=[300],
                  spectral_radius=[0.20,0.30,0.40],
                  sparsity=[0.80],
                  regularization=['l2'],
                  lambda_r=[0.0001,0.001,0.01],
                  in_activation=['tanh'],
                  out_activation=['identity'],
                  use_additive_noise_when_forecasting=[True],
                  random_state=[42],
                  use_bias=[True],
                  use_b=[True],
                  input_scaling=[1.0])

model = EsnForecaster(
        n_reservoir=300,
        spectral_radius=0.20,
        sparsity=0.8,
        regularization='l2',
        lambda_r=0.0001,
        in_activation='tanh',
        out_activation='identity',
        use_additive_noise_when_forecasting=True,
        random_state=42,
        use_bias=True,
        use_b=True)

n_prediction = 198
trainlen = 528
synclen = 5
ensembles = 50

MSE_params, PH_params = grid_search_SSV(EsnForecaster, param_grid, all_data, times, ensembles, n_prediction, trainlen, synclen)

# Generate all combinations of parameters
# Get the parameter labels and values
param_labels = list(param_grid.keys())
param_values = list(param_grid.values())

param_combinations = list(product(*param_values))

# Create a DataFrame where each row is a different set of parameters
df = pd.DataFrame(param_combinations, columns=param_labels)

print(df)

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

df_copy = df.copy(deep=True)
df_copy['mean_MSE'] = MSE_mean
df_copy['mean_PH'] = PH_mean

# Save df_copy as CSV
df_copy.to_csv(output_path+'/gridsearch_dataframe.csv', index=False)
print('data frame saved')
