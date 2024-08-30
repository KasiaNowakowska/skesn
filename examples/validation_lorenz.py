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
total_time = 200
all_data, times = _lorenz([1.0, 1.0, 1.0], dt=esn_dt, t_final=total_time) #[0.1, 0.2, 25.],
print(np.shape(all_data))
all_data = all_data[:, :]
ss = StandardScaler()
data = ss.fit_transform(all_data)
dt = esn_dt

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

ensembles = 10
splits = 5
LLE = 1.50
LT = 1/LLE
n_prediction_inLT = 10
n_train_inLT = 20
n_sync = 0


MSE_params, PH_params = grid_search_WFV(EsnForecaster, param_grid, data, times, ensembles, splits, LT, n_prediction_inLT, n_train_inLT, n_sync)

np.save(output_path+'/MSE_params.npy', MSE_params)
np.save(output_path+'/PH_params.npy', PH_params)

# Generate all combinations of parameters
# Get the parameter labels and values
param_labels = list(param_grid.keys())
param_values = list(param_grid.values())

param_combinations = list(product(*param_values))

# Create a DataFrame where each row is a different set of parameters
df = pd.DataFrame(param_combinations, columns=param_labels)

print(df)

MSE_mean = np.zeros((MSE_params.shape[0])) #number of combinations
for i in range(MSE_params.shape[0]):
    MSE_mean[i] = np.mean(MSE_params[i,:])
    
PH_mean_02 = np.zeros((PH_params.shape[0]))
PH_mean_05 = np.zeros((PH_params.shape[0]))
PH_mean_10 = np.zeros((PH_params.shape[0]))
for i in range(PH_params.shape[0]):
    PH_mean_02[i] = np.mean(PH_params[i,:,0])
    PH_mean_05[i] = np.mean(PH_params[i,:,1])
    PH_mean_10[i] = np.mean(PH_params[i,:,2])
    
MSE_mean_geom = np.exp(np.mean(np.log(MSE_params), axis=1))

np.save(output_path+'/MSE_means.npy', MSE_mean)
np.save(output_path+'/PH_means_02.npy', PH_mean_02)
np.save(output_path+'/PH_means_02.npy', PH_mean_05)
np.save(output_path+'/PH_means_02.npy', PH_mean_10)
np.save(output_path+'/MSE_means_geom.npy', MSE_mean_geom)

df_copy = df.copy(deep=True)
df_copy['mean_MSE'] = MSE_mean
df_copy['mean_PH_02'] = PH_mean_02
df_copy['mean_PH_05'] = PH_mean_05
df_copy['mean_PH_10'] = PH_mean_10
df_copy['mean_MSE_geom'] = MSE_mean_geom

# Save df_copy as CSV
df_copy.to_csv(output_path+'/Lorenzgridsearch_dataframe.csv', index=False)
print('data frame saved')

