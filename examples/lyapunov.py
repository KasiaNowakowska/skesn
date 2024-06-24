"""
python script for ESN grid search.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
"""

# import packages
import time

import sys
sys.path.append('/nobackup/mm17ktn/ENS/skesn/skesn/')

from math import isclose, log
import os
sys.path.append(os.getcwd())
print(sys.path)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import linregress

import h5py

from docopt import docopt
args = docopt(__doc__)

import nolds

input_path = args['--input_path']
output_path = args['--output_path']

#load in data
q = np.load(input_path+'/q.npy')
ke = np.load(input_path+'/KE.npy')
wallz = np.load(input_path+'/wallz.npy')
time_vals = np.load(input_path+'/time_vals.npy')
total_time = np.load(input_path+'/total_time.npy')
print(len(q), len(ke), len(time_vals))
print(np.shape(wallz), len(total_time))
print(total_time[50], total_time[-1])

total_time = total_time[50:]
w = wallz[50:,:,:]

x_pos = 160
z_pos = 32
w_point = w[:,x_pos,z_pos]

x_obs = w_point[:10000] #ke[:10000]
#dt = (time_vals[-1]-time_vals[0])/len(time_vals)
dt = (total_time[-1]-total_time[0])/len(total_time)
print('dt = ', dt)

### FINDS PAIRS OF NEARBY CONDITIONS ###

from scipy.signal import periodogram
def distance(xe, xi):
    return np.sqrt(np.sum((xi - xe)**2))

def get_nearest_neighbour(xi, X, mu, time_steps):
    #print(X[xi])
    xes = np.arange(len(X) - time_steps)  # Indices for potential nearest neighbors within a specified range
    #print('xes', xes)
    # Calculate distances to potential neighbors
    ds = np.array([distance(X[xe], X[xi]) for xe in xes])
    #print(xi, 'ds=', ds)
    #print(len(ds))
    # Set distance to infinity if it's the same vector or too close based on muu
    ds = np.where(ds == 0, np.inf, ds)
    ds = np.where(np.abs(xi - xes) < mu, np.inf, ds)
    #print(xi, np.argmin(ds))
    return np.argmin(ds)

def get_nearest_neighbours(X, mu, time_steps):
    gnn = [get_nearest_neighbour(i, X, mu, time_steps) for i in range(len(X))]
    return gnn

def mean_period(ts):
    freq, spec = periodogram(ts)
    w = spec / np.sum(spec)
    mean_frequency = np.sum(freq * w)
    return 1 / mean_frequency

def lyap(ts, J, m, t_end, time_steps):
  if time_steps < t_end:
      print("x is not greater than y. Stopping function.")
      return
  N = len(ts)
  M = N - (m - 1) * J
  print('J:', J, 'm:', m, 'N:', N, 'M:', M)

  X = np.full((M,m), np.nan)

  # Populate matrix X based on the loop logic
  for i in range(M):
      idx = np.arange(i, i + (m - 1) * J + 1, J)
      X[i, :] = ts[idx]

  j = get_nearest_neighbours(X, mu=mean_period(ts), time_steps=time_steps)

  #### estimate mean rate of seperation of nearest neighbours ####
  def expected_log_distance(i, X):
      n = len(X)
      d_ji = np.array([distance(X[j[k] + i], X[k + i]) for k in range(1, n - i)])
      log_only = np.log(d_ji)
      mean = np.mean(np.log(d_ji))
      return mean, log_only

  mean_log_distance = [expected_log_distance(i, X)[0] for i in range(t_end + 1)]
  distance_log_i = [expected_log_distance(i, X)[1] for i in range(t_end + 1)]
  time_innovation = np.arange(t_end + 1)

  return time_innovation, mean_log_distance, distance_log_i, X

### to find J ###
'''
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(x_obs, lags=100)
plt.axhline(1-1/np.e)
plt.xlim(90,94)
plt.show()
'''

# Compute the autocorrelation function
autocorr_result = np.correlate(x_obs-np.mean(x_obs), x_obs-np.mean(x_obs), mode='full')
autocorr = autocorr_result[len(autocorr_result) // 2:]
autocorr /= autocorr[0] #normalise by autocorrelation at lag 0


# Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

# Find the first lag where autocorrelation drops below the threshold
J_value = np.where(autocorr < threshold)[0][0]

'''
### m investigation ####
m_0 = np.arange(2,20,1)
t_end = 200
time_steps = 300
dist = np.zeros((len(m_0), t_end+1))
time_innovation = np.arange(t_end + 1)
time_values = time_innovation*dt
np.save(output_path+'/time_values.npy', time_values)
for m0 in range(len(m_0)):
  J = J_value
  m = m_0[m0]
  print('m=', m)
  time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
  dist[m0,:] = mean_log_distance

  np.save(output_path+'/mean_log_distances.npy', dist)

### length investigation ####
x_obs_index = np.arange(5000, 13000,1000)
t_end = 2000
time_steps = 3000
dist = np.zeros((len(x_obs_index), t_end+1))
time_innovation = np.arange(t_end + 1)
time_values = time_innovation*dt
np.save(output_path+'/time_values.npy', time_values)
for N in range(len(x_obs_index)):
  index = x_obs_index[N]
  x_obs = ke[:index]
  # Compute the autocorrelation function
  autocorr_result = np.correlate(x_obs-np.mean(x_obs), x_obs-np.mean(x_obs), mode='full')
  autocorr = autocorr_result[len(autocorr_result) // 2:]
  autocorr /= autocorr[0] #normalise by autocorrelation at lag 0


  # Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
  threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

  # Find the first lag where autocorrelation drops below the threshold
  J_value = np.where(autocorr < threshold)[0][0]
  J = J_value
  m = 8
  print('N=', len(x_obs))
  time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
  dist[N,:] = mean_log_distance

  np.save(output_path+'/mean_log_distances.npy', dist)

#### nonglobal RB points investigation ####
x_positions = np.arange(32,256,32)
z_positions = np.arange(16,64,16)
t_end = 200
time_steps = 300
dist = np.zeros((len(x_positions)*len(z_positions), t_end+1))
time_innovation = np.arange(t_end + 1)
time_values = time_innovation*dt
np.save(output_path+'/time_values.npy', time_values)
conuter = 0
for xi in range(len(x_positions)):
  for zi in range(len(z_positions)):
    w_point = w[:,xi,zi]
    x_obs = w_point

    # Compute the autocorrelation function
    autocorr_result = np.correlate(x_obs-np.mean(x_obs), x_obs-np.mean(x_obs), mode='full')
    autocorr = autocorr_result[len(autocorr_result) // 2:]
    autocorr /= autocorr[0] #normalise by autocorrelation at lag 0


    # Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
    threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

    # Find the first lag where autocorrelation drops below the threshold
    J_value = np.where(autocorr < threshold)[0][0]
    J = J_value
    m = 8
    print('N=', len(x_obs))
    time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
    dist[counter,:] = mean_log_distance
    counter += 1

  np.save(output_path+'/mean_log_distances.npy', dist)
'''
##### CP4 Data ######

# functions
def hours_to_custom_datetime(hours):
    # Constants
    hours_per_day = 24
    days_per_month = 30
    months_per_year = 12
    days_per_year = 360

    # Calculate total days and remaining hours
    total_days = hours // hours_per_day
    remaining_hours = hours % hours_per_day
    #print(total_days, remaining_hours)

    # Calculate the year
    year = total_days // days_per_year
    remaining_days = total_days % days_per_year
    #print(year, remaining_days)

    # Calculate the month
    month = remaining_days // days_per_month
    day = remaining_days % days_per_month

    # Adjust month and year for 0-based index
    month += 1
    day += 1
    year += 1970

    return year, month, day, remaining_hours

def convert_custom_to_datetime(years, months, days, hours):
    datetime_strings = ["{:04d}-{:02d}-{:02d} {:02d}:00:00".format(y, m, d, h)
                    for y, m, d, h in zip(years, months, days, hours)]
    return datetime_strings

#load in CP4 data
file_path='../data/CP4data/data_R25_C_c30404_NIM.txt'
data = pd.read_csv(file_path, header=None, names=['Hours', 'TCW'])
file_path2='../data/CP4data/data_CP4_regridded_C_a04203_NIM.txt'
data2 = pd.read_csv(file_path2, header=None, names=['Hours', 'precip'])

data['precip'] = data2['precip']

#plot TCW and precip against hours
TCW = np.array(data['TCW'])
Hours = np.array(data['Hours'])
precip = np.array(data['precip'])
fig, ax = plt.subplots(2, figsize=(12,6), sharex=True)
ax[0].plot(Hours, TCW)
ax[1].plot(Hours, precip)
ax[0].set_ylabel('TCW')
ax[1].set_ylabel('precip')
ax[1].set_xlabel('Hours from 1970-01-01 00:00')

CorrectTimeCP4 = np.zeros((4, len(Hours)), dtype=int)
CorrectTimeCP4[:,:] = hours_to_custom_datetime(Hours)
Time = convert_custom_to_datetime(CorrectTimeCP4[0,:], CorrectTimeCP4[1,:], CorrectTimeCP4[2,:], CorrectTimeCP4[3,:])

Year_in_Hours = 360*24
Month_in_hours = 30*24

#plot TCW and precip against yearly dates
fig, ax =plt.subplots(2, figsize=(12,6), tight_layout=True, sharex=True)
ax[0].plot(Hours, TCW)
ax[1].plot(Hours, precip)
ax[1].set_xlabel('Time')
ax[0].set_ylabel('TCW')
ax[1].set_ylabel('precip')
ax[0].grid()
ax[1].grid()

step = Year_in_Hours

tick_positions = Hours[::step]
tick_labels = Time[::step]
ax[1].set_xticks(tick_positions, tick_labels, rotation=90)

#choose domain size
y = 6
m= 0
Index_start = Year_in_Hours * y + Month_in_hours * m
Index_end = Year_in_Hours * (y+1) + Month_in_hours * (m)
print('start and end times:')
print(Time[Index_start], Time[Index_end])

x_obs = TCW[Index_start:Index_end]

### to find J ###
# Compute the autocorrelation function
autocorr_result = np.correlate(x_obs-np.mean(x_obs), x_obs-np.mean(x_obs), mode='full')
autocorr = autocorr_result[len(autocorr_result) // 2:]
autocorr /= autocorr[0] #normalise by autocorrelation at lag 0


# Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

# Find the first lag where autocorrelation drops below the threshold
J_value = np.where(autocorr < threshold)[0][0]

m_0 = np.arange(2,20,1)
t_end = 100
time_steps = 200
dist = np.zeros((len(m_0), t_end+1))
time_innovation = np.arange(t_end + 1)
time_values = time_innovation*1
np.save(output_path+'/time_values.npy', time_values)
for m0 in range(len(m_0)):
  J = J_value
  m = m_0[m0]
  print('m=', m)
  time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
  dist[m0,:] = mean_log_distance

  np.save(output_path+'/mean_log_distances.npy', dist)


'''
###### OLDER STUFF ####
m_0 = np.arange(2,13,1)
LLEs = np.zeros((len(m_0)))
for m0 in range(len(m_0)):
  J = J_value
  m = m_0[m0]
  print('m=', m)
  t_end = 2000
  time_steps = 3000
  time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
  time_values = time_innovation*1
  fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
  ax.plot(time_values[:], mean_log_distance[:], 'b-')
  ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
  ax.set_ylabel('ln $\hat{d}$', fontsize=18)
  #plt.title('Mean Log Distance over Time')
  ax.grid()
  ax.tick_params(axis='x', labelsize=14)
  ax.tick_params(axis='y', labelsize=14)
  for s in range(len(slope_end)):
      slopeend = slope_end[s]
      slope, intercept, r_value, p_value, std_err = linregress(time_values[:slopeend], mean_log_distance[:slopeend])
      print(slope)
      LLEs[s, m0] = slope
      best_fit = slope*time_values[:slopeend] + intercept
      ax.plot(time_values[:slopeend], best_fit, linestyle='--', color='orange', label='slope ends at %i' % slopeend)
  fig.savefig(output_path+'/LLE_m%i.png' % m0)
  plt.close()
'''
'''
x_obs_list = (ke[:2500], ke[:5000], ke[:7500], ke[:10000])
LLEs = np.zeros((len(x_obs_list)))
for N in range(len(x_obs_list)):
  x_obs = x_obs_list[N]
  # Compute the autocorrelation function
  autocorr_result = np.correlate(x_obs-np.mean(x_obs), x_obs-np.mean(x_obs), mode='full')
  autocorr = autocorr_result[len(autocorr_result) // 2:]
  autocorr /= autocorr[0] #normalise by autocorrelation at lag 0


  # Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
  threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

  # Find the first lag where autocorrelation drops below the threshold
  J_value = np.where(autocorr < threshold)[0][0]
  J = J_value
  m = 6
  print('N=', len(x_obs))
  t_end = 1500
  time_steps = 2000
  time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)
  time_values = time_innovation*1
  fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
  ax.plot(time_values[:], mean_log_distance[:], 'b-')
  ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
  ax.set_ylabel('ln $\hat{d}$', fontsize=18)
  #plt.title('Mean Log Distance over Time')
  ax.grid()
  ax.tick_params(axis='x', labelsize=14)
  ax.tick_params(axis='y', labelsize=14)
  slopeend = 750
  slope, intercept, r_value, p_value, std_err = linregress(time_values[:slopeend], mean_log_distance[:slopeend])
  print(slope)
  LLEs[N] = slope
  best_fit = slope*time_values[:slopeend] + intercept
  ax.plot(time_values[:slopeend], best_fit, linestyle='--', color='orange', label='slope ends at %i' % slopeend)
  fig.savefig(output_path+'/LLE_N%i.png' % N)
  plt.close()

np.save(output_path+'/LLEs.npy', LLEs)

J = J_value
m = 2
t_end = 2000
time_steps = 2100
time_innovation, mean_log_distance, distance_log_i, X = lyap(x_obs, J, m, t_end, time_steps)


time_values = time_innovation*dt
fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], mean_log_distance[:], 'b-')
ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
ax.set_ylabel('ln $\hat{d}$', fontsize=18)
#plt.title('Mean Log Distance over Time')
ax.grid()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.savefig()

slope_end = 600
slope, intercept, r_value, p_value, std_err = linregress(time_values[:slope_end], mean_log_distance[:slope_end])
'''

