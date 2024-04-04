# import packages
import time

from math import isclose
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

File1 = '/content/drive/MyDrive/Colab Notebooks/numpyarrays/rainy_restarted_Ra2.00e+08_RH00.75_RHb0.75_RHt0.75_gamma0.5_nx256_nz64_Lx20_simtime15000_withbudget/timeseries/upto15000/timeseries_s1.h5'
df = h5py.File(File, 'r')