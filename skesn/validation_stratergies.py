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
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

from esn_old_adaptations import EsnForecaster
from cross_validation import ValidationBasedOnRollingForecastingOrigin

from scipy.signal import find_peaks

import h5py
from scipy import stats
from itertools import product

from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

#Define functions
def PH_error(predictions, true_values):
    "input: predictions, true_values as (time, variables)"
    variables = predictions.shape[1]
    mse = np.mean((true_values-predictions) ** 2, axis = 1)
    rmse = np.sqrt(mse)
    #denominator = np.max(true_values) - np.min(true_values)

    norm = np.linalg.norm(true_values, axis=1)
    norm_squared = norm ** 2
    mean_norm_squared = np.mean(norm_squared)
    #denominator = np.sqrt(norm_squared)
    norm_factor = np.sqrt(mean_norm_squared)
    # Avoid division by zero
    norm_factor = np.where(norm_factor == 0, 1, norm_factor)

    PH = rmse/norm_factor

    return PH

def prediction_horizon(predictions, true_values, threshold=0.2):
    error = PH_error(predictions, true_values)
    #print(np.shape(np.where(error > threshold)))
    shape = np.shape(np.where(error > threshold))
    if shape[1] == 0:
        PredictabilityHorizon = predictions.shape[0]
    else:
        PredictabilityHorizon = np.where(error > threshold)[0][0]

    return PredictabilityHorizon

def MSE(predictions, true_values):
    Nu = predictions.shape[1]
    norm = np.zeros((predictions.shape[0]))
    for i in range(true_values.shape[0]):
        norm[i] = np.linalg.norm(true_values[i,:] - predictions[i,:])

    norm_squared = np.mean(norm**2)
    MSE = 1/Nu * norm_squared

    return MSE

def grid_search_SSV_retrained(forecaster, param_grid, data, time_vals, ensembles_retrain, n_prediction, n_train, n_sync):
    """
    Grid Search (single shot validation).
    Takes a hyper paramater combination and trains an ensemble of networks which make 1 prediction each.

    Variables:
    forecaster: the model to use.
    param_grid: hyperparameter grid.
    data: Training and validation data.
    time_vals: time series of training times and validation times.
    ensembles_retrain: number of ensembles.
    n_prediciton: number of prediction time steps.
    n_train: number of trianing time steps.
    n_sync: number of synchronisation steps.

    outputs:
    MSE_params: (array: number_of_hyperparam_combination x number_of_ensemble) the MSE for each ensemble
                                                             member for each set of hyperparameters
    PH_params: (array: number_of_hyperparam_combination x number_of_ensemble) the PH for each ensemble
                                                             member for each set of hyperparameters
    """
    print(param_grid)

    ss = StandardScaler()
    synclen = n_sync
    trainlen = n_train - synclen
    predictionlen = n_prediction
    ts = data[0:trainlen,:]
    train_times = time_vals[0:trainlen]
    dt = time_vals[1] - time_vals[0]
    future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
    sync_times = time_vals[trainlen:trainlen+synclen]
    prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
    test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
    sync_data = data[trainlen:trainlen+synclen]
    future_data = data[trainlen:trainlen+synclen+predictionlen, :]
    variables = data.shape[-1]

    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())

    PH_params = np.zeros((num_combinations, ensembles_retrain))
    MSE_params = np.zeros((num_combinations, ensembles_retrain))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        #print(param_dict)
        modelsync = forecaster(**param_dict)
        print(modelsync)
        # Now you can use `modelsync` with the current combination of parameters

        ensemble_all_vals_unscaled = np.zeros((len(prediction_times), variables, ensembles_retrain))
        MSE_ens = np.zeros((ensembles_retrain))
        PH_ens = np.zeros((ensembles_retrain))

        for i in range(ensembles_retrain):
            print('ensemble member =', i)
            modelsync.fit(ts)
            #synchronise data
            modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
            #predict
            future_predictionsync = modelsync.predict(predictionlen)

            ### ensemble mean ###
            for v in range(variables):
                ensemble_all_vals_unscaled[:, v, i] = future_predictionsync[:,v]
                
            mse = MSE(ensemble_all_vals_unscaled[:,:,i], test_data[:,:])
            MSE_ens[i] = mse
            ph = prediction_horizon(ensemble_all_vals_unscaled[:,:,i], test_data[:,:], threshold = 0.1)
            PH_ens[i] = ph

        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:] = PH_ens
        counter += 1
    return MSE_params, PH_params

def grid_search_SSV(forecaster, param_grid, data, time_vals, ensembles, n_prediction, n_train, n_sync):
    """
    Grid Search (single shot validation).
    Takes a hyper paramater combination and trains 1 network which makes an ensemble of predictions.

    Variables:
    forecaster: the model to use.
    param_grid: hyperparameter grid.
    data: Training and validation data.
    time_vals: time series of training times and validation times.
    ensembles: number of ensembles for prediciton.
    n_prediciton: number of prediction time steps.
    n_train: number of trainning time steps.
    n_sync: number of synchronisation steps.

    outputs:
    MSE_params: (array: number_of_hyperparam_combination x number_of_ensemble) the MSE for each ensemble
                                                             member for each set of hyperparameters
    PH_params: (array: number_of_hyperparam_combination x number_of_ensemble) the PH for each ensemble
                                                             member for each set of hyperparameters
    """
    print(param_grid)
    synclen = n_sync
    trainlen = n_train - synclen
    predictionlen = n_prediction
    ts = data[0:trainlen,:]
    train_times = time_vals[0:trainlen]
    dt = time_vals[1] - time_vals[0]
    future_times = time_vals[trainlen:trainlen+synclen+predictionlen]
    sync_times = time_vals[trainlen:trainlen+synclen]
    prediction_times = time_vals[trainlen+synclen:trainlen+synclen+predictionlen]
    test_data = data[trainlen+synclen:trainlen+synclen+predictionlen, :]
    sync_data = data[trainlen:trainlen+synclen]
    future_data = data[trainlen:trainlen+synclen+predictionlen, :]
    variables = data.shape[-1]
    print('number of var =', variables)

    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())

    PH_params = np.zeros((num_combinations, ensembles))
    MSE_params = np.zeros((num_combinations, ensembles))
    #prediction_data = np.zeros((num_combinations, len(prediction_times), variables, ensembles))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        #print(param_dict)
        modelsync = forecaster(**param_dict)
        print(modelsync)
        # Now you can use `modelsync` with the current combination of parameters

        modelsync.fit(ts)

        ensemble_all_vals_ke = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals_q = np.zeros((len(prediction_times), ensembles))
        ensemble_all_vals = np.zeros((len(prediction_times), variables, ensembles))
        ensemble_all_vals_unscaled = np.zeros((len(prediction_times), variables, ensembles))
        MSE_ens = np.zeros((ensembles))
        PH_ens = np.zeros((ensembles))

        for i in range(ensembles):
            print('ensemble member =', i)
            #synchronise data
            modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
            #predict
            future_predictionsync = modelsync.predict(predictionlen)
            if i==0:
                print(np.shape(future_predictionsync))

            ### ensemble mean ###
            for v in range(variables):
                ensemble_all_vals_unscaled[:, v, i] = future_predictionsync[:,v]

            mse = MSE(ensemble_all_vals_unscaled[:,:,i], test_data[:,:])
            MSE_ens[i] = mse
            ph = prediction_horizon(ensemble_all_vals_unscaled[:,:,i], test_data[:,:], threshold = 0.2)
            PH_ens[i] = ph

        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:] = PH_ens
        #prediction_data[counter, :, :, :] = ensemble_all_vals[:, :, :]
        counter += 1
    return MSE_params, PH_params

def grid_search_WFV(forecaster, param_grid, data, time_vals, ensembles, splits, n_lyap, n_prediction_inLT, n_train_inLT, n_sync):
    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())
    
    # 5 folds, 10LTs each (7LT training, 3LT validation) 
    folds = splits #number of folds
    LT_train = n_train_inLT #number of LT in training set
    LT_validation = n_prediction_inLT #number of LT in validation set
    LT_set = LT_train+LT_validation #number of LT in training + validation sets
    len_data = n_lyap*folds*LT_set #number of time steps in all the data
    len_data_fold = LT_set*n_lyap #number of time steps in a fold
    LT_index = np.arange(0,folds)
    splits_index = LT_index*n_lyap #the beginning index of each fold
    print(splits_index)

    dt = time_vals[1] - time_vals[0]
    synclen = int(n_sync)
    trainlen = int(LT_train*n_lyap) - synclen
    predictionlen = int(LT_validation*n_lyap)
    print(trainlen, predictionlen)

    PH_params = np.zeros((num_combinations, ensembles * folds, 3))
    MSE_params = np.zeros((num_combinations, ensembles * folds))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        
        modelsync = forecaster(**param_dict)
        print(modelsync)

        MSE_ens = np.zeros((ensembles*folds))
        PH_ens = np.zeros((ensembles*folds, 3))
        
        for f in range(folds):
            start_index = int(splits_index[f])
            print(start_index)
            ts = data[start_index:start_index+trainlen,:]
            train_times = time_vals[start_index:start_index+trainlen]
            dt = time_vals[1] - time_vals[0]
            future_times = time_vals[start_index+trainlen:start_index+trainlen+synclen+predictionlen]
            sync_times = time_vals[start_index+trainlen:start_index+trainlen+synclen]
            prediction_times = time_vals[start_index+trainlen+synclen:start_index+trainlen+synclen+predictionlen]
            test_data = data[start_index+trainlen+synclen:start_index+trainlen+synclen+predictionlen, :]
            sync_data = data[start_index+trainlen:start_index+trainlen+synclen]
            future_data = data[start_index+trainlen:start_index+trainlen+synclen+predictionlen, :]
            variables = data.shape[-1]
            #print('number of var =', variables)
            print('start and end training times:', train_times[0], train_times[-1])
            
            
            modelsync.fit(ts)
  
            ensemble_all_vals_unscaled = np.zeros((len(prediction_times), variables, folds, ensembles))
            
            for i in range(ensembles):
                k = f*ensembles + i
                print('fold, ensemble member =', f, i)
                #synchronise data
                if synclen !=0:
                    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
                #predict
                future_predictionsync = modelsync.predict(predictionlen)
                
                for v in range(variables):
                    ensemble_all_vals_unscaled[:, v, f, i] = future_predictionsync[:,v]
                    
                mse = MSE(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:])
                MSE_ens[k] = mse
                PH_ens[k, 0] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 0.2)
                PH_ens[k, 1] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 0.5)
                PH_ens[k, 2] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 1.0)
                
        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:,:] = PH_ens
        counter += 1        
        
    return MSE_params, PH_params

'''
def grid_search_KFV(forecaster, param_grid, data, time_vals, ensembles, splits, n_lyap, n_prediction_inLT, n_train_inLT, n_sync):
    # Get the parameter labels and values
    param_labels = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Calculate the number of combinations
    num_combinations = prod(len(values) for values in param_grid.values())
    
    # 5 folds, 10LTs each (7LT training, 3LT validation) 
    folds = splits #number of folds
    LT_train = n_train_inLT #number of LT in training set
    LT_validation = n_prediction_inLT #number of LT in validation set
    LT_set = LT_train+LT_validation #number of LT in training + validation sets
    len_data = n_lyap*folds*LT_set #number of time steps in all the data
    len_data_fold = LT_set*n_lyap #number of time steps in a fold
    LT_index = np.arange(0,folds)
    splits_index = LT_index*n_lyap #the beginning index of each fold
    print(splits_index)

    dt = time_vals[1] - time_vals[0]
    synclen = int(n_sync)
    trainlen = int(LT_train*n_lyap) - synclen
    predictionlen = int(LT_validation*n_lyap)
    print(trainlen, predictionlen)

    PH_params = np.zeros((num_combinations, ensembles * folds, 3))
    MSE_params = np.zeros((num_combinations, ensembles * folds))

    counter = 0
    # Loop through all combinations
    for params in product(*param_values):
        param_dict = dict(zip(param_labels, params))
        
        modelsync = forecaster(**param_dict)
        print(modelsync)

        MSE_ens = np.zeros((ensembles*folds))
        PH_ens = np.zeros((ensembles*folds, 3))
        
        for f in range(folds):
            start_index = int(splits_index[f])
            print(start_index)
            ts = data[start_index:start_index+trainlen,:]
            train_times = time_vals[start_index:start_index+trainlen]
            dt = time_vals[1] - time_vals[0]
            future_times = time_vals[start_index+trainlen:start_index+trainlen+synclen+predictionlen]
            sync_times = time_vals[start_index+trainlen:start_index+trainlen+synclen]
            prediction_times = time_vals[start_index+trainlen+synclen:start_index+trainlen+synclen+predictionlen]
            test_data = data[start_index+trainlen+synclen:start_index+trainlen+synclen+predictionlen, :]
            sync_data = data[start_index+trainlen:start_index+trainlen+synclen]
            future_data = data[start_index+trainlen:start_index+trainlen+synclen+predictionlen, :]
            variables = data.shape[-1]
            #print('number of var =', variables)
            print('start and end training times:', train_times[0], train_times[-1])
            
            
            modelsync.fit(ts)
  
            ensemble_all_vals_unscaled = np.zeros((len(prediction_times), variables, folds, ensembles))
            
            for i in range(ensembles):
                k = f*ensembles + i
                print('fold, ensemble member =', f, i)
                #synchronise data
                if synclen !=0:
                    modelsync._update(sync_data, UpdateModes = UpdateModes.synchronization)
                #predict
                future_predictionsync = modelsync.predict(predictionlen)
                
                for v in range(variables):
                    ensemble_all_vals_unscaled[:, v, f, i] = future_predictionsync[:,v]
                    
                mse = MSE(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:])
                MSE_ens[k] = mse
                PH_ens[k, 0] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 0.2)
                PH_ens[k, 1] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 0.5)
                PH_ens[k, 2] = prediction_horizon(ensemble_all_vals_unscaled[:,:,f,i], test_data[:,:], threshold = 1.0)
                
        MSE_params[counter,:] = MSE_ens
        PH_params[counter,:,:] = PH_ens
        counter += 1        
        
    return MSE_params, PH_params
'''
