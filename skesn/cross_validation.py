import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_squared_error, make_scorer
import pandas as pd

from base import BaseForecaster
from misc import SklearnWrapperForForecaster


from enum import Enum
UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')

class ValidationBasedOnRollingForecastingOrigin:
    """We fix the test size (this defines multi-step forecasting) and gradually increase training size.
    Optionally, we can fix training size too.
    Optionally, we can fix the training time series and validate only against rolling forecasting origin.

    See https://otexts.com/fpp3/tscv.html
    """
    
    def __init__(self, metric=mean_absolute_percentage_error,
                 n_training_timesteps=None,
                 n_test_timesteps=10,
                 n_splits=10,
                 overlap=0,
                 sync = False,
                 n_sync_timesteps = 10):
        self.metric = metric
        self.n_training_timesteps = n_training_timesteps
        self.n_test_timesteps = n_test_timesteps
        self.n_splits = n_splits
        self.overlap = overlap
        self.sync = sync
        self.n_sync_timesteps = n_sync_timesteps

    def evaluate(self, forecaster, y, X):
        return [self.metric(y_true, y_pred)
                for _, y_pred, y_true in self.prediction_generator(forecaster, y, X)]

    def prediction_generator(self, forecaster, y, X):
        ts_cv = TimeSeriesSplit(n_splits=self.n_splits,
                                test_size=self.n_test_timesteps,
                                max_train_size=self.n_training_timesteps)
        for train_index, test_index in ts_cv.split(y):
            print(f"  Train: index={train_index[0]} - {train_index[-1]}")
            print(f"  Test:  index={test_index[0]} - {test_index[-1]}")
            if X is None:
                if self.sync:
                    forecaster.fit(y=y[train_index[:-10]],
                               X=None)
                    sync_y = y[train_index[-10:]]
                    forecaster.update(sync_y, UpdateModes = UpdateModes.synchronization)
                else:
                    forecaster.fit(y=y[train_index],
                               X=None)
                y_pred = forecaster.predict(self.n_test_timesteps,
                                            X=None)
            else:
                forecaster.fit(y=y[train_index], X=X[train_index])
                y_pred = forecaster.predict(self.n_test_timesteps,
                                            X=X[test_index])
            yield test_index, y_pred, y[test_index]
            
    def prediction_generator_overlap(self, forecaster, y, X):
        ts_cv = TimeSeriesSplit(n_splits=self.n_splits,
                                test_size=self.n_test_timesteps,
                                max_train_size=self.n_training_timesteps)
        ove = self.overlap
        values=np.arange(0, self.n_test_timesteps, ove)
        print('values=', values)
        for j, (train_index, test_index) in enumerate(ts_cv.split(y)):
            print('j=', j)
            train_start = train_index[0]
            train_end = train_index[-1] + 1   # +1 because Python slicing is exclusive
            test_start = test_index[0]
            test_end = test_index[-1] + 1  # +1 because Python slicing is exclusive
            for i in values:
                if j != self.n_splits-1:
                    print('i=', i)
                    print(f"  Training: index={train_start+i}-{train_end+i}")
                    print(f"  Prediction: index={train_end+i}-{test_end+i}")
                    test_index_alt = test_index + i
                    if X is None:
                        forecaster.fit(y=y[train_start+i:train_end+i],
                                       X=None)
                        y_pred = forecaster.predict(self.n_test_timesteps,
                                                    X=None)
                    else:
                        forecaster.fit(y=y[train_index], X=X[train_index])
                        y_pred = forecaster.predict(self.n_test_timesteps,
                                                    X=X[test_index])
                    yield test_index+i, y_pred, y[test_index_alt]
                else:
                    if i == 0:
                        print('i=', i)
                        print(f"  Training: index={train_start+i}-{train_end+i}")
                        print(f"  Prediction: index={train_end+i}-{test_end+i}")
                        test_index_alt = test_index + i
                        if X is None:
                            forecaster.fit(y=y[train_start+i:train_end+i],
                                           X=None)
                            y_pred = forecaster.predict(self.n_test_timesteps,
                                                        X=None)
                        else:
                            forecaster.fit(y=y[train_index], X=X[train_index])
                            y_pred = forecaster.predict(self.n_test_timesteps,
                                                        X=X[test_index])
                        yield test_index+i, y_pred, y[test_index_alt]
                


    def grid_search(self, forecaster, param_grid, y, X):
        """
        For error metrics (they are assumed here), score values will be negative even for
        non-negative metrics. You need to compute absolute values of the scores to get
        the expected values.
        """
        if X is None: # TODO: some sklearn subroutines like gridsearchcv
            # cannot pass X=None so we need to pass array of None
            X = [None for _ in range(len(y))]
        if issubclass(type(forecaster), BaseForecaster):
            forecaster = SklearnWrapperForForecaster(forecaster)
            param_grid = {f'custom_estimator__{k}': v for k, v in param_grid.items()}
        scorer = make_scorer(self.metric, greater_is_better=False)
        grid = GridSearchCV(forecaster, param_grid,
                            scoring=scorer,
                            cv=TimeSeriesSplit(n_splits=self.n_splits,
                                               test_size=self.n_test_timesteps,
                                               max_train_size=self.n_training_timesteps))
        grid.fit(X=X, y=y)
        return grid.cv_results_, grid.best_estimator_

    def time_delay(self, y_true, y_pred):
        cross_corr = np.correlate(y_true, y_pred, mode='full')
        lag_max_corr = np.argmax(cross_corr) - len(y_true) + 1
        return lag_max_corr
    
    def time_delay_loss(forecaster, y_pred, y_true):
        cross_corr = np.correlate(y_true, y_pred, mode='full')
        lag_max_corr = np.argmax(cross_corr) - len(y_true) + 1
        abs_delay = np.abs(lag_max_corr)
        return abs_delay