import numpy as np
from tqdm import tqdm

from base import BaseForecaster
from misc import correct_dimensions, identity
from weight_generators import standart_weights_generator, standart_weights_generator_withb
import matplotlib.pyplot as plt

from enum import Enum


UpdateModes = Enum('UpdateModes', 'synchronization transfer_learning refit')


ACTIVATIONS = {
    'identity': {
        'direct': identity,
        'inverse': identity,
    },
    'tanh': {
        'direct': np.tanh,
        'inverse': np.arctanh,  
    },
    'relu': {
        'direct': lambda x: np.maximum(0.,x),
        'inverse': lambda x: np.maximum(0.,x),
    },
    'leaky_relu': {
        'direct': lambda x: np.maximum(0.,x) + np.minimum(0.,x)*0.001,
        'inverse': lambda x: np.maximum(0.,x) + np.minimum(0.,x)*0.001,
    },
    'flat_relu': {
        'direct': lambda x: np.minimum(np.maximum(0.,x), 1.),
        'inverse': lambda x: np.minimum(np.maximum(0.,x), 1.),
    },
    'gauss': {
        'direct': lambda x: np.exp(-x**2/2),
        'inverse': lambda x: -(np.log(x) * 2)**0.5,
    },
    'rel_gauss': {
        'direct': lambda x: np.maximum(np.minimum(x/2+0.5, 0.5),-0.5)+np.maximum(np.minimum(-x/2+0.5, 0.5),-0.5),
        'inverse': lambda x: -2*x+2
    },
    'sin': {
        'direct': np.sin,
        'inverse': np.arcsin
    },
}

class EsnForecaster(BaseForecaster):
    """Echo State Network time-forecaster.

    Parameters
    ----------
    spectral_radius : float
        Spectral radius of the recurrent weight matrix W
    sparsity : float
        Proportion of elements of the weight matrix W set to zero
    regularization : {'noise', 'l2', None}
        Type of regularization
    lambda_r : float
        Regularization parameter value which will be multiplied
        with the regularization term
    noise : float
        Noise stregth; added to each reservoir state element for regularization
    in_activation : {'identity', 'tanh'}
        Input activation function (applied to the linear
        combination of input, reservoir and control states)
    out_activation : {'identity', 'tanh'}
        Output activation function (applied to the readout)
    use_additive_noise_when_forecasting : bool
        Whether additive noise (same as used for regularization) is added
        before activation in the forecasting mode
    inverse_out_activation : function
        Inverse of the output activation function
    random_state : positive integer seed, np.rand.RandomState object,
                   or None to use numpy's builting RandomState.
        Used as a seed to randomize matrices W_in, W and W_c
    use_bias : bool
        Whether a bias term is added before activation
    """
    def __init__(self,
                 n_reservoir=200,
                 spectral_radius=0.95,
                 sparsity=0,
                 regularization='noise',
                 lambda_r=0.001,
                 in_activation='tanh',
                 out_activation='identity',
                 use_additive_noise_when_forecasting=True,
                 random_state=None,
                 use_b=False,
                 use_bias=True,
                 use_r_bias=False,
                 input_scaling=1.00,
                 n_washout=0):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.regularization = regularization
        self.lambda_r = lambda_r
        self.use_additive_noise_when_forecasting = \
            use_additive_noise_when_forecasting
        #if self.regularization == 'l2':
         #   self.use_additive_noise_when_forecasting = False
        self.in_activation = in_activation
        self.out_activation = out_activation
        self.random_state = random_state
        self.use_bias = use_bias
        self.use_b = use_b
        self.use_r_bias = use_r_bias
        self.input_scaling = input_scaling
        self.n_washout = n_washout
        
        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state is not None:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        super().__init__()

    def get_fitted_params(self):
        """Get fitted parameters. Overloaded method from BaseForecaster

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return {
            'W_in': self.W_in_,
            'W': self.W_,
            'W_c': self.W_c_,
            'W_out': self.W_out_,
        }

    def _fit(self, y, X=None,
             initialization_strategy=standart_weights_generator_withb,
             inspect=False):
        """Fit forecaster to training data. Overloaded method from
        BaseForecaster.
        Generates random recurrent matrix weights and fits the readout weights
        to the available time series (endogeneous time series).
        After that the function of calculating the optimal matrix W_out is called.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : array-like, shape (batch_size x n_timesteps x n_inputs)
            Time series to which to fit the forecaster
            or a sequence of them (batches).
        X : array-like or function, shape (batch_size x n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to fit to or a sequence of them (batches).
            Can also be understood as a control signal
        initialization_strategy: function
            A function generating random matrices W_in, W and W_c
        inspect : bool
            Whether to show a visualisation of the collected reservoir states

        Returns
        -------
        self : returns an instance of self.
        """
        print('fitting now')
        print(self.use_additive_noise_when_forecasting)
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='3D')
        self.W_in_, self.W_, self.W_c_, self.b = \
            initialization_strategy(self.random_state_,
                                    self.n_reservoir,
                                    self.sparsity,
                                    self.spectral_radius,
                                    endo_states,
                                    exo_states=exo_states,
                                    input_scaling=self.input_scaling)
        return self._update_via_refit(endo_states, exo_states, inspect)

    def _predict(self, n_timesteps, X=None, inspect=False):
        """Forecast time series at further time steps.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        n_timesteps : int
            Forecasting horizon
        X : array-like or function, shape (n_timesteps x n_controls), optional
            Exogeneous time series to fit to.
            Can also be understood as a control signal

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh

        Returns:
            Array of output activations
        """
        if X is None or X[0] is None:  # TODO: some sklearn subroutines like gridsearchcv
            # cannot pass X=None and pass array of None. That's why we check X[0] here
            exo_states = None
        elif isinstance(X, np.ndarray):
            exo_states = correct_dimensions(X)
        if n_timesteps is None:  # TODO: this is actually a stupid bypass because
            # some sklearn subroutines like gridsearchcv cannot pass any kwargs to predict()
            if X is not None:
                n_timesteps = len(X)
            else:
                raise ValueError('No way to deduce the number of time steps: both n_timesteps and X are set to None')
        n_endo = self.last_endo_state_.shape[0]
        n_reservoir = self.last_reservoir_state_.shape[0]
        means = np.zeros((n_timesteps))
        std_devs = np.zeros((n_timesteps))

        endo_states = np.vstack([self.last_endo_state_,
                                 np.zeros((n_timesteps, n_endo))])
        reservoir_states = np.vstack([self.last_reservoir_state_,
                                      np.zeros((n_timesteps, n_reservoir))])
        if self.use_r_bias:
                ones = np.ones(1)
                #print(np.shape(self.last_reservoir_state_))
                #print(np.shape(ones))
                last_res_state_bias = np.concatenate((self.last_reservoir_state_, ones))
                reservoir_states_bias = np.vstack([last_res_state_bias,
                                      np.zeros((n_timesteps, n_reservoir+1))])
                reservoir_states_bias[:, -1] = 1
                #print('res_states_bias shape', np.shape(reservoir_states_bias))
        
        print('prediction..')
        print('endo states shape', np.shape(endo_states))
        #print('res states shape', np.shape(reservoir_states))
        if inspect:
            print("predict...")
            pbar = tqdm(total=n_timesteps, position=0, leave=True)
        for n in range(n_timesteps):
            reservoir_state = reservoir_states[n, :]
            endo_state = endo_states[n, :]
            exo_state = None
            means[n] = np.mean(reservoir_state)
            std_devs[n] = np.std(reservoir_state)
            if exo_states is None:
                exo_state = None
            elif isinstance(exo_states, np.ndarray):
                exo_state = exo_states[n, :]
            else:  # exo_states is assumed to be a callable
                exo_state = exo_states(n, endo_state)
            reservoir_states[n + 1, :] = \
                self._iterate_reservoir_state(reservoir_state,
                                              endo_state,
                                              exo_state)
            if self.use_r_bias:
                reservoir_states_bias[n+1, :-1] = reservoir_states[n+1, :]

                endo_states[n + 1, :] = \
                ACTIVATIONS[self.out_activation]['direct'](np.dot(self.W_out_,
                                                                 reservoir_states_bias[n + 1, :]))
            else:
                endo_states[n + 1, :] = \
                ACTIVATIONS[self.out_activation]['direct'](np.dot(self.W_out_,
                                                                  reservoir_states[n + 1, :]))
            if self.use_bias:
                endo_states[n + 1, 0] = 1  # bias changed from 1
            if inspect:
                pbar.update(1)
            #print(endo_states[n+1,:])
        
        if inspect:
            pbar.close()
            fig, ax = plt.subplots(1, 2, figsize=(10,10))
            ax[0].plot(np.arange(0, n_timesteps), means)
            ax[1].plot(np.arange(0, n_timesteps), std_devs)
            plt.show()
        if self.use_bias:
            return endo_states[1:, 1:]
        else:
            return endo_states[1:]

    def _update(self, y, X=None,
                mode: UpdateModes = UpdateModes.synchronization,
                **kwargs):
        """Update the model to incremental training data.
        Depending on the mode, it can be done as synchronization
        or transfer learning.

        Writes to self:
            If mode == 'synchronization'
                updates last_reservoir_state_ and last_endo_state_
            If mode == 'transfer_learning'
                updates W_out_, last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to update the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to update to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        if mode == UpdateModes.synchronization:
            return self._update_via_synchronization(y, X)
        elif mode == UpdateModes.transfer_learning:
            return self._update_via_transfer_learning(y, X, **kwargs)
        elif mode == UpdateModes.refit:
            endo_states, exo_states = \
                self._treat_dimensions_and_bias(y, X, representation='3D')
            return self._update_via_refit(endo_states, exo_states, **kwargs)

    def _update_via_refit(self, endo_states, exo_states=None, inspect=False):
        """Refit forecaster to training data. 
        The model can be fitted based on a single time series (batch_size == 1)
        or a sequence of disconnected time series (batch_size > 1).
        Optionally, a multivariate control signal (exogeneous time series)
        can be passed which we will also be included into fitting process.
        Note that in this case, the control signal must also be passed during
        prediction.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        endo_states : array-like, shape (batch_size x n_timesteps x n_inputs)
            Time series to which to fit the forecaster
            or a sequence of them (batches).
        exo_states : array-like or function, shape (batch_size x n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to fit to or a sequence of them (batches).
            Can also be understood as a control signal
        inspect : bool
            Whether to show a visualisation of the collected reservoir states

        Returns
        -------
        self : returns an instance of self.
        """
        n_batches = endo_states.shape[0]
        n_timesteps = endo_states.shape[1]
        n_washout = self.n_washout
        reservoir_states = np.random.rand(n_batches, n_timesteps, self.n_reservoir) * 2 -1 ###changed from 0
        means_train = np.zeros((n_batches*n_timesteps))
        std_devs_train = np.zeros((n_batches*n_timesteps))

        if inspect:
            print("fitting...")
            pbar = tqdm(total=n_batches*n_timesteps,
                        position=0,
                        leave=True)

        for b in range(n_batches):
            for n in range(1, n_timesteps):
                if exo_states is None:
                    reservoir_states[b, n, :] = \
                        self._iterate_reservoir_state(reservoir_states[b, n - 1],
                                                      endo_states[b, n - 1, :])
                else:
                    reservoir_states[b, n, :] = \
                        self._iterate_reservoir_state(reservoir_states[b, n - 1,],
                                                      endo_states[b, n - 1, :],
                                                      exo_states[b, n, :])
                means_train[n] = np.mean(reservoir_states[b,n,:])
                std_devs_train[n] = np.std(reservoir_states[b,n,:])
                if inspect:
                    pbar.update(1)
        if inspect:
            pbar.close()
            

        reservoir_states = np.reshape(reservoir_states,
                                      (-1, reservoir_states.shape[-1]))
        endo_states = np.reshape(endo_states,
                                 (-1, endo_states.shape[-1]))
        if self.use_r_bias:
            # Create a column of ones
            ones_column = np.ones((n_timesteps, 1))
            reservoir_states = np.concatenate((reservoir_states, ones_column), axis=1)
        print('fitting')
        print('shape endo', np.shape(endo_states))
        print('shape res', np.shape(reservoir_states))
        if inspect:
            print("solving...")
        if self.regularization == 'l2':
            if self.use_r_bias:
                idenmat = self.lambda_r * np.identity(self.n_reservoir+1)
            else:
                idenmat = self.lambda_r * np.identity(self.n_reservoir)
            print('shape identity matrix', np.shape(idenmat))
            print('shape res', np.shape(reservoir_states))
            U = np.dot(reservoir_states[n_washout:, :].T, reservoir_states[n_washout:, :]) + idenmat
            self.W_out_ = np.linalg.solve(U, reservoir_states[n_washout:, :].T @ endo_states[n_washout:, :]).T
        elif self.regularization == 'noise' or self.regularization is None:
            # same formulas as above but with lambda = 0
            U = np.dot(reservoir_states.T, reservoir_states)
            self.W_out_ = np.linalg.solve(U, reservoir_states.T @ endo_states).T
        else:
            raise ValueError(f'Unknown regularization: {self.regularization}')
        # remember the last state for later:
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_state_ = endo_states[-1, :]
        if inspect:
            fig1, ax1 = plt.subplots(1,2, figsize=(10,10))
            ax1[0].plot(np.arange(n_washout, n_timesteps), means_train[n_washout:])
            ax1[1].plot(np.arange(n_washout, n_timesteps), std_devs_train[n_washout:])
            plt.show()
        if exo_states is None:
            self.last_exo_state_ = 0
        else:
            raise NotImplementedError('forgot to implement')
        return self

    def _update_via_synchronization(self, y, X=None):
        """Update the model to incremental training data
        by synchnorizing the reservoir state with the
        given time series.

        Writes to self:
            updates last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to synchronize the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to synchronize to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='2D')
        n_timesteps = y.shape[0]
        reservoir_states = np.random.rand(n_timesteps, self.n_reservoir)* 2 -1
        for n in range(1, n_timesteps):
            if X is None:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1, :],
                                                  endo_states[n - 1, :])
            else:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1, :],
                                                  endo_states[n - 1, :],
                                                  exo_states[n - 1, :])
                                                 
                                                  
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_state_ = endo_states[-1, :]
        return self

    def _update_via_transfer_learning(self, y, X=None, mu=1e-8, inspect=False):
        """Update the model to incremental training data using transfer
        learning.

        Writes to self:
            updates W_out_, last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to synchronize the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to synchronize to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='2D')
        n_timesteps = endo_states.shape[0]

        # step the reservoir through the given input,output pairs:
        reservoir_states = np.zeros((n_timesteps, self.n_reservoir))

        if inspect:
            print("transfer...")
            pbar = tqdm(total=n_timesteps - 1,
                        position=0,
                        leave=True)

        for n in range(1, n_timesteps):
            if(exo_states is None):
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1],
                                                  endo_states[n - 1, :])
            else:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1],
                                                  endo_states[n - 1, :],
                                                  exo_states[n, :])
            if inspect:
                pbar.update(1)
        if inspect:
            pbar.close()

        identmat = mu * np.identity(self.n_reservoir)
        if inspect:
            print("solving...")
        R = reservoir_states.T @ reservoir_states
        dW = np.linalg.solve(R + identmat, reservoir_states.T @ endo_states - (self.W_out_ @ R).T).T
        self.W_out_ += dW

        # remember the last state for later:
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_state_ = endo_states[-1, :]
        return self

    def _iterate_reservoir_state(self, reservoir_state, endo_state,
                                 exo_state=None,
                                 forecasting_mode=True):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and
        output patterns.
        """
        n_reservoir = reservoir_state.shape[0]
        preactivation = np.dot(self.W_, reservoir_state) + np.dot(self.W_in_,
                                                                  endo_state) #+(self.random_state_.rand(n_reservoir) - 0.5)
        if exo_state is not None:
            preactivation += np.dot(self.W_c_, exo_state)
        if self.use_b:
            preactivation += self.b
        s = ACTIVATIONS[self.in_activation]['direct'](preactivation)
        if (forecasting_mode and self.use_additive_noise_when_forecasting) or \
           (not forecasting_mode and self.regularization == 'noise'):
            s += self.lambda_r * (self.random_state_.rand(n_reservoir) - 0.5)
        return s

    def _treat_dimensions_and_bias(self, y, X=None, representation='2D'):
        """Transform array shapes following the specified representation
        and add the bias term if necessary.

        Parameters
        ----------
        y : array-like
            Time series of endogeneous states.
        X : array-like or function, optional (default=None)
            Time series of exogeneous states.
            Can also be understood as a control signal

        Returns
        -------
        endo_states, exo_states
        """
        y = correct_dimensions(y, representation=representation)
        if y is None:
            raise ValueError(f'Inconsistent combination of y shape '
                             f'and {representation} representation')
        if X is None or X[0] is None:
            X = None
        elif isinstance(X, np.ndarray):
            X = correct_dimensions(X, representation=representation)
        exo_states = X
        endo_states = y
        print('endo states before bias', np.shape(endo_states))
        if(self.use_bias):
            ones_shape = None
            if representation == '2D':
                ones_shape = (endo_states.shape[0], 1)
            elif representation == '3D':
                ones_shape = (endo_states.shape[0], endo_states.shape[1], 1)
            else:
                raise ValueError(f'Unsupported representation: '
                                 f'{representation}')
            endo_states = np.concatenate((np.ones(ones_shape), endo_states),
                                         axis=-1)
        print("shape of endo states after bias", np.shape(endo_states))
        return endo_states, exo_states

