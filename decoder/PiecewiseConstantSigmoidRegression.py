import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PiecewiseConstantRegression:
    """
    Piecewise Constant Regression.

    This class performs piecewise constant regression, which fits constant
    functions on each segment of the data.

    Parameters
    ----------
    num_pieces : int
        The number of constant segments in the piecewise regression model.
    lam : float, optional (default=0.1)
        The regularization strength (lambda) for L1 regularization. Higher values of
        'lam' encourage sparsity, resulting in fewer segments.

    Attributes
    ----------
    num_pieces : int
        The number of constant segments in the piecewise regression model.
    lam : float
        The regularization strength (lambda) for L1 regularization.
    breakpoints : numpy.ndarray, shape (num_pieces - 1,)
        The breakpoints that divide the data into segments for the piecewise model.
    constants : numpy.ndarray, shape (num_pieces,)
        The constant values for each segment in the piecewise model.

    Methods
    -------
    fit(x, y)
        Fit the piecewise constant regression model to the provided data.
    predict(x)
        Predict the output for the given input 'x' using the fitted model.
    visualize(x, y, new_x=None)
        Visualize the piecewise constant regression along with the data.
    """

    def __init__(self, num_pieces, lam=0.1):
        self.num_pieces = num_pieces
        self.lam = lam
        self.breakpoints = None
        self.constants = None

    def _total_loss(self, params, x, y):
        """
        Calculate the total loss function (MSE + L1 regularization).

        Parameters
        ----------
        params : numpy.ndarray
            Concatenation of breakpoints and constants.
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output (target) data.

        Returns
        -------
        float
            Total loss function value.
        """
        self.breakpoints = np.sort(params[:self.num_pieces - 1])
        self.constants = params[self.num_pieces - 1:]

        piecewise_y = self._piecewise_predict_for_minimize(x)
        return self._mse_loss(piecewise_y, y) + self._l1_regularization()

    def metric(self, x, y, piecewise_y = None):
        if piecewise_y is None:
            piecewise_y = self._piecewise_predict(x)
        
        return self._mse_loss(piecewise_y, y) + self._l1_regularization()

    def _mse_loss(self, y_pred, y):
        """
        Calculate the mean squared error loss for the piecewise model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output (target) data.

        Returns
        -------
        float
            Mean squared error loss.
        """
        return np.mean((y_pred - y) ** 2)

    def _l1_regularization(self):
        """
        Calculate the L1 regularization term on breakpoints.

        Returns
        -------
        float
            L1 regularization term.
        """
        return self.lam * self.num_pieces**2

    def _piecewise_predict_for_minimize(self, x):
        """
        Predict the output for the given input 'x' for minimizing total losses. 
        It has mild differences with _piecewise_predict.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        if self.constants is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        temp_breakpoints = np.concatenate([[np.min(x)], self.breakpoints, [np.max(x)]])
        piece_indices = np.searchsorted(temp_breakpoints, x, side='right') - 1
        piece_indices = np.clip(piece_indices, 0, len(self.constants) - 1)
        return self.constants[piece_indices]

    def _piecewise_predict(self, x):
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        if self.constants is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")
        piece_indices = np.searchsorted(self.breakpoints, x, side='right') -1
        piece_indices = np.clip(piece_indices, 0, len(self.constants) - 1)
        return self.constants[piece_indices]

    def fit(self, x, y):
        """
        Fit the piecewise constant regression model to the provided data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output (target) data.
        """
        initial_breakpoints = np.linspace(np.min(x), np.max(x), self.num_pieces + 1)[1:-1] + (np.random.rand(self.num_pieces-1)-0.5)*0.05*(np.max(x)-np.min(x))
        initial_constants = np.mean(y) * np.ones(self.num_pieces) 
        initial_params = np.concatenate([initial_breakpoints, initial_constants])

        bounds = [(np.min(x), np.max(x))] * (self.num_pieces - 1) + [(np.min(y), np.max(y))] * self.num_pieces

        # Optimization using Powell algorithm with callback for ordering constraint
        result = minimize(self._total_loss, initial_params, args=(x, y), bounds=bounds)

        self.breakpoints = np.concatenate([[np.min(x)], np.sort(result.x[:self.num_pieces - 1]), [np.max(x)]])
        self.constants = result.x[self.num_pieces - 1:]
        piecewise_y = self._piecewise_predict(x)
        self.total_loss = self._mse_loss(piecewise_y, y) + self._l1_regularization()

    def predict(self, x):
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        return self._piecewise_predict(x)

    def visualize(self, x, y, new_x=None):
        """
        Visualize the piecewise constant regression along with the data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output (target) data.
        new_x : numpy.ndarray, optional
            New input data for which to visualize the predictions, by default None.
        """
        if self.breakpoints is None or self.constants is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        if new_x is None:
            new_x = np.linspace(np.min(x), np.max(x), 1000)

        plt.scatter(x, y, label='Data', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Piecewise Constant Regression')
        plt.grid(True)

        piecewise_y = self._piecewise_predict(new_x)

        plt.plot(new_x, piecewise_y, label='Piecewise Constant Regression', color='red')
        plt.legend()
        plt.show()

from numba import jit
import time

@jit(nopython=True)
def _loss(x, y, breakpoint: float, constants: list):
    y_pred = np.where(x<=breakpoint, constants[0], constants[1])
    return np.sum((y - y_pred)**2)

@jit(nopython=True)
def _twopiece_fit(init_breakpoints, x, y):
    total_losses = np.zeros_like(init_breakpoints, dtype=np.float64)
    for i, b in enumerate(init_breakpoints):
        total_losses[i] = _loss(x, y, b, (np.nanmean(y[np.where(x<=b)[0]]), np.nanmean(y[np.where(x>b)[0]])))
    return total_losses

class TwoPiecesPiecewiseSigmoidRegression:
    def __init__(self) -> None:
        pass
        self.breakpoints = None
        self.constants = None

        self.L, self.k, self.x0, self.b = None, None, None, None

    def _piecewise_predict(self, x):
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        if self.constants is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        return np.where(x<=self.breakpoints, self.constants[0], self.constants[1])

    @staticmethod
    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def _sigmoid(self, x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def _sigmoid_predict(self, x:np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        if self.constants is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        return self._sigmoid(x, self.L, self.k, self.x0, self.b)

    def _loss(self, x, y):
        y_pred = self._piecewise_predict(x)
        return np.sum((y - y_pred)**2)

    def fit(self, x: np.ndarray, y: np.ndarray, k: float) -> None:
        uniq_x = np.unique(x)
        dx = np.ediff1d(uniq_x)
        break_point_candidates = uniq_x[:-1] + dx/2
        
        
        """        
        total_losses = np.zeros_like(break_point_candidates, dtype=np.float64)

        for i, b in enumerate(break_point_candidates):
            self.breakpoints = b
            self.constants = [np.nanmean(y[np.where(x<=b)[0]]), np.nanmean(y[np.where(x>b)[0]])]
            total_losses[i] = self._loss(x, y)
        """
        total_losses = _twopiece_fit(break_point_candidates, x, y)
        best_idx = np.nanargmin(total_losses)

        b = break_point_candidates[best_idx]
        self.breakpoints = b
        self.constants = [np.nanmean(y[np.where(x<=b)[0]]), np.nanmean(y[np.where(x>b)[0]])]

        if self.constants[0] != self.constants[1]:
            self.L = np.abs(self.constants[0] - self.constants[1])
            self.k = k if self.constants[0] < self.constants[1] else -k
            self.x0 = b
            self.b = min(self.constants[0], self.constants[1])  
        else:
            self.L = 0
            self.k = 0
            self.x0 = np.median(x)
            self.b = self.constants[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid_predict(x)


class PiecewiseConstantSigmoidRegression:
    def __init__(self, num_pieces, lam=0.1):
        self.num_pieces = num_pieces
        self.lam = lam
        self.x0 = None
        self.breakpoints = None

    @staticmethod
    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def _sigmoid(self, x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def _piecewise_predict(self, x):
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        if self.breakpoints is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        piece_indices = np.searchsorted(self.breakpoints, x, side='right') - 1
        piece_indices = np.clip(piece_indices, 0, self.L.shape[0] - 1)

        L = self.L[piece_indices]
        k = self.k[piece_indices]
        x0 = self.x0[piece_indices]
        b = self.b[piece_indices]

        return self._sigmoid(x, L, k, x0, b)

    def fit(self, x, y, k):
        model = PiecewiseConstantRegression(self.num_pieces, self.lam)
        model.fit(x, y)

        if self.num_pieces != 1:
            self.x0 = model.breakpoints[1:-1]
            self.total_loss = model.total_loss
            self.breakpoints = np.concatenate([[x[0]], np.ediff1d(self.x0)/2 + self.x0[:-1], [x[-1]]])
            self.L, self.k, self.b = np.zeros_like(self.x0, dtype = np.float64), np.zeros_like(self.x0, dtype = np.float64), np.zeros_like(self.x0, dtype = np.float64)
            for i in range(self.x0.shape[0]):
                if model.constants[i] >= model.constants[i+1]:
                    self.L[i], self.b[i] = model.constants[i]-model.constants[i+1], model.constants[i+1]
                    self.k[i] = -np.abs(k)
                else:
                    self.L[i], self.b[i] = model.constants[i+1]-model.constants[i], model.constants[i]
                    self.k[i] = np.abs(k)
        else:
            self.x0 = np.array([0], dtype=np.float64)
            self.total_loss = model.total_loss
            self.breakpoints = np.array([np.min(x), np.max(x)])
            self.L = np.zeros(1)
            self.k = np.zeros(1)
            self.b = np.array([model.constants[0]])
        
    def predict(self, x):
        """
        Predict the output for the given input 'x' using the fitted model.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Predicted output values.
        """
        return self._piecewise_predict(x)

    def visualize(self, x, y, new_x=None):
        """
        Visualize the piecewise constant regression along with the data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Output (target) data.
        new_x : numpy.ndarray, optional
            New input data for which to visualize the predictions, by default None.
        """
        if self.breakpoints is None:
            raise ValueError("Model not fitted. Call 'fit' method first.")

        if new_x is None:
            new_x = np.linspace(np.min(x), np.max(x), 1000)

        plt.scatter(x, y, label='Data', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Piecewise Constant Regression')
        plt.grid(True)

        piecewise_y = self._piecewise_predict(new_x)

        plt.plot(new_x, piecewise_y, label='Piecewise Constant Regression', color='red')
        plt.legend()
        plt.show()

class PiecewiseRegressionModel:
    """
    Automated testing and visualization of Piecewise Constant Regression.

    This class automates the process of testing and visualizing the results of the
    Piecewise Constant Regression for different values of 'num_pieces'.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Output (target) data.
    num_pieces_range : list or tuple
        A range of 'num_pieces' values to test for the regression model.
    lam : float, optional (default=0.1)
        The regularization strength (lambda) for L1 regularization in the regression model.

    Attributes
    ----------
    x : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Output (target) data.
    num_pieces_range : list or tuple
        A range of 'num_pieces' values to test for the regression model.
    lam : float
        The regularization strength (lambda) for L1 regularization in the regression model.
    models : dict
        A dictionary storing the fitted PiecewiseConstantRegression models for
        each 'num_pieces' value as keys.
    best_model : PiecewiseConstantRegression
        The best model with the lowest total loss.
    best_num_pieces : int
        The optimal 'num_pieces' value that resulted in the lowest total loss.

    Methods
    -------
    test()
        Test the Piecewise Constant Regression for different 'num_pieces' values.

    visualize_results(new_x=None)
        Visualize the results of the Piecewise Constant Regression for each 'num_pieces'
        value along with the data points and predictions for new data points.
    """

    def __init__(self, x, y, num_pieces_range=range(1, 4), lam=0.1, k_default=1):
        self.x = x
        self.y = y
        self.num_pieces_range = num_pieces_range
        self.lam = lam
        self.k = k_default
        self.models = {}
        self.best_model = None
        self.best_num_pieces = None

    def fit(self):
        """
        Test the Piecewise Constant Regression for different 'num_pieces' values.

        This method fits the PiecewiseConstantRegression model for each 'num_pieces'
        value specified in 'num_pieces_range'. The fitted models are stored in the 'models'
        attribute as a dictionary with 'num_pieces' values as keys. The method also selects
        the model with the lowest total loss and stores it as the 'best_model' along with
        the optimal 'num_pieces' value in 'best_num_pieces'.

        Raises
        ------
        ValueError
            If 'x' and 'y' have different lengths or 'num_pieces_range' is empty.
        """
        if len(self.x) != len(self.y):
            raise ValueError("The lengths of 'x' and 'y' must be the same.")
        if not self.num_pieces_range:
            raise ValueError("'num_pieces_range' cannot be empty.")

        for num_pieces in self.num_pieces_range:
            model = PiecewiseConstantSigmoidRegression(num_pieces, self.lam)
            model.fit(self.x, self.y, self.k)
            self.models[num_pieces] = model

        # Select the model with the lowest total loss as the best model
        self.best_model = min(self.models.values(), key=lambda model: model.total_loss)
        #print('losses:',[self.models[k].total_loss for k in self.models.keys()])
        self.best_num_pieces = self.best_model.num_pieces

    def visualize_results(self, new_x=None):
        """
        Visualize the results of the Piecewise Constant Regression for each 'num_pieces'
        value along with the data points and predictions for new data points.

        Parameters
        ----------
        new_x : numpy.ndarray, optional
            New input data for which to visualize the predictions, by default None.

        Raises
        ------
        ValueError
            If no models are fitted. Call 'test' method first.
        """
        if not self.models:
            raise ValueError("No models fitted. Call 'test' method first.")

        if new_x is None:
            new_x = np.linspace(np.min(self.x), np.max(self.x), 1000)

        plt.scatter(self.x, self.y, label='Data', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Piecewise Constant Regression')
        plt.grid(True)

        piecewise_y = self.best_model.predict(new_x)

        plt.plot(new_x, piecewise_y, label=f'Num Pieces: {self.best_num_pieces}')
        plt.xlim([1,10])
        plt.xticks(np.linspace(1,10,10))
        plt.legend()
        plt.show()



"""PiecewiseConstantRegression
# Example usage and visualization:
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 3, 5, 7, 8, 10, 12, 14, 15, 17])

num_pieces = 3
lam = 0.1

regressor = PiecewiseConstantRegression(num_pieces, lam)
regressor.fit(x, y)

# Visualize the piecewise constant sigmoid regression
regressor.visualize(x, y)

# Optionally, provide new_x for predictions and visualize them
new_x = np.array([11, 12, 13, 14])
regressor.visualize(x, y, new_x=new_x)
"""
if __name__ == '__main__':
    # Example usage and visualization with multiple num_pieces values:
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 3, 5, 7, 8, 9, 12, 1, 5, 2])

    num_pieces_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Test for different num_pieces values
    lam = 4

    regression_tester = PiecewiseRegressionTester(x, y, num_pieces_range, lam=4, k_default=50)
    regression_tester.test()
    regression_tester.visualize_results()
