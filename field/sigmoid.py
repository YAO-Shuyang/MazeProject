from scipy.optimize import curve_fit
import numpy as np

def init_guess(x, y):
    t = np.linspace(x[0], x[-1], x.shape[0])
    return [[np.nanmax(y)-np.nanmin(y), 0.000001, i, np.nanmin(y)] for i in t] + [[np.nanmax(y), -0.000001, i, np.nanmin(y)] for i in t]

def simplified_sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def sigmoid(x, L, k, x0, h):
    return L / (1 + np.exp(-k * (x - x0))) + h

def sigmoid_metrix(params: list, x: list | np.ndarray, y: list | np.ndarray):
    L, k, x0, h = params
    y_fit = sigmoid(x, L, k, x0, h)
    return np.sqrt(np.nanmean((y_fit-y)**2))

def leastsq_h(y, low: float = 0.0, high: float = 1.0):
    y_pred = np.linspace(low, high, 20)
    res = np.zeros(20, dtype=np.float64)

    for i in range(20):
        res[i] = np.sqrt(np.mean((y - y_pred[i])**2))

    min_arg = np.argmin(res)
    low_idx = min_arg-1 if min_arg >= 1 else 0
    high_idx = min_arg+1 if min_arg+1 < 20 else 19
    if y_pred[high_idx] - y_pred[low_idx] < 0.00001:
        return y_pred[min_arg]
    else:
        return leastsq_h(y, y_pred[low_idx], y_pred[high_idx])

def leastsq_x0(x, y):
    dy = (y-np.min(y))/(np.max(y)-np.min(y))

    dx = np.ediff1d(x)/2
    x0_set = x[0:-1] + dx
    RMSE = np.zeros(dx.shape[0])
    params = []

    for i, x0 in enumerate(x0_set):
        y1 = dy[np.where(x<=x0)[0]]
        y2 = dy[np.where(x>x0)[0]]

        a1 = leastsq_h(y1)
        a2 = leastsq_h(y2)

        if a1 <= a2:
            L, h = (a2-a1)*(np.max(y)-np.min(y)), a1*(np.max(y)-np.min(y))+np.min(y)
            k = 0.0005
        else:
            k = -0.0005
            L, h = (a1-a2)*(np.max(y)-np.min(y)), a2*(np.max(y)-np.min(y))+np.min(y)
        
        RMSE[i] = sigmoid_metrix([L, k, x0, h], x, y)
        params.append([L, k, x0, h])

    idx = np.argmin(RMSE)
    return params[idx]

def sigmoid_fit(x, y):
    # delelte nan values
    nan_idx = np.where((np.isnan(x))|(np.isnan(y)))[0]
    x = np.delete(x, nan_idx)
    y = np.delete(y, nan_idx)

    params = leastsq_x0(x, y)
    L, k, x0, h = params

    return L, k, x0, h, sigmoid_metrix([L, k, x0, h], x, y)


def residual(params: list, x: list | np.ndarray, y: list | np.ndarray):
    L, k, x0, h = params

    # Apply bounds by adding penalty terms
    if L < 0 or L > np.nanmax(y)-np.nanmin(y):
        L_penalty = 1e6 * (np.abs(L)+1)**2
    else:
        L_penalty = 0

    if k < -0.00005 or k > 0.00005:
        k_penalty = 1e6 * (k*100)**2  # Quadratic penalty for negative k
    else:
        k_penalty = 0

    if x0 < x[0] or x0 > x[-1]:
        x0_penalty = 1e6
    else:
        x0_penalty = 0
    
    if h < np.nanmin(y):
        h_penalty = 1e6 * (np.abs(h-np.nanmin(y))+1)**2
    else:
        h_penalty = 0 

    return y - sigmoid(x, L, k, x0, h) + L_penalty + k_penalty + x0_penalty + h_penalty

from scipy.optimize import leastsq

def sigmoid_fit_leastsq(x, y):
    # delelte nan values
    nan_idx = np.where((np.isnan(x))|(np.isnan(y)))[0]
    x = np.delete(x, nan_idx)
    y = np.delete(y, nan_idx)

    # fit
    guesses = init_guess(x, y)
    parameters = []
    RMSE = np.zeros(len(guesses), dtype=np.float64)
    bounds = ([0-0.001, -0.0005, np.nanmin(x)-0.001, 0-0.001], 
              [np.nanmax(y)-np.nanmin(y), 0.0005, np.nanmax(x)+0.001, np.nanmax(y)*1.1])

    for i, p0 in enumerate(guesses):
        params, _ = leastsq(residual, p0, args=(x, y))
        #curve_fit(sigmoid, x, y, p0=p0, method='dogbox', bounds=bounds)
        RMSE[i] = sigmoid_metrix(params, x, y)
        parameters.append(params)

    best_guess = np.argmin(RMSE)
    L, k, x0, h = parameters[best_guess]

    return L, k, x0, h, RMSE[best_guess]