import numpy as np
import pandas as pd
from mylib.local_path import f_CellReg_modi
import pickle
import copy as cp

from mylib.field.tracker_v2 import Tracker2d
from scipy.optimize import curve_fit, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(
    maze_type: int,
    paradigm: str,
    direc: str = 'cis',
    f: pd.DataFrame = f_CellReg_modi,
):
    file_indices = np.where(
        (f_CellReg_modi['maze_type'] == maze_type) & 
        (f_CellReg_modi['paradigm'] == paradigm) &
        (f_CellReg_modi['Type'] == 'Real')
    )[0]
    
    I, A, P = [], [], []
    
    if paradigm == 'CrossMaze':
        for i in file_indices:
            with open(f['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            field_reg = cp.deepcopy(trace['field_reg'])
            tracker = Tracker2d(field_reg=field_reg)
            prob = tracker.calc_P1()
            
            x, y = np.meshgrid(np.arange(prob.shape[1]), np.arange(1, prob.shape[0]+1))
            
            I.append(x.flatten())
            A.append(y.flatten())
            P.append(prob.flatten())
    else:
        for i in file_indices:
            with open(f['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            field_reg = cp.deepcopy(trace[direc]['field_reg'])
            tracker = Tracker2d(field_reg=field_reg)
            prob = tracker.calc_P1()
            
            x, y = np.meshgrid(np.arange(prob.shape[1]), np.arange(1, prob.shape[0]+1))
            I.append(x.flatten())
            A.append(y.flatten())
            P.append(prob.flatten())

    I = np.concatenate(I)
    A = np.concatenate(A)
    P = np.concatenate(P)
    
    idx = np.where(np.isnan(P) == False)[0]
    I = I[idx]
    A = A[idx]
    P = P[idx]
    
    return I, A, P

def get_surface(
    maze_type: int,
    paradigm: str,
    direc: str = 'cis',
    f: pd.DataFrame = f_CellReg_modi,
) -> np.ndarray:
    """
    Return
    ------
    P Matrix: np.ndarray, (nsession times n_session)
    """
    file_indices = np.where(
        (f_CellReg_modi['maze_type'] == maze_type) & 
        (f_CellReg_modi['paradigm'] == paradigm) &
        (f_CellReg_modi['Type'] == 'Real')
    )[0]
    
    P = []
    
    if paradigm == 'CrossMaze':
        for i in file_indices:
            with open(f['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            field_reg = cp.deepcopy(trace['field_reg'])
            tracker = Tracker2d(field_reg=field_reg)
            prob = tracker.calc_P1()
            
            P.append(prob)
    else:
        for i in file_indices:
            with open(f['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
            
            field_reg = cp.deepcopy(trace[direc]['field_reg'])
            tracker = Tracker2d(field_reg=field_reg)
            prob = tracker.calc_P1()

            P.append(prob)

    x_max = np.max([p.shape[1] for p in P])
    y_max = np.max([p.shape[0] for p in P])
    
    
    total_P = np.zeros((y_max, x_max, len(P)))
    for i in range(len(P)):
        P_MEAN = np.full((y_max, x_max), np.nan)
        P_MEAN[:P[i].shape[0], :P[i].shape[1]] = P[i]
        total_P[:, :, i] = P_MEAN
    
    return np.nanmean(total_P, axis=2)

def reci_decay(x, u, v):
    return u / (x + v)
    #return a*np.exp(-np.power(x/b, c))

def kww_decay(x, u, v, w):
    return u * np.exp(-np.power(x/v, w))

def exp_decay(x, u, v):
    return u * np.exp(-v * x)

def secorder_reci_decay(x, u, v, w):
    return u / (x**2 - v*x + w)

def fit_decay(I, A, P, func = reci_decay, act = 0, ax = None, **kwargs):
    idx = np.where((A == act) & (np.isnan(P) == False))[0]
    
    try:
        params, _ = curve_fit(func, I[idx], P[idx], 
                          bounds=([0, 0], [np.inf, np.inf]))
    except:
        params, _ = curve_fit(func, I[idx], P[idx], 
                          bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))   
    
    x_fit = np.linspace(np.min(I[idx]), np.max(I[idx]), 10000)
    z_fit = func(x_fit, *params)
    
    print(f"Act = {act}, u = {params[0]}, v = {params[1]}")
    if ax is not None:
        ax.plot(x_fit, z_fit, label = f"{act}", **kwargs)
    else:
        sns.stripplot(x=I[idx], y=P[idx], jitter=0.2, edgecolor='k', linewidth=0.15, size=3, alpha=0.8)
        plt.plot(x_fit, z_fit, label = f"{act}", **kwargs)
    
    return params

def reci(x, L, b):
    return 1- L / (x + b)

def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def logistic_kww(x, k, b):
    return 1 / (1 + np.exp(- np.power(k/x, b)))

def secorder_reci(x, L, k, b):
    return  1 - L /  (x**2 - k * x + b)

def fit_increase(I, A, P, func = reci, inact = 0, ax = None, **kwargs):
    idx = np.where((I == inact) & (np.isnan(P) == False))[0]
    try:
        params, _ = curve_fit(func, A[idx], P[idx], bounds=([0, 0], [np.inf, np.inf]))
    except:
        params, _ = curve_fit(func, A[idx], P[idx], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

    y_fit = np.linspace(np.min(A[idx]), np.max(A[idx]), 10000)
    z_fit = func(y_fit, *params)
    print(f"Inact = {inact}, k = {params[0]}, b = {params[1]}")
    if ax is not None:
        ax.plot(y_fit, z_fit, label = f"{inact}", **kwargs)
    else:
        sns.stripplot(x=A[idx], y=P[idx], jitter=0.2, edgecolor='k', linewidth=0.15, size=3, alpha=0.8)
        plt.plot(y_fit-1, z_fit, label = f"{inact}", **kwargs)
    
    return params

def calc_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def compare_fit_retention(I, A, P):
    imax = np.max(I)+1
    
    IV = [] # I value
    RMSE = []
    func_names = ['reci', 'logistic', 'logistic_kww', 'secorder_reci']
    
    funcs = [reci, logistic, logistic_kww, secorder_reci]
    bounds = [
        ([0.00001, 0.00001], [np.inf, np.inf]),
        ([0.00001, 0.00001], [np.inf, np.inf]),
        ([0.00001, 0.00001], [np.inf, np.inf]),
        ([0.00001, 0.00001, 0.00001], [np.inf, np.inf, np.inf])
    ]
    
    for inact in range(imax-2):
        idx = np.where((I == inact) & (np.isnan(P) == False))[0]
        
        for i, func in enumerate(funcs):
            try:
                params, _ = curve_fit(func, A[idx], P[idx], bounds=bounds[i])
                z_fit = func(A[idx], *params)
                RMSE.append(calc_loss(P[idx], z_fit))
                IV.append(inact)
            except:
                RMSE.append(np.nan)
                IV.append(inact)
                
    return np.array(IV), np.array(RMSE), np.array(func_names * (imax-2))

def compare_fit_recover(I, A, P):
    amax = np.max(A)
    
    AV = [] # A value
    RMSE = []
    func_names = ['reci', 'kww', 'exp', 'secorder_reci']
    
    funcs = [reci_decay, kww_decay, exp_decay, secorder_reci_decay]
    bounds = [
        ([0.00001, 0.00001], [np.inf, np.inf]),
        ([0.00001, 0.00001, 0.00001], [np.inf, np.inf, np.inf]),
        ([0.00001, 0.00001], [np.inf, np.inf]),
        ([0.00001, 0.00001, 0.00001], [np.inf, np.inf, np.inf])
    ]
    
    for act in range(1, amax-1):
        idx = np.where((A == act) & (np.isnan(P) == False))[0]
        
        for i, func in enumerate(funcs):
            params, _ = curve_fit(func, I[idx], P[idx], bounds=bounds[i])

            z_fit = func(I[idx], *params)
            RMSE.append(calc_loss(P[idx], z_fit))
            AV.append(act)
            
    return np.array(AV), np.array(RMSE), np.array(func_names * (amax-2))

def func_curved_surface(data, b, u, v, w):
    y, x = data
    return (x + b) / (u*y + v*x + w)

def fit_curved_surface(I, A, P):
    
    def objective_function(params):
        return np.sum((func_curved_surface((I, A), *params) - P) ** 2)
    
    bounds = [(0.00001, 100), (0.00001, 100), (0.00001, 100), (0.00001, 100)]
    result = differential_evolution(objective_function, bounds, maxiter=10000)
    print("Optimized Parameters:", result.x)
    return result.x