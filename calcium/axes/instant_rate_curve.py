import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mylib.maze_utils3 import Clear_Axes
from mylib.maze_graph import NRGs, correct_paths, incorrect_paths
import time
import seaborn as sns

from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from mylib.calcium.smooth.interpolation import interpolated_smooth
from mylib.field.sigmoid import sigmoid_fit_leastsq, sigmoid, sigmoid_fit
from mylib.decoder.PiecewiseConstantSigmoidRegression import PiecewiseRegressionModel, TwoPiecesPiecewiseSigmoidRegression
from scipy.stats import t

def calc_ci(data, ci: float = 0.95):
    data = np.delete(data, np.where(np.isnan(data))[0])
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    t_score = t.ppf(1 - (1 - ci) / 2, n-1)
    margin_of_error = t_score * (std_dev / np.sqrt(n))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound

def smooth_lineplot(
    ax: Axes, 
    x: np.ndarray | list,
    y: np.ndarray | list,
    shadow_color: str = 'grey',
    line_color: str = 'black',
    orient: str = 'x',
    alpha=0.5,
    M: np.ndarray = None,
    window: int = 10,
    sigma=3, 
    folder=1, 
    kind: str = 'linear',
    insert_num: int = 1000,
    fill_kwargs = {},
    **kwargs
) -> Axes:

    uniq_x = np.unique(x)
    uniq_y = np.zeros_like(uniq_x)
    lower_bounds, upper_bounds = np.zeros_like(uniq_x), np.zeros_like(uniq_x)

    for i in range(uniq_x.shape[0]):
        sub_y = y[np.where(x == uniq_x[i])[0]]
        uniq_y[i] = np.nanmean(sub_y)
        lower_bounds[i], upper_bounds[i] = calc_ci(sub_y)

    smooth_x = np.linspace(np.min(uniq_x), np.max(uniq_x), insert_num)

    if M is None:
        M = gaussian_smooth_matrix1d(insert_num, window=window, sigma=sigma, folder=folder, dis_stamp=smooth_x)

    _, smooth_lower_bounds = interpolated_smooth(uniq_x, lower_bounds, kind=kind, insert_num=insert_num)
    smooth_lower_bounds = np.dot(M, smooth_lower_bounds)
    _, smooth_upper_bounds = interpolated_smooth(uniq_x, upper_bounds, kind=kind, insert_num=insert_num)
    smooth_upper_bounds = np.dot(M, smooth_upper_bounds)
    _, smooth_y = interpolated_smooth(uniq_x, uniq_y, kind=kind, insert_num=insert_num)
    smooth_y = np.dot(M, smooth_y)

    if orient == 'x':
        ax.plot(smooth_x, smooth_y, color=line_color, **kwargs)
        ax.fill_between(x=smooth_x, y1=smooth_lower_bounds, y2=smooth_upper_bounds, color=shadow_color, alpha = alpha, **fill_kwargs)
    elif orient == 'y':
        ax.plot(smooth_y, smooth_x, color=line_color, **kwargs)
        ax.fill_betweenx(y=smooth_x, x1=smooth_lower_bounds, x2=smooth_upper_bounds, color=shadow_color, alpha = alpha, **fill_kwargs)

    return ax

def InstantRateCurveAxes(
    ax: Axes,
    time_stamp: np.ndarray,
    content: np.ndarray,
    title: str="",
    is_display_max_only: bool=True,
    smooth_window_length: int = 100,
    folder: float = 0.1,
    sigma: float = 3,
    num_pieces_range: int = range(1, 3),
    lam: float = 0,
    k_default: float = 0.0005,
    M: np.ndarray = None,
    t_max: float = None,
    smooth_kwarg: dict = {'kind':'linear', 'insert_num':1000, 'shadow_color':'grey', 'line_color':'black','alpha':0.5},
    **kwargs
) -> tuple[Axes, list, list]:

    ax = Clear_Axes(ax, close_spines=['top', 'right', 'left'], ifxticks=True)
    
    x = np.linspace(time_stamp[0], time_stamp[-1], 1000)
    model = TwoPiecesPiecewiseSigmoidRegression()
    model.fit(time_stamp, content,k=k_default)
    y = model.predict(x)
    #regression_tester = PiecewiseRegressionModel(time_stamp, content, num_pieces_range, lam=lam, k_default=k_default)
    #regression_tester.fit()
    #y = regression_tester.best_model.predict(x)

    ax = smooth_lineplot(
        ax=ax,
        x=time_stamp,
        y=content,
        orient='y',
        M=M,
        sigma=sigma,
        folder=folder,
        window=smooth_window_length,
        **smooth_kwarg
    )
    ax.plot(y, x, ls = '--', color = 'orange', label='fit by sigmoid')
    ax.axvline(0, ls = ':', color = 'black')

    rate_max = (int(np.nanmax(content)*100))/100
    if np.nanmax(y) > rate_max:
        rate_max = np.nanmax(y)

    if t_max is not None:
        ax.set_ylim([0, t_max])

    if is_display_max_only:
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 1])
    
    ax.set_xlabel("Normalized\nEvents Rate", fontsize=12)
    ax.set_title(title, fontsize=12)
    
    return ax
