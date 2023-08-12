import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mylib.maze_utils3 import Clear_Axes
import numpy as np
import copy as cp

def _insert_nan(
    cal_trace: np.ndarray,
    cal_time: np.ndarray,
    t_thre: float = 500.
):
    dt = np.ediff1d(cal_time)
    insert_indices = np.where(dt >= t_thre)[0]
    
    cal_time = cal_time.astype(np.float64)
    
    return np.insert(cal_trace, insert_indices+1, np.nan), np.insert(cal_time, insert_indices+1, np.nan)

def LapwiseCalciumTraceAxes(
    ax: Axes,
    cal_trace: np.ndarray,
    cal_time: np.ndarray,
    t_thre: float = 500.,
    magnify: float = 60000.,
    color: str = 'black',
    linewidth: float = 0.8,
    y_max: float | None = None,
    **kwargs
) -> Axes:
    ax = Clear_Axes(ax, close_spines=['top', 'right'],ifxticks=True, ifyticks=True)
    in_field_trace, in_field_time = _insert_nan(cal_trace, cal_time, t_thre = t_thre)
    
    nan_indices = np.concatenate([[-1], np.where(np.isnan(in_field_time))[0], [in_field_time.shape[0]]])
    
    base_t = cp.deepcopy(in_field_time)
    dt = cp.deepcopy(in_field_time)
    
    labels = []
    yticks = []
    
    for i in range(nan_indices.shape[0]-1):
        beg, end = nan_indices[i]+1, nan_indices[i+1]
        base_t[beg:end] = in_field_time[beg]
        dt[beg:end] = (in_field_time[beg:end] - in_field_time[beg]) / (in_field_time[end-1] - in_field_time[beg])
        labels = labels + ['0', str(round(np.nanmax(in_field_trace[beg:end]), 2))]
        yticks = yticks + [base_t[beg], base_t[beg]+np.nanmax(in_field_trace[beg:end])*magnify]
        
    y = base_t + in_field_trace*magnify
    
    ax.plot(dt, y, color=color, linewidth=linewidth, **kwargs)
    
    ax.set_yticks(yticks, labels)
    
    if y_max is not None:
        ax.set_ylim([0, y_max])
    return ax

