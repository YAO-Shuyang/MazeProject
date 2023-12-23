import numpy as np
import matplotlib.pyplot as plt
from mylib.maze_utils3 import DrawMazeProfile, Clear_Axes, spike_nodes_transform, DateTime, ColorBarsTicks, FastDistance, Delete_NAN
import copy as cp
import os
from mylib.maze_graph import correct_paths, StartPoints, EndPoints
from mylib.decoder.PiecewiseConstantSigmoidRegression import PiecewiseRegressionModel, TwoPiecesPiecewiseSigmoidRegression
import sklearn.preprocessing
from tqdm import tqdm
import time
from scipy.stats import ttest_1samp

LOWER_BOUND = 5
UPPER_BOUND = 95
CORRECT_PASS = True
FAIL_TO_PASS = False

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

@jit(nopython=True)
def parallel_shuffle(
    trial_num, 
    total_events_num, 
    frames_set, 
    cal_events_time,
    durations: np.ndarray,
    shuffle_times: int=5000,
):
    dt = np.ediff1d(np.unique(cal_events_time))
    init_break_points = np.unique(cal_events_time)[0:-1] + dt
    const1, const2, bps = np.zeros(shuffle_times, np.float64), np.zeros(shuffle_times, np.float64), np.zeros(shuffle_times, np.float64)
    
    for i in range(shuffle_times):
        rand_events_num = np.zeros(trial_num)
        rand_events_frame = np.random.choice(frames_set, total_events_num, replace=False)
        for frame in rand_events_frame:
            rand_events_num[frame] += 1
            
        rand_events_rate = (rand_events_num / durations * 1000)
            
        rand_events_rate = rand_events_rate / np.nanmax(rand_events_rate)
        rand_events_rate[np.where(np.isnan(rand_events_rate))[0]] = 0
        
        total_losses = _twopiece_fit(init_break_points, cal_events_time, rand_events_rate)
        
        total_losses[np.where(np.isnan(total_losses))[0]] = 100000
   
        idx =  np.argmin(total_losses)
        if np.min(total_losses) == 100000:
            bps[i] = -1000
        else:
            bps[i] = init_break_points[idx]
        const1[i] = np.nanmean(rand_events_rate[np.where(cal_events_time<=bps[i])[0]])
        const2[i] = np.nanmean(rand_events_rate[np.where(cal_events_time>bps[i])[0]])

    return const1, const2, bps

@jit(nopython=True)
def _set_range_jit(maze_type: int, father_field: np.ndarray, CP: np.ndarray) -> tuple[float, float]:
    """
    set_range: set the range of the field on the correct track.

    Parameters
    ----------
    maze_type : int
        Maze type
    father_field : np.ndarray
        The bins in the field.

    Returns
    -------
    tuple[float, float]
        The range of the field on the correct track, (min, max).
        Return (np.nan, np.nan) if all of the bins in the field are 
        situated at incorrect track.
    """
    
    field_range = np.zeros(len(father_field), dtype=np.float64)

    IS_INCORRECT_TRACK_FIELDS = True

    for i, n in enumerate(father_field):
        if n not in CP:
            field_range[i] = np.nan
        else:
            IS_INCORRECT_TRACK_FIELDS = False
            field_range[i] = np.where(CP == n)[0][0]
    
    if IS_INCORRECT_TRACK_FIELDS:
        return (np.nan, np.nan)
    else:
        return (np.nanmin(field_range), np.nanmax(field_range))    

@jit(nopython=True)
def temporal_analysis(
    ms_time_original: np.ndarray,
    deconv_signal: np.ndarray,
    in_field_nodes: np.ndarray,
    in_field_time: np.ndarray,
    interval_indices: np.ndarray,
    father_field: np.ndarray,
    maze_type: int,
    CP: np.ndarray,
    signal_folder: np.ndarray = np.linspace(1, 3, 11),
    t_thre: float = 500.,
    t_unit: float = 1000.,
) -> np.ndarray:
            
    field_range = _set_range_jit(maze_type=maze_type, father_field=father_field, CP=CP)
    
    if np.isnan(field_range[0]) or np.isnan(field_range[1]):
        return
    
    signal_std = np.nanstd(deconv_signal)
    thre_num = signal_folder.shape[0]
    err_events = np.zeros((interval_indices.shape[0]-1, 5), np.float64)
    cal_events_rate = np.zeros((interval_indices.shape[0]-1, thre_num), dtype=np.float64)
    cal_events_time = np.zeros((interval_indices.shape[0]-1, thre_num), dtype=np.float64)
    cal_events_num = np.zeros((interval_indices.shape[0]-1, thre_num), dtype=np.float64)
    cal_frames_num = np.zeros(interval_indices.shape[0]-1, dtype=np.int64)
    durations = np.zeros(interval_indices.shape[0]-1, dtype=np.float64)
    
    for i in range(interval_indices.shape[0]-1):
        
        beg, end = interval_indices[i], interval_indices[i+1]
        beg_node, end_node = in_field_nodes[beg], in_field_nodes[end-1]
        ms_indices = np.where((ms_time_original >= in_field_time[beg])&(ms_time_original <= in_field_time[end-1]))[0]
        SUITABLE_FRAME_NUM = len(ms_indices) > 1
        if beg_node == CP[int(field_range[0])] and end_node == CP[int(field_range[1])] and SUITABLE_FRAME_NUM:

            cal_frames_num[i] = len(ms_indices)
            durations[i] = (ms_indices[-1]-ms_indices[0])/1000*t_unit + 0.05*t_unit
            for j in range(thre_num):
                cal_events_num[i, j] = len(np.where(deconv_signal[ms_indices] >= signal_folder[j]*np.nanstd(deconv_signal))[0])
                cal_events_time[i, j] = (in_field_time[beg] + in_field_time[end-1])/2
                cal_events_rate[i, j] = cal_events_num[i, j] / durations[i] * 1000
        else:
            err_events[i, 0] = 1
            err_events[i, 1] = field_range[0]
            err_events[i, 2] = field_range[1]
            err_events[i, 3] = in_field_time[beg]/1000
            err_events[i, 4] = in_field_time[end-1]/1000
            continue
    
    err_indices = np.where(err_events[:, 0] == 1)[0]
    cal_events_indices = np.where(err_events[:, 0] == 0)[0]
    return (
        cal_events_num[cal_events_indices, :],
        cal_events_time[cal_events_indices, :],
        cal_events_rate[cal_events_indices, :],
        durations[cal_events_indices],
        cal_frames_num[cal_events_indices],
        err_events[err_indices, :],
        field_range
    )
        
    
def set_range(maze_type: int, field: list | np.ndarray) -> tuple[float, float]:
    """
    set_range: set the range of the field on the correct track.

    Parameters
    ----------
    maze_type : int
        Maze type
    field : list | np.ndarray
        The bins in the field.

    Returns
    -------
    tuple[float, float]
        The range of the field on the correct track, (min, max).
        Return (np.nan, np.nan) if all of the bins in the field are 
        situated at incorrect track.
    """
    

    correct_path = correct_paths[maze_type]
    field_range = np.zeros(len(field), dtype=np.float64)

    IS_INCORRECT_TRACK_FIELDS = True

    for i, n in enumerate(field):
        if n not in correct_path:
            field_range[i] = np.nan
        else:
            IS_INCORRECT_TRACK_FIELDS = False
            field_range[i] = np.where(correct_path == n)[0][0]
    
    if IS_INCORRECT_TRACK_FIELDS:
        return (np.nan, np.nan)
    else:
        return (np.nanmin(field_range), np.nanmax(field_range))    
    
            
def _star(p:str):
    '''
    Note: notations of significance according to the input p value.

    # Input:
    - p:str, the p value.

    # Output:
    - str, the notation.
    '''
    if p > 0.05:
        return 'ns'
    elif p <= 0.05 and p > 0.01:
        return '*'
    elif p <= 0.01 and p > 0.001:
        return '**'
    elif p <= 0.001 and p > 0.0001:
        return '***'
    elif p <= 0.0001:
        return '****'



class InFieldRateChangeModel:
    def __init__(self) -> None:
        self.total_events_num = 0
        self.total_durations = 0
        
        self.cal_events_num = None
        self.cal_events_rate = None
        self.durations = None

        self.cal_events_rate = None
        self.cal_events_time = None

        self.field = None
        self.field_range = None

        self.L, self.k, self.x0, self.b = None, None, None, None
        self.rand_L, self.rand_k, self.rand_x0, self.rand_b = None, None, None, None
        self.is_change = False
        self._ctype = 'retain'

    def _init_deconv_signal(
        self, 
        deconv_signal: np.ndarray,
        ms_time_original: np.ndarray,
        ms_time_behav: np.ndarray
    ) -> np.ndarray:
        deconv_signal_behav = np.zeros_like(ms_time_behav, dtype=np.float64)

        for i, t in enumerate(ms_time_behav):
            if np.isnan(t):
                deconv_signal_behav[i] = np.isnan
            else:
                t_idx = np.where(ms_time_original >= t)[0][0]
                deconv_signal_behav[i] = deconv_signal[t_idx]

        self.signal_std = np.std(deconv_signal)
        
        return deconv_signal_behav

    def temporal_analysis(self,
        field: np.ndarray | list,
        maze_type: int,
        ms_time_original: np.ndarray,
        deconv_signal: np.ndarray,
        t_thre: float = 500,
        t_unit: float = 1000, # ms
        signal_folder: np.ndarray = np.linspace(1, 3, 11),
        behav_time: np.ndarray = None,
        behav_nodes: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:

        # Within field trajectory and corresponding frames.
        if len(field) == 1:
            in_field_indices = np.where(behav_nodes == field[0])[0]
        else:
            in_field_indices = np.sort(np.concatenate([np.where(behav_nodes == k)[0] for k in field]))

        in_field_time = behav_time[in_field_indices]
        in_field_nodes = behav_nodes[in_field_indices]

        dt = np.ediff1d(in_field_time)
        interval_indices = np.concatenate([[0], np.where(dt >= t_thre)[0]+1, [in_field_time.shape[0]]])
      
        cal_events_num, cal_events_time, cal_events_rate, durations, cal_frames_num, err_events, field_range = temporal_analysis(
            ms_time_original=cp.deepcopy(ms_time_original),
            deconv_signal=cp.deepcopy(deconv_signal),
            in_field_time=in_field_time,
            in_field_nodes=in_field_nodes,
            interval_indices=interval_indices,
            father_field=np.array(field, np.int64),
            CP=cp.deepcopy(correct_paths[int(maze_type)]),
            signal_folder=signal_folder,
            t_thre=t_thre,
            t_unit=t_unit,
            maze_type=maze_type
        )
        """
        maze_type = trace['maze_type']
        ms_time_behav = cp.deepcopy(trace['ms_time_behav'])
        deconv_signal = cp.deepcopy(trace['DeconvSignal'][n, :])
        ms_time = cp.deepcopy(trace['ms_time'])

        self.in_field_indices = np.array([], dtype=np.int64)

        # set field range
        field_range = set_range(maze_type=maze_type, field=field)

        # get in field data
        in_field_time, in_field_nodes, in_field_pos = self.in_field_trajectory(behav_nodes=behav_nodes, behav_time=behav_time, behav_pos=behav_pos, field=field, t_thre = t_thre)
        deconv_signal_behav = self._init_deconv_signal(deconv_signal=deconv_signal, ms_time_original=ms_time, ms_time_behav=ms_time_behav)
        self.deconv_signal = deconv_signal_behav

        break_points = np.concatenate([[-1], np.where(np.isnan(in_field_time))[0], [in_field_time.shape[0]]])
        include_indices = []
        err_events = []
        CP = correct_paths[maze_type]
        for i in range(break_points.shape[0]-1):
            beg, end = break_points[i]+1, break_points[i+1]-1

            beg_behav_idx = np.where(behav_time==in_field_time[beg])[0][0]
            end_behav_idx = np.where(behav_time==in_field_time[end])[0][0]

            prev_idx = beg_behav_idx-1 if beg_behav_idx != 0 else 0
            next_idx = end_behav_idx+1 if end_behav_idx != behav_time.shape[0]-1 else behav_time.shape[0]-1

            prev_node = behav_nodes[prev_idx]
            next_node = behav_nodes[next_idx]

            INCLUDE_START_OR_END_POINT = StartPoints[maze_type] in field or EndPoints[maze_type] in field or prev_node == EndPoints[maze_type] or next_node == StartPoints[maze_type]

            # Unsuccess events
            ENTER_INCORRECT_PATH = prev_node not in CP or next_node not in CP
            TOO_LESS_FRAME = beg == end # only 1 frame.
            t_min, t_max = behav_time[prev_idx], behav_time[next_idx]
            if ENTER_INCORRECT_PATH:
                print("ENTER INCORRECT PATH")
                err_events.append((field_range[0], field_range[1], t_min, t_max, "ENTER INCORRECT PATH"))
                continue

            INCLUDE_START_OR_END_POINT = StartPoints[maze_type] in field or EndPoints[maze_type] in field or prev_node == EndPoints[maze_type] or next_node == StartPoints[maze_type]
            TURN_AROUND_EVENT = prev_node == next_node and not INCLUDE_START_OR_END_POINT
            WRONG_DIRECTION = np.where(CP==prev_node)[0][0] > np.where(CP==next_node)[0][0] and not INCLUDE_START_OR_END_POINT

            if TOO_LESS_FRAME:
                #print("TOO LESS FRAM")
                err_events.append((field_range[0], field_range[1], t_min, t_max, "TOO LESS FRAM"))
                continue
            if TURN_AROUND_EVENT:
                #print("TURN AROUND")
                err_events.append((field_range[0], field_range[1], t_min, t_max, "TURN AROUND"))
                
            if WRONG_DIRECTION:
                #print("WRONG DIRECTION")
                err_events.append((field_range[0], field_range[1], t_min, t_max, "WRONG DIRECTION"))
                
            cal_events_indices = np.where((ms_time_behav >= in_field_time[beg])&(ms_time_behav <= in_field_time[end]))[0]
            if len(cal_events_indices) <= 1:
                #print("CALCIUM FRAME LESS THAN 1")
                err_events.append((field_range[0], field_range[1], t_min, t_max, "CALCIUM FRAME LESS THAN 1"))
                continue
                     
            include_indices.append(i)
 
        
        cal_events_rate = np.zeros((len(include_indices), len(signal_folder)), dtype=np.float64)
        cal_events_time = np.zeros((len(include_indices), len(signal_folder)), dtype=np.float64)
        cal_events_num = np.zeros((len(include_indices), len(signal_folder)), dtype=np.float64)
        cal_frames_num = np.zeros(len(include_indices), dtype=np.int64)
        durations = np.zeros(len(include_indices), dtype=np.float64)
        
        for i, idx in enumerate(include_indices):
            beg, end = break_points[idx]+1, break_points[idx+1]-1
            cal_events_indices = np.where((ms_time_behav >= in_field_time[beg])&(ms_time_behav <= in_field_time[end]))[0]
            cal_frames_num[i] = len(cal_events_indices)
            durations[i] = (cal_events_indices[-1] - cal_events_indices[0])/1000*t_unit + 1*0.05*t_unit
            self.in_field_indices = np.concatenate([self.in_field_indices, cal_events_indices])

            for j, thre in enumerate(signal_folder): # len(np.where(spikes[cal_events_indices]==1)[0])
                cal_events_num[i, j] = len(np.where(deconv_signal_behav[cal_events_indices]>=thre*self.signal_std)[0])
                cal_events_time[i, j] = (in_field_time[end] + in_field_time[beg])/2
                cal_events_rate[i, j] = cal_events_num[i, j] / durations[i] * 1000
        """ 
        cal_events_rate = cal_events_rate / np.nanmax(cal_events_rate, axis=0)
        cal_events_rate[np.where(np.isnan(cal_events_rate))[0]] = 0

        self.field_range = field_range
        self.durations = durations

        self.total_durations = np.nansum(durations)
        self.total_events_num = np.nansum(cal_events_num, axis=0)
        self.cal_events_num = cal_events_num
        self.cal_events_rate = cal_events_rate
        self.cal_events_time = cal_events_time
        self.cal_frames_num = cal_frames_num
        self.err_events = err_events
        
        self._field_emerge_lap = self._find_field_emerge_lap()
        self._field_disappear_lap = self._find_field_disappear_lap()
        self._active_lap_percent = len(np.where(self.cal_events_num[:, -1]==0)[0])/self.cal_events_num.shape[0]

        return cal_events_rate, cal_events_time, err_events

    @property
    def field_emerge_lap(self):
        return self._field_emerge_lap
    
    @property
    def field_disapper_lap(self):
        return self._field_disappear_lap
    
    @property
    def active_lap_percent(self):
        return self._active_lap_percent

    def _fit(
        self,
        cal_events_time: np.ndarray,
        cal_events_rate: np.ndarray,
        num_pieces_range: list | np.ndarray=[1,2], 
        lam: float=0,
        k_default=0.0005
    ) -> tuple[float, float, float, float]:
        model = TwoPiecesPiecewiseSigmoidRegression()
        model.fit(cal_events_time.flatten(), cal_events_rate.flatten(), k=k_default)
        return model.L, model.k, model.x0, model.b
        #model = PiecewiseRegressionModel(cal_events_time.flatten(), cal_events_rate.flatten(), num_pieces_range, lam=lam, k_default=k_default)
        #model.fit()
        #return model.best_model.L[0], model.best_model.k[0], model.best_model.x0[0], model.best_model.b[0]

    def _classify(self):
        WEAKEN = self.k < 0 and self.is_change == True
        ENHANCE = self.k > 0 and self.is_change == True
        RETAIN = self.k == 0 or self.is_change == False

        if WEAKEN:
            self._ctype = 'weakened'
        elif ENHANCE:
            self._ctype = 'enhanced'
        elif RETAIN:
            self._ctype = 'retained'

    @property
    def ctype(self):
        return self._ctype

    def fit(
        self, 
        num_pieces_range: list | np.ndarray=[1,2], 
        lam: float=0,
        k_default=0.0005
    ) -> tuple[float, float, float, float]:
        if self.cal_events_rate is None or self.cal_events_time is None:
            return None, None, None, None
            raise ValueError("Model should undergo temporal_analysis first!")

        self.L, self.k, self.x0, self.b = self._fit(self.cal_events_time.flatten(), self.cal_events_rate.flatten(), num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        return self.L, self.k, self.x0, self.b

    def shuffle_test(
        self,
        shuffle_times: int = 1000,
        num_pieces_range: list | np.ndarray=[1,2], 
        lam: float=0,
        k_default=0.0005
    ) -> None:
        if self.cal_events_rate is None or self.cal_events_time is None:
            raise ValueError("Model should undergo temporal_analysis first!")
        self.shuffle_times = shuffle_times
        trial_num, thre_num = self.cal_events_num.shape[0], self.cal_events_num.shape[1]
        frames_set = np.concatenate([np.repeat(i, self.cal_frames_num[i]) for i in range(trial_num)])
        rand_L, rand_k, rand_x0, rand_b = np.zeros(shuffle_times), np.zeros(shuffle_times), np.zeros(shuffle_times), np.zeros(shuffle_times)
        
        total_events_num = int(self.total_events_num[-1])
        const1, const2, rand_x0 = parallel_shuffle(
            trial_num=trial_num,
            total_events_num=total_events_num,
            frames_set=frames_set,
            cal_events_time=self.cal_events_time[:, -1],
            durations=self.durations,
            shuffle_times=shuffle_times
        )
        
        dc = const2-const1
        case1 = np.where(dc > 0)[0]
        case2 = np.where(dc < 0)[0]
        case3 = np.where(dc == 0)[0]
        rand_L = np.abs(dc)
        rand_k[case1] = k_default
        rand_k[case2] = -k_default
        rand_b[case1] = const1[case1]
        rand_b[case2] = const2[case2]
        rand_b[case3] = const1[case3]

        self.rand_L, self.rand_k, self.rand_x0, self.rand_b = rand_L, rand_k, rand_x0, rand_b
        ENHANCE = self.L*self.k/k_default >= np.percentile(rand_L*rand_k/k_default, UPPER_BOUND)
        WEAKEN = self.L*self.k/k_default <= np.percentile(rand_L*rand_k/k_default, LOWER_BOUND)

        _, self.pvalue = ttest_1samp(rand_L, self.L)

        if ENHANCE or WEAKEN:
            self.is_change = True

        self._classify()
        self._report = f"The field is {self._ctype} with significance {_star(self.pvalue)} (pvalue {self.pvalue})"
        print(self._report)


    @property
    def report(self)->str:
        print(self._report)
        return self._report
    
    def _find_field_emerge_lap(self):
        if self.cal_events_rate is None or self.cal_events_time is None:
            return None

        trial_num = self.cal_events_num.shape[0]
        total_events_num = 0
        
        for i in range(trial_num):
            total_events_num += self.cal_events_num[i, -1]
            if total_events_num != 0:
                return i
        
        return trial_num
    
    def _find_field_disappear_lap(self):
        if self.cal_events_rate is None or self.cal_events_time is None:
            return None

        trial_num = self.cal_events_num.shape[0]
        total_events_num = 0
        
        for i in range(trial_num):
            total_events_num += self.cal_events_num[-i-1, -1]
            if total_events_num != 0:
                return trial_num-i
        
        return 0

    def get_info(self):
        return {
            'L': self.L,
            'k': self.k,
            'x0': self.x0,
            'b': self.b,
            'field_range': self.field_range,
            'effective laps': self.cal_events_num.shape[0],
            'durations': self.durations,
            'total_durations': self.total_durations,
            'total_events_num': self.total_events_num,
            'mean_rate': self.total_events_num / self.total_durations * 1000,
            'cal_events_num': self.cal_events_num,
            'cal_events_rate': self.cal_events_rate,
            'cal_events_time': self.cal_events_time,
            'cal_frames_num': self.cal_frames_num,
            'err_events': self.err_events,
            'is_change': self.is_change,
            'ctype': self._ctype,
            'pvalue': self.pvalue,
            'rand_L': self.rand_L,
            'rand_k': self.rand_k,
            'rand_x0': self.rand_x0,
            'rand_b': self.rand_b,
            'significance': _star(self.pvalue),
            'emerge lap': self._field_emerge_lap,
            'disappear lap': self._field_disappear_lap,
            'active lap percent': self._active_lap_percent,
            'shuffle times': self.shuffle_times
        }

    def visualize_shuffle_result(self, save_loc: str, file_name: str = "", k_default = 0.0005):
        if self.rand_L is None or self.rand_k is None or self.rand_b is None or self.rand_x0 is None:
            return
            raise ValueError("You should perform shuffle test first.")
            
        plt.figure(figsize = (6,4.5))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        b = ax.hist(self.rand_L*self.rand_k/k_default, bins=100, range=(-1,1), rwidth=0.8)[0]
        ax.axvline(self.L*self.k/k_default, color='red')
        ax.axvline(np.percentile(self.rand_L*self.rand_k/k_default, UPPER_BOUND), color = 'black')
        ax.axvline(np.percentile(self.rand_L*self.rand_k/k_default, LOWER_BOUND), color = 'black')
        y_max = np.nanmax(b)
        ax.set_xlim([-1,1])
        ax.set_ylim([0, y_max])
        ax.set_xticks(np.linspace(-1,1,11))
        ax.set_yticks(ColorBarsTicks(peak_rate=y_max, is_auto=True, tick_number=5))
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - two side L.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - two side L.svg'), dpi=600)
        plt.close()

        plt.figure(figsize = (6,4.5))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        b = ax.hist(self.rand_L, bins=100, range=(-1,1), rwidth=0.8)[0]
        ax.axvline(self.L, color='red')
        ax.axvline(np.percentile(self.rand_L, UPPER_BOUND), color = 'black')
        ax.axvline(np.percentile(self.rand_L, LOWER_BOUND), color = 'black')
        y_max = np.nanmax(b)
        ax.set_xlim([0,1])
        ax.set_ylim([0, y_max])
        ax.set_xticks(np.linspace(0,1,11))
        ax.set_yticks(ColorBarsTicks(peak_rate=y_max, is_auto=True, tick_number=5))
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - L.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - L.svg'), dpi=600)
        plt.close()

        plt.figure(figsize = (6,4.5))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        b = ax.hist((self.rand_x0 - np.min(self.cal_events_time))/(np.max(self.cal_events_time) - np.min(self.cal_events_time)), bins=self.cal_events_time.shape[0]-1, range=(0,1), rwidth=0.8, color = 'gray')[0]
        ax.axvline(self.x0, color='red')
        ax.set_xlim([0,1])
        ax.set_ylim([0, y_max*1.5])
        y_max = np.nanmax(b)
        ax.set_xticks(np.linspace(0,1,11))
        ax.set_yticks(ColorBarsTicks(peak_rate=y_max*1.5, is_auto=True, tick_number=5))
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - x0.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, file_name+'shuffle test - x0.svg'), dpi=600)
        plt.close()


    @staticmethod
    def calc_instant_rate(
        trace: dict,
        n: int,
        field: np.ndarray | list,
        t_thre: float = 500,
        t_unit: float = 1000 # ms
    ) -> tuple[np.ndarray, np.ndarray, list]:
        model = InFieldRateChangeModel()
        model.temporal_analysis(trace=trace, n=n, field=field, t_thre=t_thre,t_unit=t_unit)
        return model


    @staticmethod
    def analyze_field(
        trace: dict,
        n: int,
        field: np.ndarray | list,
        shuffle_times: int = 1000,
        signal_folder: list | np.ndarray = np.linspace(1,3,11),
        num_pieces_range: list | np.ndarray = [1,2],
        lam : float = 0,
        k_default: float = 0.0005,
        t_thre: float = 500,
        t_unit: float = 1000, # ms
        behav_time: np.ndarray | None = None,
        behav_nodes: np.ndarray | None = None,
        behav_pos: np.ndarray | None = None
    ):  
        #t1 = time.time()
        model = InFieldRateChangeModel()
        # = time.time()
        #print("Initialization time cost", t2-t1)
        model.temporal_analysis(
            field=field, 
            maze_type=cp.deepcopy(trace['maze_type']),
            t_thre=t_thre,
            t_unit=t_unit, 
            ms_time_original=cp.deepcopy(trace['ms_time']),
            deconv_signal=cp.deepcopy(trace['DeconvSignal'][n, :]),
            signal_folder=signal_folder, 
            behav_nodes=cp.deepcopy(behav_nodes), 
            behav_time=cp.deepcopy(behav_time)
        )
        #t3 = time.time()
        #print("temporal analysis time cost", t3-t2)
        model.fit(num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        #t4 = time.time()
        #print("fit time cost:", t4-t3)
        model.shuffle_test(shuffle_times=shuffle_times, num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        #t5 = time.time()
        #print("shuffle time cost:", t5-t4, end='\n\n')
        return model
