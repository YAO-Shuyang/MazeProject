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
from mylib.field.in_field import InFieldRateChangeModel, set_range, temporal_analysis, parallel_shuffle

LOWER_BOUND = 5
UPPER_BOUND = 95
CORRECT_PASS = True
FAIL_TO_PASS = False

from numba import jit
import time


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

class MultiDayInFieldRateChangeModel:
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
        
        
    def concat_behav(
        self,
        behav_nodes1: np.ndarray,
        behav_nodes2: np.ndarray,
        behav_time1: np.ndarray,
        behav_time2: np.ndarray,
        base_time: float
    ):
        """
        Concatenates two arrays of behavior nodes and two arrays of behavior times, and returns the concatenated arrays.

        Parameters:
            behav_nodes1 (np.ndarray): The first array of behavior nodes.
            behav_nodes2 (np.ndarray): The second array of behavior nodes.
            behav_time1 (np.ndarray): The first array of behavior times.
            behav_time2 (np.ndarray): The second array of behavior times.
            base_time (float): The base time value.

        Returns:
            np.ndarray: The concatenated array of behavior nodes.
            np.ndarray: The concatenated array of behavior times.
        """
        return np.concatenate([behav_nodes1, behav_nodes2]), np.concatenate([behav_time1, behav_time2 + base_time])

    def concat_image(
        self,
        deconv_signal1: np.ndarray,
        deconv_signal2: np.ndarray,
        ms_time_original1: np.ndarray,
        ms_time_original2: np.ndarray,
        base_time: float
    ):
        """
        Concatenates two image signals and their corresponding time arrays.

        Parameters
        ----------
        deconv_signal1 : np.ndarray
            The first image signal to be concatenated.
        deconv_signal2 : np.ndarray
            The second image signal to be concatenated.
        ms_time_original1 : np.ndarray
            The time array corresponding to the first image signal.
        ms_time_original2 : np.ndarray
            The time array corresponding to the second image signal.
        base_time : float
            The base time to be added to the second time array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the concatenated image signals and their corresponding time arrays.
        """
        return np.concatenate([deconv_signal1, deconv_signal2]), np.concatenate([ms_time_original1, ms_time_original2 + base_time])

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
        fir_trace: dict,
        i: int,
        sec_trace: dict,
        j:int,
        behav_nodes1: np.ndarray,
        behav_nodes2: np.ndarray,
        behav_time1: np.ndarray,
        behav_time2: np.ndarray,
        field: np.ndarray | list,
        shuffle_times: int = 10000,
        signal_folder: list | np.ndarray = np.linspace(1,3,11),
        num_pieces_range: list | np.ndarray = [1,2],
        lam : float = 0,
        k_default: float = 0.0005,
        t_thre: float = 500,
        t_unit: float = 1000, # ms
        behav_dir: int = 1
    ):  
        model = MultiDayInFieldRateChangeModel()

        behav_nodes, behav_time = model.concat_behav(
            behav_nodes1=cp.deepcopy(behav_nodes1),
            behav_nodes2=cp.deepcopy(behav_nodes2),
            behav_time1=cp.deepcopy(behav_time1),
            behav_time2=cp.deepcopy(behav_time2),
            base_time=behav_time1[-1]+10000
        )
        
        deconv_signal, ms_time = model.concat_image(
            cp.deepcopy(fir_trace['DeconvSignal'][i, :]),
            cp.deepcopy(sec_trace['DeconvSignal'][j, :]),
            ms_time_original1=cp.deepcopy(fir_trace['ms_time']),
            ms_time_original2=cp.deepcopy(sec_trace['ms_time']),
            base_time=behav_time1[-1]+10000
        )
        
        model.temporal_analysis(
            field=field, 
            maze_type=fir_trace['maze_type'],
            t_thre=t_thre,
            t_unit=t_unit, 
            ms_time_original=ms_time,
            deconv_signal=deconv_signal,
            signal_folder=signal_folder, 
            behav_nodes=behav_nodes, 
            behav_time=behav_time
        )
        model.fit(num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        #t4 = time.time()
        #print("fit time cost:", t4-t3)
        model.shuffle_test(shuffle_times=shuffle_times, num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        return model