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

class InFieldRateChangeModel:
    def __init__(self) -> None:
        self.total_events_num = 0
        self.total_durations = 0
        
        self.cal_events_num = None
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

    def in_field_trajectory(
        self, 
        trace: dict, 
        field: list | np.ndarray, 
        t_thre: float = 500
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        behav_pos, behav_time, behav_nodes = Delete_NAN(
            cp.deepcopy(trace['correct_pos']), 
            cp.deepcopy(trace['correct_time']), 
            spike_nodes_transform(cp.deepcopy(trace['correct_nodes']), nx=12)
        ) 
        assert len(field) != 0

        idx = np.sort(np.concatenate([np.where(behav_nodes == i)[0] for i in field])) if len(field) > 1 else np.where(behav_nodes == field[0])[0]

        in_field_time = behav_time[idx].astype(np.float64)
        in_field_nodes = behav_nodes[idx].astype(np.float64)
        in_field_pos = behav_pos[idx, :]

        dt = np.ediff1d(in_field_time)
        break_idx = np.where(dt >= t_thre)[0]  # If The time gap is greater than a time threshold, insert a nan value as a gap.
        in_field_time = np.insert(in_field_time, break_idx+1, np.nan)
        in_field_nodes = np.insert(in_field_nodes, break_idx+1, np.nan)
        in_field_pos = np.insert(in_field_pos, break_idx+1, [np.nan, np.nan], axis = 0)

        return in_field_time, in_field_nodes, in_field_pos

    def temporal_analysis(self,
        trace: dict,
        n: int,
        field: np.ndarray | list,
        t_thre: float = 500,
        t_unit: float = 1000, # ms
        signal_folder: np.ndarray = np.linspace(1, 3, 11)
    ) -> tuple[np.ndarray, np.ndarray, list]:
        behav_time = cp.deepcopy(trace['correct_time'])
        behav_nodes = spike_nodes_transform(cp.deepcopy(trace['correct_nodes']), nx=12)
        behav_pos = cp.deepcopy(trace['correct_pos'])
        maze_type = trace['maze_type']
        ms_time_behav = cp.deepcopy(trace['ms_time_behav'])
        deconv_signal = cp.deepcopy(trace['DeconvSignal'][n, :])
        ms_time = cp.deepcopy(trace['ms_time'])

        # set field range
        field_range = set_range(maze_type=maze_type, field=field)

        # get in field data
        in_field_time, in_field_nodes, in_field_pos = self.in_field_trajectory(trace=trace, field=field, t_thre = t_thre)
        deconv_signal_behav = self._init_deconv_signal(deconv_signal=deconv_signal, ms_time_original=ms_time, ms_time_behav=ms_time_behav)

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
            if ENTER_INCORRECT_PATH:
                t_min, t_max = behav_time[prev_idx], behav_time[next_idx]
                err_events.append((field_range[0], field_range[1], t_min, t_max))
                continue

            INCLUDE_START_OR_END_POINT = StartPoints[maze_type] in field or EndPoints[maze_type] in field or prev_node == EndPoints[maze_type] or next_node == StartPoints[maze_type]
            TURN_AROUND_EVENT = prev_node == next_node and not INCLUDE_START_OR_END_POINT
            WRONG_DIRECTION = np.where(CP==prev_node)[0][0] > np.where(CP==next_node)[0][0] and not INCLUDE_START_OR_END_POINT

            if TOO_LESS_FRAME or TURN_AROUND_EVENT or WRONG_DIRECTION:
                t_min, t_max = behav_time[prev_idx], behav_time[next_idx]
                err_events.append((field_range[0], field_range[1], t_min, t_max))
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
            durations[i] = (cal_frames_num[i] + 1) * 0.033 * t_unit

            for j, thre in enumerate(signal_folder): # len(np.where(spikes[cal_events_indices]==1)[0]) #
                cal_events_num[i, j] = len(np.where(deconv_signal_behav[cal_events_indices]>=thre*self.signal_std)[0])
                cal_events_time[i, j] = (in_field_time[end] + in_field_time[beg])/2
                cal_events_rate[i, j] = cal_events_num[i, j] / durations[i] * 1000

        # norm
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

        self.L, self.k, self.x0, self.b = self._fit(self.cal_events_time, self.cal_events_rate, num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        return self.L, self.k, self.x0, self.b

    def shuffle_test(
        self,
        shuffle_times: int = 1000,
        num_pieces_range: list | np.ndarray=[1,2], 
        lam: float=0,
        k_default=0.0005, 
        is_draw: bool = False,
        save_loc: str = None
    ) -> None:
        if self.cal_events_rate is None or self.cal_events_time is None:
            return
            raise ValueError("Model should undergo temporal_analysis first!")

        if is_draw:
            assert save_loc is not None

        trial_num, thre_num = self.cal_events_num.shape[0], self.cal_events_num.shape[1]
        frames_set = np.concatenate([np.repeat(i, self.cal_frames_num[i]) for i in range(trial_num)])
        rand_L, rand_k, rand_x0, rand_b = np.zeros(shuffle_times), np.zeros(shuffle_times), np.zeros(shuffle_times), np.zeros(shuffle_times)
        
        total_events_num = int(self.total_events_num[-1])

        plt.figure(figsize = (1.5,4))
        ax = plt.axes()
        for n in range(shuffle_times):
            rand_events_num = np.zeros(trial_num)
            rand_events_frame = np.random.choice(frames_set, total_events_num, replace=False)
            
            for frame in rand_events_frame:
                rand_events_num[frame] += 1
    
            rand_events_rate = (rand_events_num / self.durations * 1000)
            
            rand_events_rate = rand_events_rate / np.nanmax(rand_events_rate)
            rand_events_rate[np.where(np.isnan(rand_events_rate))[0]] = 0
            
            """Previous version
            rand_events_num = np.zeros_like(self.cal_events_num)
            for i in range(thre_num):
                total_events_num = int(self.total_events_num[i])
            
                rand_events_frame = np.random.choice(frames_set, total_events_num, replace=False)

                
            rand_events_rate = (rand_events_num.T / self.durations * 1000).T
            
            rand_events_rate = rand_events_rate / np.nanmax(rand_events_rate, axis=0)
            rand_events_rate[np.where(np.isnan(rand_events_rate))[0]] = 0
            """
            
            # previous version of model
            # model = PiecewiseRegressionModel(self.cal_events_time.flatten(), rand_events_rate.flatten(), num_pieces_range, lam=lam, k_default=k_default)
            # model.fit()
            
            # Simply shuffle events rate.
            # rand_order = np.random.choice(np.arange(trial_num), trial_num, replace=False)
            # rand_events_rate = self.cal_events_rate[rand_order, :]
            rand_L[n], rand_k[n], rand_x0[n], rand_b[n] = self._fit(cal_events_rate=rand_events_rate, cal_events_time=self.cal_events_time[:, -1], k_default=k_default) 

            if is_draw:
                InstantRateCurveAxes(
                    ax = ax,
                    time_stamp=self.cal_events_time.flatten(),
                    content=rand_events_rate.flatten(),
                    folder = 0.000001,
                    smooth_window_length=30
                )
                plt.tight_layout()
                plt.savefig(os.path.join("G:\YSY\Cross_maze\PiecewiseRegression\shuffle_test", f'shuffle {n+1}.png'), dpi=600)
                plt.savefig(os.path.join("G:\YSY\Cross_maze\PiecewiseRegression\shuffle_test", f'shuffle {n+1}.svg'), dpi=600)
                ax.clear()
            
        plt.close()
        self.rand_L, self.rand_k, self.rand_x0, self.rand_b = rand_L, rand_k, rand_x0, rand_b
        ENHANCE = self.L*self.k/k_default >= np.percentile(rand_L*rand_k/k_default, UPPER_BOUND)
        WEAKEN = self.L*self.k/k_default <= np.percentile(rand_L*rand_k/k_default, LOWER_BOUND)

        _, self.pvalue = ttest_1samp(rand_L, self.L)

        if ENHANCE or WEAKEN:
            self.is_change = True

        self._classify()
        self._report = f"The field is {self._ctype} with significance {_star(self.pvalue)} (pvalue {self.pvalue})"

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
        b = ax.hist(self.rand_x0/(np.max(self.cal_events_time) - np.min(self.cal_events_time)), bins=10, range=(0,1), rwidth=0.8, color = 'gray')[0]
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
    def get_in_field_trajectory(
        trace: dict, 
        field: list | np.ndarray, 
        t_thre: float = 500
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = InFieldRateChangeModel()
        model.in_field_trajectory(trace=trace, field=field, t_thre=t_thre)
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
        is_draw: bool = False,
        save_loc: str | None = None
    ):
        model = InFieldRateChangeModel()
        model.temporal_analysis(trace=trace, n=n, field=field, t_thre=t_thre,t_unit=t_unit, signal_folder=signal_folder)
        model.fit(num_pieces_range=num_pieces_range, lam=lam, k_default=k_default)
        model.shuffle_test(shuffle_times=shuffle_times, num_pieces_range=num_pieces_range, lam=lam, k_default=k_default, is_draw=is_draw, save_loc=save_loc)
        return model

if __name__ == '__main__':
    import pickle
    import os
    from mylib import InstantRateCurveAxes
    from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d

    with open(r"G:\YSY\Cross_maze\11095\20220828\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    #[138, 126, 127, 115], [92, 80, 81]

    model = InFieldRateChangeModel.analyze_field(trace, 23-1, [10, 11, 12, 24, 23, 22, 34, 33, 32])
    model.visualize_shuffle_result(save_loc=r"G:\YSY\Cross_maze\PiecewiseRegression", file_name="")
    cal_events_rate, cal_events_time, err_events = model.cal_events_rate.flatten(), model.cal_events_time.flatten(), model.err_events
    #M = gaussian_smooth_matrix1d(temporal_stamp.shape[0], dis_stamp=temporal_stamp/1000, window = 6, folder=0.1)

    with open("G:\YSY\Cross_maze\PiecewiseRegressionShuffleTest.pkl", 'wb') as f:
        pickle.dump(model, f)


    plt.figure(figsize = (1.5,4))
    InstantRateCurveAxes(
        ax = plt.axes(),
        time_stamp=cal_events_time,
        content=cal_events_rate,
        folder = 0.000001,
        smooth_window_length=40
    )
    plt.tight_layout()
    plt.savefig(os.path.join("G:\YSY\Cross_maze\PiecewiseRegression", 'total.png'), dpi=600)
    plt.close()