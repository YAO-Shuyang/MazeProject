import numpy as np
import pickle
import sklearn.preprocessing
from tqdm import tqdm
import pandas as pd

from mylib.behavior.correct_rate import calc_behavioral_score as Dr_cis
from mylib.behavior.correct_rate import calc_behavioral_score_trs as Dr_trs
from mylib.maze_utils3 import GetDMatrices, maze_graphs
from mylib.divide_laps.lap_split import LapSplit
from mazepy.datastruc.neuact import SpikeTrain
from mazepy.datastruc.variables import VariableBin

import copy as cp
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
"""
1. Behavioral Data:
    A. Overral behavioral performance.
    B. Amount of active time spent within the field.
    C. Total session mice have been trained.
2. Session Interval (Passage of time):
3. Field Properties:
    A. Peak Rate
    B. Within-session Half-Half Stability.
    C. First-Lap field appear.
    D. First Transient vs. Later Transient.
    E. First-lap center position vs field center position.
    F. Degree of Fluctuation.
"""

class GLMParameters:
    def __init__(
        self, 
        f: pd.DataFrame,
        f_indices: np.ndarray,
        trace_mdays: np.ndarray,
        direction: str = None
    ) -> None:
        """
        This class is designed to extract parameters that a GLM
        will consider from data in separated trace files and integrated
        them into a unified matrix for the GLM to predict.
        
        Parameters:
        -----------
        f: pd.DataFrame
            The dataframe containing the trace files.
        f_indices: np.ndarray
            The indices of the trace files in the dataframe.
        """
        self._f = f
        self._f_indices = f_indices
        self._paradigm = f['behavior_paradigm'][f_indices[0]]
        self.mouse = f['MiceID'][f_indices[0]]
        self._direction = direction
        self._maze_type = f['maze_type'][f_indices[0]]
        
        self.D = GetDMatrices(self._maze_type, 48)
        self.G = maze_graphs[(self._maze_type, 48)]
        
        self._traces = []
        print("A. Read Trace Files.")
        for i in tqdm(self._f_indices):
            with open(f['Trace File'][i], 'rb') as handle:
                trace = pickle.load(handle)
                
            self._traces.append(self._preprocess_trace(trace))
            del trace
            
        self._trace_m = trace_mdays
        
        if self._paradigm != "CrossMaze":
            if direction is None:
                raise ValueError(f"Please specify a direction for {self._paradigm} paradigm.")
            elif direction not in ["cis", "trs"]:
                raise ValueError(f"{direction} is not a valid direction for {self._paradigm} paradigm."
                                 f" Please choose either 'cis' or 'trs'.")
            self._trace_m = {
                "field_reg": cp.deepcopy(trace_mdays[direction]["field_reg"]),
                "field_info": cp.deepcopy(trace_mdays[direction]["field_info"]),
                "place_field_all": cp.deepcopy(trace_mdays[direction]["place_field_all"])
            }
        else:
            self._trace_m = {
                "field_reg": cp.deepcopy(trace_mdays["field_reg"]),
                "field_info": cp.deepcopy(trace_mdays["field_info"]),
                "place_field_all": cp.deepcopy(trace_mdays["place_field_all"])
            }
        
        date_obj = np.array([datetime.strptime(str(date), '%Y%m%d') for date in f['date'][f_indices]])
        self.session_intervals = np.diff(date_obj).astype('timedelta64[D]').astype(int)
    
    def _preprocess_trace(self, trace):
        if self._paradigm == "CrossMaze":
            trace_condensed = {
                'maze_type': cp.deepcopy(trace['maze_type']),
                'correct_time': cp.deepcopy(trace['correct_time']),
                'correct_nodes': cp.deepcopy(trace['correct_nodes']),
                'Spikes': cp.deepcopy(trace['Spikes']),
                'ms_time_behav': cp.deepcopy(trace['ms_time_behav']),
                'spike_nodes': cp.deepcopy(trace['spike_nodes']),
                'occu_time_spf': cp.deepcopy(trace['occu_time_spf']),
                'smooth_map_all': cp.deepcopy(trace['smooth_map_all']),
                'smooth_map_fir': cp.deepcopy(trace['smooth_map_fir']),
                'smooth_map_sec': cp.deepcopy(trace['smooth_map_sec']),
                "RawTraces": cp.deepcopy(trace["RawTraces"]),
                "ms_time": cp.deepcopy(trace["ms_time"]),
                "spike_nodes_original": cp.deepcopy(trace["spike_nodes_original"])
            }
            beg, end = LapSplit(trace, self._paradigm)
            trace_condensed['lap beg time'] = trace['correct_time'][beg]
            trace_condensed['lap end time'] = trace['correct_time'][end] 
            
            ms_beg, ms_end = np.zeros(beg.shape[0], np.int64), np.zeros(end.shape[0], np.int64)
            smooth_map_lap = np.zeros((trace['smooth_map_all'].shape[0], trace['smooth_map_all'].shape[1], end.shape[0]))
            for i in range(end.shape[0]):
                idx = np.where(
                    (trace['ms_time_behav'] >= trace_condensed['lap beg time'][i]) & 
                    (trace['ms_time_behav'] <= trace_condensed['lap end time'][i])
                )[0]
                ms_beg[i] = idx[0]
                ms_end[i] = idx[-1]
                spike_train = SpikeTrain(
                    activity=trace['Spikes'][:, idx],
                    time=trace['ms_time_behav'][idx],
                    variable=VariableBin(trace['spike_nodes'][idx].astype(np.int64))-1,
                )
                firing_rate = spike_train.calc_tuning_curve(
                    nbins=2304,
                    t_interv_limits=100
                ).smooth(trace['Ms'].T)
                smooth_map_lap[:, :, i] = np.asarray(firing_rate, np.float64)
                
            trace_condensed['smooth_map_lap'] = smooth_map_lap
            trace_condensed['ms_beg'] = ms_beg
            trace_condensed['ms_end'] = ms_end
            return trace_condensed
        else:
            trace_condensed = {
                'maze_type': cp.deepcopy(trace['maze_type']),
                'Spikes': cp.deepcopy(trace[self._direction]['Spikes']),
                'ms_time_behav': cp.deepcopy(trace[self._direction]['ms_time_behav']),
                'spike_nodes': cp.deepcopy(trace[self._direction]['spike_nodes']),
                'occu_time_spf': cp.deepcopy(trace[self._direction]['occu_time_spf']),
                'smooth_map_all': cp.deepcopy(trace[self._direction]['smooth_map_all']),
                'smooth_map_fir': cp.deepcopy(trace[self._direction]['smooth_map_fir']),
                'smooth_map_sec': cp.deepcopy(trace[self._direction]['smooth_map_sec'])
            }
            
            beg, end = LapSplit(trace, self._paradigm)
            
            if self._paradigm == "ReverseMaze":
                A = self._direction == "trs" and self.mouse in [10209, 10212]
                B = self._direction == "cis" and self.mouse in [10224, 10227]
                if A or B:
                    beg, end = beg[np.arange(0, beg.shape[0], 2)], end[np.arange(0, end.shape[0], 2)]
                else:
                    beg, end = beg[np.arange(1, beg.shape[0], 2)], end[np.arange(1, end.shape[0], 2)]
            else:
                if self._direction == "cis":
                    beg, end = beg[np.arange(0, beg.shape[0], 2)], end[np.arange(0, end.shape[0], 2)]
                else:
                    beg, end = beg[np.arange(1, beg.shape[0], 2)], end[np.arange(1, end.shape[0], 2)]
            
            if self.mouse == 10227 and self._paradigm == "ReverseMaze":
                # A patch for mouse 10227 S1 the last lap of 'trs'. I think the lap onset and end are incorrectly labeled.
                idx = np.where(
                    (trace[self._direction]['ms_time_behav'] >= trace['correct_time'][9]) & 
                    (trace[self._direction]['ms_time_behav'] <= trace['correct_time'][9])
                )[0]
                if idx.shape[0] == 0:
                    beg, end = np.delete(beg, 9), np.delete(end, 9)
                
            trace_condensed['lap beg time'] = trace['correct_time'][beg]
            trace_condensed['lap end time'] = trace['correct_time'][end]
            trace_condensed['correct_time'] = trace['correct_time'][np.concatenate([np.arange(beg[i], end[i]+1) for i in range(beg.shape[0])])]
            trace_condensed['correct_nodes'] = trace['correct_nodes'][np.concatenate([np.arange(beg[i], end[i]+1) for i in range(beg.shape[0])])]
            
            ms_beg, ms_end = np.zeros(beg.shape[0], dtype=np.int64), np.zeros(end.shape[0], dtype=np.int64)
            smooth_map_lap = np.zeros((trace[self._direction]['smooth_map_all'].shape[0], 
                                       trace[self._direction]['smooth_map_all'].shape[1], end.shape[0]))
            for i in range(beg.shape[0]):
                idx = np.where(
                    (trace[self._direction]['ms_time_behav'] >= trace_condensed['lap beg time'][i]) & 
                    (trace[self._direction]['ms_time_behav'] <= trace_condensed['lap end time'][i])
                )[0]
                if idx.shape[0] == 0:
                    print(trace_condensed['lap beg time'][i], trace_condensed['lap end time'][i],
                          len(idx))

                ms_beg[i] = idx[0]
                ms_end[i] = idx[-1]            
                
                spike_train = SpikeTrain(
                    activity=trace[self._direction]['Spikes'][:, idx],
                    time=trace[self._direction]['ms_time_behav'][idx],
                    variable=VariableBin(trace[self._direction]['spike_nodes'][idx].astype(np.int64))-1,
                )
                firing_rate = spike_train.calc_tuning_curve(
                    nbins=2304,
                    t_interv_limits=100
                ).smooth(trace['Ms'].T)
                smooth_map_lap[:, :, i] = np.asarray(firing_rate, np.float64)
            
            trace_condensed['smooth_map_lap'] = smooth_map_lap
            trace_condensed['ms_beg'] = ms_beg
            trace_condensed['ms_end'] = ms_end
                
            idx = np.concatenate([
                np.where(
                    (trace['ms_time'] >= trace_condensed['lap beg time'][i]) & 
                    (trace['ms_time'] <= trace_condensed['lap end time'][i])
                )[0] for i in range(beg.shape[0])
            ])
            trace_condensed['RawTraces'] = trace['RawTraces'][:, idx]
            trace_condensed['ms_time'] = trace['ms_time'][idx]
            trace_condensed['spike_nodes_original'] = trace['spike_nodes_original'][idx]
            
            return trace_condensed
    
    def get_behavior_data(self):
        print(f"B. Get Behavior Data.")
        field_reg = self._trace_m['field_reg']
        field_info = self._trace_m['field_info']
        
        behav_dims = np.zeros((field_reg.shape[0], field_reg.shape[1], 3))
        # Total Sessions
        behav_dims[:, :, 1] = np.repeat(np.arange(1, field_reg.shape[0]+1)[:, np.newaxis], field_reg.shape[1], axis=1)
        
        # Correct Decision Rate
        scores = np.zeros(len(self._traces), np.float64)
        for i, trace in enumerate(self._traces):
            if self._paradigm == "CrossMaze":
                err_num, pass_num, _ = Dr_cis(trace)
            elif self._paradigm == "ReverseMaze":
                if self._direction == "cis":
                    err_num, pass_num, _ = Dr_cis(trace)
                else:
                    err_num, pass_num, _ = Dr_trs(trace)
            else:
                err_num, pass_num = 0, 1
                    
            scores[i] = 1 - err_num / pass_num
        
        behav_dims[:, :, 0] = np.repeat(scores[:, np.newaxis], field_reg.shape[1], axis=1)
        
        # Session Interval (Passage of time)
        behav_dims[:-1, :, 2] = np.repeat(self.session_intervals[:, np.newaxis], field_reg.shape[1], axis=1)
        
        return behav_dims
    
    def _BFS4Field(self, smooth_map: np.ndarray, field_center: int) -> np.ndarray:
        """Extract a field from its centers"""   
        thre = 0.2
        
        field_center0 = field_center
        
        field_area = [field_center]
        bound_bins = [field_center]
        peak_rate = smooth_map[field_center-1]
        
        while len(bound_bins) > 0:
            neighbors = self.G[bound_bins[0]]
            
            for neighbor in neighbors:
                is_qulified = (
                    smooth_map[neighbor-1] >= peak_rate or  # peak is not the real peak.
                    smooth_map[neighbor-1] <= smooth_map[bound_bins[0]-1] # Gradient descends.
                ) 
                is_enough = smooth_map[neighbor-1] >= thre or len(field_area) < 16 or smooth_map[neighbor-1] >= peak_rate
                if is_enough and is_qulified and neighbor not in field_area:
                    field_area.append(neighbor)
                    bound_bins.append(neighbor)
                
            bound_bins.pop(0)
        
        field_area = np.array(field_area, dtype=np.int64)
        field_center = field_area[np.argmax(smooth_map[field_area-1])]
        
        if field_center != field_center0:
            return self._BFS4Field(smooth_map, field_center)
        else:
            return field_area, field_center
         
    def get_field_property(self):
        field_reg = self._trace_m['field_reg']
        field_info = self._trace_m['field_info']
        
        field_dims = np.full((field_reg.shape[0], field_reg.shape[1], 8), np.nan, np.float64)
        
        # Time Spent Within Each Field
        print(f"C. Register Field Properties.")
        for j in tqdm(range(field_reg.shape[1])):
            field_area = None
            prev_session_with_field = None
            for i in range(field_reg.shape[0]):
                cell = int(field_info[i, j, 0])
                if field_reg[i, j] == 1:
                    prev_session_with_field = i
                    field_center = int(field_info[i, j, 2])
                    field_area, field_center = self._BFS4Field(
                        self._traces[i]['smooth_map_all'][cell-1, :], field_center
                    )
                    
                    # Field States
                    field_dims[i, j, 0] = 1
                    
                    # Occupied Time
                    field_dims[i, j, 1] = np.nansum(self._traces[i]['occu_time_spf'][field_area-1])/1000
                    
                    # Peak Rate
                    field_dims[i, j, 2] = self._traces[i]['smooth_map_all'][cell-1, field_center-1]
                    
                    # Half-Half Correlation
                    if field_area.shape[0] >= 8:
                        field_dims[i, j, 3] = pearsonr(
                            x = self._traces[i]['smooth_map_fir'][cell-1, field_area-1],
                            y = self._traces[i]['smooth_map_sec'][cell-1, field_area-1],
                        )[0]
                    
                    # Formation Lap
                    n_events = np.zeros(self._traces[i]['smooth_map_lap'].shape[2])
                    idx = np.array([np.where(
                        (self._traces[i]['ms_time_behav'] >= self._traces[i]['lap beg time'][k])
                    )[0][0] for k in range(n_events.shape[0])] + [self._traces[i]['ms_time_behav'].shape[0]])
                    for k in range(n_events.shape[0]):
                        beg, end = idx[k], idx[k+1]-1
                        n_events[k] = np.where(
                            (self._traces[i]['Spikes'][cell-1, beg:end+1] == 1) & 
                            (np.isin(self._traces[i]['spike_nodes'][beg:end+1], field_area))
                        )[0].shape[0]
                    
                    with_event_laps = np.where(n_events != 0)[0]
                    # If with event laps < 4
                    if with_event_laps.shape[0] < 4:
                        continue
                    field_dims[i, j, 4] = with_event_laps[0]
                    
                    # First Transient vs Later Transient Peak
                    idx = np.array([np.where(
                        (self._traces[i]['ms_time'] >= self._traces[i]['lap beg time'][k])
                    )[0][0] for k in range(n_events.shape[0])] + [self._traces[i]['ms_time'].shape[0]])
                    peak_transient = np.zeros(n_events.shape[0]) * np.nan
                    for k in range(n_events.shape[0]):
                        if k not in with_event_laps:
                            continue
                        beg, end = idx[k], idx[k+1]-1
                        
                        aidx = np.where(
                            (np.isin(self._traces[i]['spike_nodes_original'][beg:end+1], field_area))
                        )[0]
                        
                        if aidx.shape[0] == 0:
                            with_event_laps = with_event_laps[with_event_laps != k]
                            continue
                        else:
                            peak_transient[k] = np.max(self._traces[i]['RawTraces'][cell-1, aidx])
                    
                    fir_transient = peak_transient[with_event_laps[0]]
                    later_transient = peak_transient[with_event_laps[1:]]
                    
                    field_dims[i, j, 5] = fir_transient / np.nanmean(later_transient)
                    
                    # First Position vs Later positions
                    peak_centers = np.argmax(
                        self._traces[i]['smooth_map_lap'][cell-1, field_area-1, :][:, with_event_laps],
                        axis = 0
                    )
                    dis = self.D[peak_centers, 0]
                    field_dims[i, j, 6] = dis[0] - np.mean(dis[1:])
                    
                    # Fluctuation
                    field_dims[i, j, 7] = np.std(dis)
                elif field_reg[i, j] == 0:
                    # Field States
                    field_dims[i, j, 0] = 0
            
                    if field_area is not None:
                        if prev_session_with_field is None:
                            assert False
                        
                        field_dims[i, j, 1:] = field_dims[prev_session_with_field, j, 1:]
        return field_dims
    
    @staticmethod
    def process(
        f: pd.DataFrame,
        f_indices: np.ndarray,
        trace_mdays: np.ndarray,
        direction: str = None
    ):
        """
        Returns
        -------
        np.ndarray
            The concatenated behavior and field dimensions.
            
            0    Decision Rate
            1    Session that The Field is Current At
            2    Intervals to the next session
            3    Field Observable States
            4    Occupied Time Within Field
            5    Peak Rate
            6    Within-session Half-Half Stability
            7    First-Lap field appear
            8    First Transient vs. Later Transient
            9    Field Shift: First-lap center position vs field center position
            10   Degree of Fluctuation
        """
        processor = GLMParameters(f, f_indices, trace_mdays, direction=direction)
        behav_dims = processor.get_behavior_data()
        field_dims =  processor.get_field_property()
        return np.concatenate([behav_dims, field_dims], axis=2)
    
import numpy as np
import statsmodels.api as sm
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

class GLM:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.model = None
        self.results = None
    
    def fit(self):        
        # Fit the GLM model using a binomial family
        self.model = sm.GLM(self.y, self.X, family=sm.families.Binomial())
        self.results = self.model.fit()

    def transform(self, X_test: np.ndarray):
        return self.results.predict(X_test)

    def calc_loss(self, y_test: np.ndarray, predicted_p: np.ndarray):
        # Calculate negative log-likelihood loss
        loss = -np.mean(y_test * np.log(predicted_p) + (1 - y_test) * np.log(1 - predicted_p))
        return loss

if __name__ == '__main__':
    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace_mdays = pickle.load(handle)
        
    from mylib.field.tracker_v2 import Tracker2d
    
    def delete_column(X, col):
        return np.delete(X, col, axis=1)
    """
    X, Y = Tracker2d.convert_for_glm(trace_mdays['field_reg'], trace_mdays['GLM'])
    X = delete_column(X, 1)
    print(X.shape, Y.shape)
    X[:, :-1] = sklearn.preprocessing.normalize(X[:, :-1], axis=0)
    # Split the data
    split_ratio = 0.8
    idx = np.random.choice(np.arange(X.shape[0]), int(X.shape[0] * split_ratio), replace=False)
    test_idx = np.setdiff1d(np.arange(X.shape[0]), idx)
    
    X_train, X_test, Y_train, Y_test = X[idx, :][:, :-1], X[test_idx, :][:, :-1], Y[idx], Y[test_idx]
    model = GLM(X_train, Y_train)
    model.fit()
    
    loss = model.calc_loss(Y_test, model.transform(X_test))
    print(model.results.params)
    print(model.results.pvalues)
    print(f"loss: {loss}")
    
    loss_array = np.full(26, np.nan)
    
    for i in range(25):
        idx = np.where(X[:, -1] == i)[0]
        X_i = X[idx, :-1]
        Y_i = Y[idx]
        
        # Use the model to calculate loss on X_i
        loss_array[i] = model.calc_loss(Y_i, model.transform(X_i))
        
    plt.plot(loss_array)
    plt.show()
    """
    
    