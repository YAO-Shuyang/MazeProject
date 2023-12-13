import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import copy as cp
from mylib.maze_graph import S2F
from mylib.calcium.field_criteria import place_field
import gc

class MultiDayCore:
    def __init__(
        self,
        keys: list[str] | None = ['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                  'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type'],
        interv_time: float = 50000
    ) -> None:
                
        self._res = {}        
        for k in keys:
            self._res[k] = []
        
        self.interv_time = interv_time

    @property
    def num(self):
        """
        Get the value of the 'num' property.

        Returns
        -------
            The value of the 'num' property.
        """
        return self._num
        
    def get_trace_set(
        self,
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        keys: list[str] | None = ['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                  'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type']
    ) -> dict:
        """
        Generate the function comment for the given function body in a markdown 
        code block with the correct language syntax.
    
        Parameters
        ----------
        f : pd.DataFrame
            A pandas DataFrame.
        file_indices : np.ndarray
            A list or numpy array of file indices.
    
        Returns
        -------
        list
            A list containing the trace set.
    
        Notes
        -----
        This function iterates over each row in the DataFrame `f` and checks if the corresponding trace file exists. If the file exists, it loads the trace file using pickle and appends the trace to the trace set. Finally, it returns the trace set.
        """
        self._len_f = len(file_indices)
        
        print("    Getting trace...")
        base = 0
        for i in tqdm(file_indices):
            if os.path.exists(f['Trace File'][i]):
                with open(f['Trace File'][i], 'rb') as handle:
                    trace = pickle.load(handle)
                    
                t_max = 0
                for k in keys:
                    if k in ['ms_time', 'ms_time_behav', 'behav_time', 'correct_time']:
                        t_max = max(t_max, trace[k][-1])
                
                    if k in ['ms_time', 'behav_time', 'correct_time', 'ms_time_behav']:
                        self._res[k].append(trace[k]+base)
                    elif k == 'place_field_all_multiday': 
                        self._res[k].append(trace[k])
                    else:
                        self._res[k].append(trace[k])
            else:
                print(f['Trace File'][i], " is not existed!")
                for k in keys:
                    self._res[k].append([])
            
            base = base + t_max + self.interv_time
            del trace
            gc.collect()
            
        self.t_max = base/1000
        return self._res
    
    def concatenate_params(self, cell_indices: np.ndarray) -> None:
        """
        Concatenate the trace set and cell indices into a list of dicts.
    
        Parameters
        ----------
        trace_set : list[dict]
            A list of dicts containing the traces.
        cell_indices : np.ndarray
            A list or numpy array of cell indices.
        """
        try:
            assert self._len_f == len(cell_indices)
        except:
            raise ValueError(f'file_indices {self._len_f} and cell_indices {cell_indices.shape} must have the same length')
        
        try:
            assert len(cell_indices) > 1
        except:
            raise ValueError('cell_indices must have more than 1 element, or use LocTimeCurve instead.')
        
        
        self._num = len(cell_indices)
        self.cell_indices = cell_indices
        session_num = len(cell_indices)
        
        # Need behav_nodes, behav_times, behav_positions, spikes, ms_time_behav, smooth_maps, DeconvSignal, ms_time_original, place fields
        behav_nodes = []
        behav_times = []
        behav_positions = []
        spikes = []
        ms_time_behav = []
        
        behav_nodes_all = np.array([], dtype=np.int32)
        behav_times_all = np.array([], dtype=np.int32)
        ms_time_behav_all = np.array([], dtype=np.int32)
        spikes_all = np.array([], dtype=np.int32)
        smooth_maps = np.zeros((session_num, 2304), dtype=np.float32)
        old_maps = np.zeros((session_num, 144), dtype=np.float32)
        smooth_maps_info = np.zeros((session_num, 3), dtype=np.float32)
        DeconvSignal = np.array([], dtype=np.float32)
        ms_time_original = np.array([], dtype=np.int32)
        place_fields = []
        
        for i in range(session_num):
            if self.cell_indices[i] != 0:
                behav_nodes.append(S2F[cp.deepcopy(self._res['correct_nodes'][i].astype(np.int32))-1])
                behav_times.append(cp.deepcopy(self._res['correct_time'][i]))
                behav_positions.append(cp.deepcopy(self._res['correct_pos'][i]))
                
                behav_nodes_all = np.concatenate([behav_nodes_all, S2F[cp.deepcopy(self._res['correct_nodes'][i].astype(np.int32))-1]])
                behav_times_all = np.concatenate([behav_times_all, cp.deepcopy(self._res['correct_time'][i])])
                spikes.append(self._res['Spikes'][i][int(self.cell_indices[i])-1, :])
                ms_time_behav.append(cp.deepcopy(self._res['ms_time_behav'][i]))
                
                spikes_all = np.concatenate([spikes_all, self._res['Spikes'][i][int(self.cell_indices[i])-1, :]])
                ms_time_behav_all = np.concatenate([ms_time_behav_all, self._res['ms_time_behav'][i]])
                smooth_maps[i, :] = self._res['smooth_map_all'][i][int(self.cell_indices[i])-1, :]
                old_maps[i, :] = self._res['old_map_clear'][i][int(self.cell_indices[i])-1, :]
                smooth_maps_info[i, 0] = self._res['SI_all'][i][int(self.cell_indices[i])-1]
                smooth_maps_info[i, 1] = np.nanmax(smooth_maps[i, :])
                smooth_maps_info[i, 2] = self._res['is_placecell'][i][int(self.cell_indices[i])-1]
                DeconvSignal = np.concatenate([DeconvSignal, self._res['DeconvSignal'][i][int(self.cell_indices[i])-1, :]])
                ms_time_original = np.concatenate([ms_time_original, self._res['ms_time'][i]])
                place_fields.append(self._res['place_field_all_multiday'][i][int(self.cell_indices[i])-1])
            else:
                behav_nodes.append(None)
                behav_times.append(None)
                behav_positions.append(None)
                spikes.append(None)
                ms_time_behav.append(None)
                place_fields.append(None)
        
            
        self.behav_nodes_list = cp.deepcopy(behav_nodes)
        self.behav_times_list = cp.deepcopy(behav_times)
        self.behav_positions_list = cp.deepcopy(behav_positions)
        self.spikes_list = cp.deepcopy(spikes)
        self.ms_time_behav_list = cp.deepcopy(ms_time_behav)
        self.place_fields_list = cp.deepcopy(place_fields)
        
        self.smooth_maps = smooth_maps
        self.old_maps = old_maps
        self.smooth_maps_info = smooth_maps_info
        self.maze_type = self._res['maze_type'][0]
        
        self.behav_nodes_all = behav_nodes_all
        self.behav_times_all = behav_times_all
        self.ms_time_original_all = ms_time_original
        self.deconv_signal_all = DeconvSignal
        self.ms_time_behav_all = ms_time_behav_all
        self.spikes_all = spikes_all
    
    @property
    def res(self) -> dict:
        return self._res
    
    @staticmethod
    def concat_core(
        f: pd.DataFrame,
        file_indices: np.ndarray,
        cell_indices: np.ndarray,
        core: None = None,
        keys: list[str] | None = ['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'old_map_clear', 'SI_all', 'is_placecell', 
                                  'DeconvSignal', 'ms_time', 'place_field_all', 'maze_type'],
        interv_time: float = 50000
    ):  
        if core is None:
            core = MultiDayCore(keys=keys, interv_time=interv_time)
            core.get_trace_set(f, file_indices, keys=keys)
            
        core.concatenate_params(cell_indices)
        return core
        
    
    @staticmethod
    def get_core(
        f: pd.DataFrame,
        file_indices: np.ndarray,
        keys: list[str] | None = None,
        interv_time: float = 50000
    ):
        core = MultiDayCore(
            keys=keys,
            interv_time=interv_time
        )
        core.get_trace_set(f, file_indices, keys=keys)
        return core