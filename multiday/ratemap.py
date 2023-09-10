import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os
import pandas as pd
import scipy.stats

from mylib.maze_utils3 import spike_nodes_transform
from mylib.multiday.core import MultiDayCore

class MultiDayRatemapGenerator:
    def __init__(
        self,
        core: MultiDayCore | None = None,
        keys: list[str] | None = None,
        f: pd.DataFrame | None = None,
        file_indices: np.ndarray | None = None
    ) -> None:
        if core is None:
            self.core = MultiDayCore.get_core(
                f=f,
                file_indices=file_indices,
                keys=keys
            )
        else:
            self.core = core
    
    def concate_old_map(self) -> np.ndarray:
        if 'dt' not in self.core.res.keys():
            raise ValueError("Do not detect 'dt' in core.res")
        
        if 'spikes' not in self.core.res.keys():
            raise ValueError("Do not detect 'spikes' in core.res")
        
        if 'spike_nodes' not in self.core.res.keys():
            raise ValueError("Do not detect 'spike_nodes' in core.res")
        
        _nbins = 144
        _coords_range = [0, _nbins +0.0001 ]
        
        self.dt = np.concatenate(self.core.res['dt'])
        self.spike_nodes = np.concatenate(self.core.res['spike_nodes'])
        self.spikes = np.concatenate(self.core.res['spikes'])

        self.occu_time, _, _ = scipy.stats.binned_statistic(
            self.spike_nodes,
            self.dt,
            bins=_nbins,
            statistic="sum",
            range=_coords_range
        )    
        
        events_num = np.zeros(self.occu_time.shape[0], dtype=np.int64)
        for i in range(self.occu_time.shape[0]):
            events_num[i] = len(np.where((self.spike_nodes == i+1)&(self.spikes == 1))[0])
        
        self.old_map = events_num / self.occu_time * 1000
        self.old_map[np.where(np.isnan(self.old_map))[0]] = 0
        return self.old_map
    
    def concate_old_map(self, Ms: np.ndarray) -> np.ndarray:
        if 'dt' not in self.core.res.keys():
            raise ValueError("Do not detect 'dt' in core.res")
        
        if 'spikes' not in self.core.res.keys():
            raise ValueError("Do not detect 'spikes' in core.res")
        
        if 'spike_nodes' not in self.core.res.keys():
            raise ValueError("Do not detect 'spike_nodes' in core.res")
        
        _nbins = 2304
        
        _coords_range = [0, _nbins +0.0001 ]
        
        self.dt = np.concatenate(self.core.res['dt'])
        self.spike_nodes = np.concatenate(self.core.res['spike_nodes'])
        self.spikes = np.concatenate(self.core.res['spikes'])

        self.occu_time, _, _ = scipy.stats.binned_statistic(
            self.spike_nodes,
            self.dt,
            bins=_nbins,
            statistic="sum",
            range=_coords_range
        )    
        
        events_num = np.zeros(self.occu_time.shape[0], dtype=np.int64)
        for i in range(self.occu_time.shape[0]):
            events_num[i] = len(np.where((self.spike_nodes == i+1)&(self.spikes == 1))[0])
        
        self.old_map = events_num / self.occu_time * 1000
        self.old_map[np.where(np.isnan(self.old_map))[0]] = 0
        return self.old_map