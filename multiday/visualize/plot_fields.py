from mylib.multiday.core import MultiDayCore
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mylib.maze_utils3 import Clear_Axes
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import copy as cp
import pandas as pd
import os
from mylib import LocTimeCurveAxes
from mylib.field.in_field import InFieldRateChangeModel, set_range
from mylib.maze_graph import correct_paths
from mylib.maze_utils3 import spike_nodes_transform, SF2FF
import seaborn as sns


class MultiDayFields:
    def __init__(
        self, 
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 80000,
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        field_reg: np.ndarray | None = None,
        field_info: np.ndarray | None = None,
        place_field_all: list[dict] | None = None
    ) -> None:
        if core is None:
            self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
        else:
            self.core = core
         
        self.cell_indices = cell_indices
        
        if figsize is None:
            figsize = (8, self.core.num*2)
            
        fig = plt.figure(figsize=figsize)
        self.ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        
        self.field_reg = field_reg
        self.field_info = field_info
        self.place_field_all = place_field_all
        
    def _add_loc_time_curve(
        self,
        ax: Axes,
        behav_time: np.ndarray,
        behav_nodes: np.ndarray,
        spikes: np.ndarray,
        spike_time: np.ndarray,
        **kwargs
    ) -> Axes:
        ax, _, _ = LocTimeCurveAxes(
            ax=ax,
            behav_time=behav_time,
            behav_nodes=behav_nodes,
            spikes=spikes,
            spike_time=spike_time,
            maze_type=self.core.maze_type,
            **kwargs
        )
        
        CP = correct_paths[int(self.core.maze_type)]
        ax.set_xlim([0, CP.shape[0]])
        return ax
    
    
    def _add_field_range(
        self,
        ax: Axes,
        place_field_all: list[dict],
        behav_time: list[np.ndarray],
    ):
        assert self.field_reg.shape[0] == self.cell_indices.shape[0]
        
        n_fields = self.field_reg.shape[1]
        n_sessions = self.field_reg.shape[0]
        
        colors = sns.color_palette("Spectral", n_fields)
        np.random.shuffle(colors)
        
        for i in range(n_fields):
            
            for d in range(n_sessions):
                if np.isnan(self.field_info[d, i]) or self.field_info[d, i] == 0:
                    continue
                
                lef, rig = set_range(
                    self.core.maze_type, 
                    spike_nodes_transform(
                        place_field_all[d][int(self.field_info[d, i])], 
                        12
                    )
                )
                lef += 0.5
                rig += 1.5
                ax.fill_betweenx(y=[np.min(behav_time[d])/1000+np.random.rand()*20, 
                                    np.max(behav_time[d])/1000-np.random.rand()*20], 
                                 x1=lef, 
                                 x2 = rig, 
                                 alpha=0.5, 
                                 edgecolor=None, 
                                 linewidth=0, 
                                 color = colors[i], 
                                 zorder = 0)
    
        return ax
    
    
    def visualize(
        self, 
        save_loc: str | None = None, 
        file_name: str | None = None,
        is_show: bool = False,
        field: np.ndarray | None = None,
        loctimecurve_kwargs: dict = {},
        **kwargs
    ) -> None:
        yticks, ylabels = [], []
        for i in range(self.core.num):
            if self.core.cell_indices[i] == 0:
                continue
            
            self._add_loc_time_curve(
                ax=self.ax,
                behav_time=cp.deepcopy(self.core.behav_times_list[i]),
                behav_nodes=cp.deepcopy(self.core.behav_nodes_list[i]),
                spikes=cp.deepcopy(self.core.spikes_list[i]),
                spike_time=cp.deepcopy(self.core.ms_time_behav_list[i]),
                **loctimecurve_kwargs
            )
        
        self._add_field_range(
            ax=self.ax,
            place_field_all=cp.deepcopy(self.place_field_all),
            behav_time=cp.deepcopy(self.core.behav_times_list),
        )
        self.ax.set_ylim([0, self.core.t_max])
        ymin, ymax = np.min(self.core.behav_times_list[i]), np.max(self.core.behav_times_list[i])
        yticks = yticks + [ymin/1000, ymax/1000]
        ylabels = ylabels + [0, int((ymax-ymin)/1000)+1]
            
        self.ax.set_yticks(yticks, ylabels)
        
        print("    Plot over, saving figure...")
        if is_show or file_name is None or save_loc is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi=600)
            plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi=600)
            plt.close()
    
    @staticmethod
    def visualize_cells(
        index_map: np.ndarray,
        field_info: np.ndarray,
        field_reg: np.ndarray,
        cell_pairs: np.ndarray,
        f: pd.DataFrame,
        mouse: int,
        maze_type: int,
        session: int,
        dates: list[str],
        core: MultiDayCore | None = None,
        save_loc: str | None = None,
        file_name: str | None = None,
        layout_kw: dict = {},
        is_show: bool = False,
        is_fit: bool = False,
        field: np.ndarray | None = None,
        shuffle_name: str | None = None,
        interv_time: float = 80000,
        place_field_all: list[list[dict]] | None = None,
        **kwargs
    ):  
        try:
            assert len(dates) > 1
        except:
            raise ValueError(f'Please set dates! It needs more than one date but {dates} was given.')
        
        file_indices = np.array([np.where((f['MiceID'] == mouse)&(f['date'] == d)&(f['session'] == session)&(f['maze_type'] == maze_type))[0][0] for d in dates], dtype=np.int64)
        print("    File Indices: ", file_indices)
     
        if file_indices.shape[0] == 0:
            raise ValueError(f"No file was found for {mouse} session {session} {maze_type} {dates}!")    
        
        
        if core is None:
            core = MultiDayCore(keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                  'DeconvSignal', 'ms_time', 'maze_type', 'place_field_all_multiday',
                                  'old_map_clear'], 
                                interv_time=interv_time)
            core.get_trace_set(f, file_indices, 
                               keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                     'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                     'DeconvSignal', 'ms_time', 'maze_type', 'place_field_all_multiday',
                                     'old_map_clear'])
        
        for n, i in enumerate(cell_pairs):
            existed_cell = np.where(index_map[:, i] > 0)[0][0]
            field_indices = np.where(field_info[existed_cell, :, 0] == index_map[existed_cell, i])[0]
            cell_indices = index_map[:, i]
            print("Cell Indices: ", cell_indices)
            print(f"   {n}/{len(cell_pairs)}, cell num = {np.count_nonzero(cell_indices)}")
            core2 = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            Visualizer = MultiDayFields(f, file_indices, cell_indices, core=core2, 
                                        field_info=field_info[:, field_indices, 2], 
                                        field_reg=field_reg[:, field_indices], # Field centers.
                                        place_field_all = place_field_all[i],
                                        **layout_kw)   
            Visualizer.visualize(
                save_loc=save_loc, 
                file_name="Line "+str(i+1), 
                shuffle_name=shuffle_name, 
                is_fit=is_fit, 
                field=field, 
                is_show=is_show, 
                **kwargs
            )