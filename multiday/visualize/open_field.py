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
from mylib import RateMapAxes, TraceMapAxes, LocTimeCurveAxes, InstantRateCurveAxes, LinearizedRateMapAxes
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from mylib.field.in_field import InFieldRateChangeModel, set_range
from mylib.maze_graph import correct_paths
from mylib.maze_utils3 import spike_nodes_transform, SF2FF
import seaborn as sns
import h5py


class MultiDayLayoutOpenField:
    def __init__(
        self, 
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 80000,
        width_ratios: list = [2, 2],
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        gridspec_kw: dict = {}
    ) -> None:
        if core is None:
            self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
        else:
            self.core = core
         
        grid = GridSpec(self.core.num, 2, width_ratios=width_ratios, **gridspec_kw)
        self.cell_indices = cell_indices
        
        if figsize is None:
            figsize = (6, self.core.num*2)
            
        fig = plt.figure(figsize=figsize)
        
        self.smooth_map_axes = []
        self.trace_map_axes = []
        
        # Clear Axes at first
        for i in range(self.core.num):
            ax1 = Clear_Axes(fig.add_subplot(grid[i, 0]))
            ax1.axis([-0.6, 47.6, 47.6, -0.6])
            ax2 = Clear_Axes(fig.add_subplot(grid[i, 1]))
            ax2.axis([-0.6, 47.6, 47.6, -0.6])
            self.smooth_map_axes.append(ax1)
            self.trace_map_axes.append(ax2)
            
    
    def _add_smooth_map(self, ax: Axes, smooth_map: np.ndarray, info: np.ndarray, **kwargs) -> Axes:
        tit = "SI: "+str(round(info[0], 2))+", Peak: "+str(round(info[1], 2))
        color = 'red' if info[2] == 1 else 'black'
        ax, _, _ = RateMapAxes(
            ax=ax,
            content=smooth_map,
            maze_type=self.core.maze_type,
            is_inverty=True,
            is_colorbar=False,
            maze_args={'linewidth':0.5, 'color': 'white'},
            title=tit,
            title_color=color,
            **kwargs
        )
        ax.set_aspect("equal")
        ax.axis([-0.6, 47.6, 47.6, -0.6])
        return ax
    
    def _add_trace_map(
        self, 
        ax: Axes, 
        behav_time: np.ndarray, 
        behav_pos: np.ndarray,
        spikes: np.ndarray, 
        spike_time: np.ndarray,
        place_field_all: dict,
        maze_kwargs: dict = {'linewidth':0.5, 'color': 'black'},
        traj_kwargs: dict = {'linewidth':0.5},
        **kwargs
    ) -> Axes:
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        for j, k in enumerate(place_field_all.keys()):
            for b in place_field_all[k]:
                x, y = (b-1)%48, (b-1)//48
                ax.fill_betweenx(y=[y-0.5, y+0.5], x1=x-0.5, x2 = x+0.5, alpha=0.6, edgecolor=None, linewidth=0, color = colors[j], zorder = j)
                
        ax, _, _ = TraceMapAxes(
            ax=ax,
            behav_time=behav_time,
            trajectory=behav_pos,
            spikes=spikes,
            spike_time=spike_time,
            maze_type=self.core.maze_type,
            markersize=1.5,
            maze_kwargs=maze_kwargs,
            traj_kwargs=traj_kwargs,
            **kwargs
        )
        ax.set_aspect("equal")
        ax.axis([-0.6, 47.6, 47.6, -0.6])
        return ax
    
    def visualize(
        self, 
        save_loc: str | None = None, 
        file_name: str | None = None,
        shuffle_name: str = None, 
        is_fit: bool = False,
        is_show: bool = False,
        field: np.ndarray | None = None,
        smooth_map_kwargs: dict = {},
        trace_map_kwargs: dict = {},
        loctimecurve_kwargs: dict = {},
        **kwargs
    ) -> None:
        yticks, ylabels = [], []
        for i in range(self.core.num):
            if self.core.cell_indices[i] == 0:
                continue
            
            self._add_smooth_map(
                ax=self.smooth_map_axes[self.core.num - i-1],
                smooth_map=self.core.smooth_maps[i],
                info=self.core.smooth_maps_info[i],
                **smooth_map_kwargs
            )
            
            self._add_trace_map(
                ax=self.trace_map_axes[self.core.num - i-1],
                behav_time=cp.deepcopy(self.core.behav_times_list[i]),
                behav_pos=cp.deepcopy(self.core.behav_positions_list[i]),
                spikes=cp.deepcopy(self.core.spikes_list[i]),
                spike_time=cp.deepcopy(self.core.ms_time_behav_list[i]),
                place_field_all=cp.deepcopy(self.core.place_fields_list[i]),
                title="Cell "+str(int(self.core.cell_indices[i])),
                **trace_map_kwargs
            )
        
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
        cell_pairs: np.ndarray,
        f: pd.DataFrame,
        mouse: int,
        maze_type: int,
        session: int,
        dates: list[str],
        file_indices: np.ndarray | None = None,
        core: MultiDayCore | None = None,
        save_loc: str | None = None,
        file_name: str | None = None,
        layout_kw: dict = {},
        is_show: bool = False,
        is_fit: bool = False,
        field: np.ndarray | None = None,
        shuffle_name: str | None = None,
        interv_time: float = 80000,
        direction: str | None = None,
        **kwargs
    ):  
        try:
            assert len(dates) > 1
        except:
            raise ValueError(f'Please set dates! It needs more than one date but {dates} was given.')
        
        if file_indices is None:
            file_indices = np.array([np.where((f['MiceID'] == mouse)&(f['date'] == d)&(f['session'] == session)&(f['maze_type'] == maze_type))[0][0] for d in dates], dtype=np.int64)
        print("    File Indices: ", file_indices)
     
        if file_indices.shape[0] == 0:
            raise ValueError(f"No file was found for {mouse} session {session} {maze_type} {dates}!")    
        
        if core is None:
            core = MultiDayCore(keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                  'place_field_all_multiday', 'maze_type', 
                                  'old_map_clear'], direction=direction,
                                interv_time=interv_time)
            core.get_trace_set(f, file_indices, 
                               keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                     'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                     'place_field_all_multiday', 'maze_type', 
                                     'old_map_clear'])
             
        for n, i in enumerate(cell_pairs):
            cell_indices = index_map[:, i]
            print("Cell Indices: ", cell_indices)
            print(f"   {n}/{len(cell_pairs)}, cell num = {np.count_nonzero(cell_indices)}")
            core2 = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            Visualizer = MultiDayLayoutOpenField(f, file_indices, cell_indices, core=core2, **layout_kw)   
            Visualizer.visualize(
                save_loc=save_loc, 
                file_name="Line "+str(i+1), 
                shuffle_name=shuffle_name, 
                is_fit=is_fit, 
                field=field, 
                is_show=is_show, 
                **kwargs
            )
