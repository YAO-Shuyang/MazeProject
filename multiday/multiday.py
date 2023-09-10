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
from mylib import RateMapAxes, TraceMapAxes, LocTimeCurveAxes, InstantRateCurveAxes
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from mylib.field.in_field import InFieldRateChangeModel
from mylib.maze_graph import correct_paths

class MultiDayLayout:
    def __init__(
        self, 
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 50000,
        width_ratios: list = [2, 2, 6, 2],
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        gridspec_kw: dict = {}
    ) -> None:
        self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            
        grid = GridSpec(self.core.num, 4, width_ratios=width_ratios, **gridspec_kw)
        self.cell_indices = cell_indices
        
        if figsize is None:
            figsize = (12, self.core.num*2)
            
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
        
        self.ax = Clear_Axes(fig.add_subplot(grid[:, 2]), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        self.ax_rig = Clear_Axes(fig.add_subplot(grid[:, 3]))
            
    
    def _add_smooth_map(self, ax: Axes, smooth_map: np.ndarray, info: np.ndarray, **kwargs) -> Axes:
        tit = "SI: "+str(round(info[0], 2))+", Peak: "+str(round(info[1], 2))
        color = 'red' if info[2] == 1 else 'black'
        ax, _, _ = RateMapAxes(
            ax=ax,
            content=smooth_map,
            maze_type=self.core.maze_type,
            is_inverty=True,
            is_colorbar=False,
            maze_args={'linewidth':0.8, 'color': 'white'},
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
        maze_kwargs: dict = {'linewidth':0.8, 'color': 'black'},
        traj_kwargs: dict = {},
        **kwargs
    ) -> Axes:
        ax, _, _ = TraceMapAxes(
            ax=ax,
            behav_time=behav_time,
            trajectory=behav_pos,
            spikes=spikes,
            spike_time=spike_time,
            maze_type=self.core.maze_type,
            maze_kwargs=maze_kwargs,
            traj_kwargs=traj_kwargs,
            **kwargs
        )
        ax.set_aspect("equal")
        ax.axis([-0.6, 47.6, 47.6, -0.6])
        return ax
        
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
        ax.set_xlim([0, CP.shape[0]+1])
        return ax
    
    def _add_instant_rate(
        self,
        ax: Axes,
        field: np.ndarray,
        t_max: float | None = None,
        save_loc: str | None = None,
        shuffle_name: str | None = None,
        model_kwargs: dict = {},
        **kwargs
    ) -> Axes:
        if os.path.exists(save_loc) == False:
            os.mkdir(save_loc)
        
        if shuffle_name is None:
            shuffle_name = "shuffle"
        
        try:
            model = InFieldRateChangeModel()
            model.temporal_analysis(
                field=field, 
                maze_type=self.core.maze_type,
                ms_time_original=cp.deepcopy(self.core.ms_time_original_all),
                deconv_signal=cp.deepcopy(self.core.deconv_signal_all),
                behav_nodes=cp.deepcopy(self.core.behav_nodes_all), 
                behav_time=cp.deepcopy(self.core.behav_times_all),
                **model_kwargs
            )
            
            model.fit()
            model.shuffle_test(shuffle_times=10000)
            model.visualize_shuffle_result(save_loc=save_loc, file_name=shuffle_name)
            
            cal_events_time, cal_events_rate = model.cal_events_time, model.cal_events_rate
            MRIG = gaussian_smooth_matrix1d(1000, window = 40, sigma=3, folder=0.001)
            ax = InstantRateCurveAxes(
                ax=ax,
                time_stamp=cal_events_time.flatten(),
                content=cal_events_rate.flatten(),
                field=field,
                M=MRIG,
                t_max=t_max*1000,
                title=model.get_info()['ctype'],
                **kwargs
            )
        except:
            print("Some errors were raisen by InFieldRateChangeModel")
            
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
                title="Cell "+str(int(self.core.cell_indices[i])),
                **trace_map_kwargs
            )
            
        self._add_loc_time_curve(
            ax=self.ax,
            behav_time=cp.deepcopy(self.core.behav_times_all),
            behav_nodes=cp.deepcopy(self.core.behav_nodes_all),
            spikes=cp.deepcopy(self.core.spikes_all),
            spike_time=cp.deepcopy(self.core.ms_time_behav_all),
            **loctimecurve_kwargs
        )
        self.ax.set_ylim([0, self.core.t_max])
        
        if is_fit:
            if field is None:
                raise ValueError("field is None! Please set field or use is_fit = False")
            t_max = np.nanmax(self.core.behav_times_all[-1])/1000
            self._add_instant_rate(
                ax=self.ax_rig,
                field=field,
                t_max=t_max,
                save_loc=save_loc,
                shuffle_name=shuffle_name,
                **loctimecurve_kwargs
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
        i: int,
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
        **kwargs
    ):  
        try:
            assert len(dates) > 1
        except:
            raise ValueError(f'Please set dates! It needs more than one date but {dates} was given.')
        
        cell_indices = index_map[:, i]
        print("    Cell Indices: ", cell_indices)
        
        file_indices = np.array([np.where((f['MiceID'] == mouse)&(f['date'] == d)&(f['session'] == session)&(f['maze_type'] == maze_type))[0][0] for d in dates], dtype=np.int64)
        
        if file_indices.shape[0] == 0:
            raise ValueError(f"No file was found for {mouse} session {session} {maze_type} {dates}!")
        
        print("    File Indices: ", file_indices)

        Visualizer = MultiDayLayout(f, file_indices, cell_indices, core=core, **layout_kw)
        Visualizer.visualize(
            save_loc=save_loc, 
            file_name=file_name, 
            shuffle_name=shuffle_name, 
            is_fit=is_fit, 
            field=field, 
            is_show=is_show, 
            **kwargs
        )
        del Visualizer
        
            