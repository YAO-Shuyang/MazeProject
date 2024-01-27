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


class MultiDayLayout:
    def __init__(
        self, 
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 80000,
        width_ratios: list = [2, 2, 6, 2],
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        gridspec_kw: dict = {}
    ) -> None:
        if core is None:
            self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
        else:
            self.core = core
         
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
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        #for j, k in enumerate(place_field_all.keys()):
        #    for b in place_field_all[k]:
        #        x, y = (b-1)%48, (b-1)//48
        #        ax.fill_betweenx(y=[y-0.5, y+0.5], x1=x-0.5, x2 = x+0.5, alpha=0.6, edgecolor=None, linewidth=0, color = colors[j])
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
        place_field_all: dict,
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
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        #for j, k in enumerate(place_field_all.keys()):
        #    lef, rig = set_range(self.core.maze_type, spike_nodes_transform(place_field_all[k], 12))
        #    lef += 0.5
        #    rig += 1.5
        #    ax.fill_betweenx(y=[np.min(behav_time)/1000, np.max(behav_time)/1000], x1=lef, x2 = rig, alpha=0.5, edgecolor=None, linewidth=0, color = colors[j])
        
        CP = correct_paths[int(self.core.maze_type)]
        ax.set_xlim([0, CP.shape[0]])
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
            
            self._add_loc_time_curve(
                ax=self.ax,
                behav_time=cp.deepcopy(self.core.behav_times_list[i]),
                behav_nodes=cp.deepcopy(self.core.behav_nodes_list[i]),
                spikes=cp.deepcopy(self.core.spikes_list[i]),
                spike_time=cp.deepcopy(self.core.ms_time_behav_list[i]),
                place_field_all=cp.deepcopy(self.core.place_fields_list[i]),
                **loctimecurve_kwargs
            )
            self.ax.set_ylim([0, self.core.t_max])
            ymin, ymax = np.min(self.core.behav_times_list[i]), np.max(self.core.behav_times_list[i])
            yticks = yticks + [ymin/1000, ymax/1000]
            ylabels = ylabels + [0, int((ymax-ymin)/1000)+1]
            
        self.ax.set_yticks(yticks, ylabels)
        
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
                                  'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type', 
                                  'old_map_clear'], 
                                interv_time=interv_time)
            core.get_trace_set(f, file_indices, 
                               keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                     'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                     'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type', 
                                     'old_map_clear'])
             
        for i in cell_pairs:
            cell_indices = index_map[:, i]
            print("Cell Indices: ", cell_indices)
            print(f"   {i}/{len(cell_pairs)}, cell num = {np.count_nonzero(cell_indices)}")
            core2 = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            Visualizer = MultiDayLayout(f, file_indices, cell_indices, core=core2, **layout_kw)   
            Visualizer.visualize(
                save_loc=save_loc, 
                file_name="Line "+str(i+1), 
                shuffle_name=shuffle_name, 
                is_fit=is_fit, 
                field=field, 
                is_show=is_show, 
                **kwargs
            )
        
class MultiDayLayout2:
    def __init__(
        self, 
        f: pd.DataFrame, 
        footprints: list[str],
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 80000,
        width_ratios: list = [2, 2, 2, 6],
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        gridspec_kw: dict = {}
    ) -> None:
        if core is None:
            self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
        else:
            self.core = core
         
        grid = GridSpec(self.core.num, 4, width_ratios=width_ratios, **gridspec_kw)
        self.cell_indices = cell_indices
        
        if figsize is None:
            figsize = (12, self.core.num*2)
            
        fig = plt.figure(figsize=figsize)
        
        self.footprint_axes = []
        self.smooth_map_axes = []
        self.trace_map_axes = []
        self.linearized_map_axes = []
        
        # Clear Axes at first
        for i in range(self.core.num):
            ax1 = Clear_Axes(fig.add_subplot(grid[i, 0]))
            ax2 = Clear_Axes(fig.add_subplot(grid[i, 1]))
            ax2.axis([-0.6, 47.6, 47.6, -0.6])
            ax3 = Clear_Axes(fig.add_subplot(grid[i, 2]))
            ax3.axis([-0.6, 47.6, 47.6, -0.6])
            ax4 = Clear_Axes(fig.add_subplot(grid[i, 3]), close_spines=['top', 'right', 'bottom'], ifyticks=True)
            self.footprint_axes.append(ax1)
            self.smooth_map_axes.append(ax2)
            self.trace_map_axes.append(ax3)
            self.linearized_map_axes.append(ax4)
            
        self.footprints = footprints
        self.cell_indices = cell_indices
        
    
    def _add_footprint(self, ax: Axes, footprint_dir: np.ndarray, cell_id: int, **kwargs):
        with h5py.File(footprint_dir, 'r') as f:
            sfp = np.array(f['SFP'])
            for i in range(sfp.shape[2]):
                sfp[:, :, i] = sfp[:, :, i] / np.nanmax(sfp[:, :, i])
            
        footprint = np.nanmax(sfp, axis = 2)
        center_x, center_y = np.where(sfp[:, :, cell_id] == np.max(sfp[:, :, cell_id]))
        center_x, center_y = center_x[0], center_y[0]
        
        ax.imshow(footprint.T, cmap = 'gray')
        ax.plot([center_x], [center_y], 'o', markeredgewidth=0, markersize = 3, color = 'orange')
        ax.set_aspect("equal")
        ax.axis([center_x - 15, center_x + 15, center_y - 15, center_y + 15])
        return ax
    
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
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        for j, k in enumerate(place_field_all.keys()):
            for b in place_field_all[k]:
                x, y = (b-1)%48, (b-1)//48
                ax.fill_betweenx(y=[y-0.5, y+0.5], x1=x-0.5, x2 = x+0.5, alpha=0.6, edgecolor=None, linewidth=0, color = colors[j])
        ax.set_aspect("equal")
        ax.axis([-0.6, 47.6, 47.6, -0.6])
        return ax
        
    def _add_linearized_rate_curve(
        self,
        ax: Axes,
        old_map: np.ndarray,
        place_field_all: dict,
        global_ymax: float,
        **kwargs
    ) -> Axes:
        MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)
        CP = cp.deepcopy(correct_paths[int(self.core.maze_type)])

        ax, _ = LinearizedRateMapAxes(
            ax=ax,
            content=old_map,
            maze_type=self.core.maze_type,
            M=MTOP,
            linewidth=0.5
        )
        ax.set_xlim([0, len(CP)+1])
        y_max = np.nanmax(old_map)
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        
        for j, k in enumerate(place_field_all.keys()):
            lef, rig = set_range(self.core.maze_type, spike_nodes_transform(place_field_all[k], 12))
            lef += 0.5
            rig += 1.5
            ax.plot([lef, rig], [-global_ymax*0.09, -global_ymax*0.09], color = colors[j], linewidth=0.5)
        
        CP = correct_paths[int(self.core.maze_type)]
        ax.set_xlim([0, CP.shape[0]])
        ax.set_ylim([-global_ymax*0.15, global_ymax])
        ax.set_yticks([0, round(y_max,2)])
        return ax
        
    def visualize(
        self, 
        save_loc: str | None = None, 
        file_name: str | None = None,
        is_show: bool = False,
        smooth_map_kwargs: dict = {},
        trace_map_kwargs: dict = {},
        loctimecurve_kwargs: dict = {},
        **kwargs
    ) -> None:
        yticks, ylabels = [], []
        for i in range(self.core.num):
            if self.core.cell_indices[i] == 0:
                Clear_Axes(self.linearized_map_axes[self.core.num - i-1])
                continue
            
            self._add_footprint(
                ax=self.footprint_axes[self.core.num - i-1],
                footprint_dir=self.footprints[i],
                cell_id=self.cell_indices[i]-1
            )
            
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
            
            self._add_linearized_rate_curve(
                ax=self.linearized_map_axes[self.core.num - i-1],
                old_map=cp.deepcopy(self.core.old_maps[i]),
                place_field_all=cp.deepcopy(self.core.place_fields_list[i]),
                global_ymax=np.nanmax(self.core.old_maps),
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
        cell_pairs: np.ndarray,
        f: pd.DataFrame,
        footprint_dirs: list,
        mouse: int,
        maze_type: int,
        session: int,
        dates: list[str],
        core: MultiDayCore | None = None,
        save_loc: str | None = None,
        file_name: str | None = None,
        layout_kw: dict = {},
        is_show: bool = False,
        interv_time: float = 80000,
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
                                  'DeconvSignal', 'ms_time', 'place_field_all_multiday', 
                                  'maze_type', 'old_map_clear'], 
                                interv_time=interv_time)
            core.get_trace_set(f, file_indices, 
                               keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                     'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                     'DeconvSignal', 'ms_time', 'place_field_all_multiday', 
                                     'maze_type', 'old_map_clear'])
             
        for i in cell_pairs:
            cell_indices = index_map[:, i]
            print("    Cell Indices: ", cell_indices)
            print(f"{i}/{len(cell_pairs)}, cell num = {cell_indices.shape[0]}")
            core2 = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            Visualizer = MultiDayLayout2(f, footprint_dirs, file_indices, cell_indices, core=core2, **layout_kw)
            Visualizer.visualize(
                save_loc=save_loc, 
                file_name="Line "+str(i+1), 
                is_show=is_show, 
                **kwargs
            )
        del Visualizer
        



class MultiDayLayout3:
    def __init__(
        self, 
        f: pd.DataFrame, 
        footprints: list[str],
        file_indices: np.ndarray, 
        cell_indices: np.ndarray,
        interv_time: float = 80000,
        width_ratios: list = [2, 2, 2, 10],
        figsize: tuple | None = None,
        core: MultiDayCore | None = None,
        gridspec_kw: dict = {}
    ) -> None:
        if core is None:
            self.core = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
        else:
            self.core = core
         
        grid = GridSpec(1, 4, width_ratios=width_ratios, **gridspec_kw)
        self.cell_indices = cell_indices
        
        if figsize is None:
            figsize = (12, 2)
            
        fig = plt.figure(figsize=figsize)
        
        self.smooth_map_axes = []
        self.trace_map_axes = []
        
        ax1 = Clear_Axes(fig.add_subplot(grid[0]))
        ax2 = Clear_Axes(fig.add_subplot(grid[1]))
        ax2.axis([-0.6, 47.6, 47.6, -0.6])
        ax3 = Clear_Axes(fig.add_subplot(grid[2]))
        ax3.axis([-0.6, 47.6, 47.6, -0.6])
        ax4 = Clear_Axes(fig.add_subplot(grid[3]), close_spines=['top', 'right', 'bottom'], ifyticks=True)
        self.footprint_axes = ax1
        self.smooth_map_axes = ax2
        self.trace_map_axes = ax3
        self.loc_time_axes = ax4
        
        self.footprints = footprints
        self.cell_indices = cell_indices
        
    
    def _add_footprint(self, ax: Axes, footprint_dir: np.ndarray, cell_id: int, **kwargs):
        with h5py.File(footprint_dir, 'r') as f:
            sfp = np.array(f['SFP'])
            for i in range(sfp.shape[2]):
                sfp[:, :, i] = sfp[:, :, i] / np.nanmax(sfp[:, :, i])
            
        footprint = np.nanmax(sfp, axis = 2)
        center_x, center_y = np.where(sfp[:, :, cell_id] == np.max(sfp[:, :, cell_id]))
        center_x, center_y = center_x[0], center_y[0]
        
        ax.imshow(footprint.T, cmap = 'gray')
        ax.plot([center_x], [center_y], 'o', markeredgewidth=0, markersize = 3, color = 'orange')
        ax.set_aspect("equal")
        ax.axis([center_x - 15, center_x + 15, center_y - 15, center_y + 15])
        return ax
            
    
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
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        for j, k in enumerate(place_field_all.keys()):
            for b in place_field_all[k]:
                x, y = (b-1)%48, (b-1)//48
                ax.fill_betweenx(y=[y-0.5, y+0.5], x1=x-0.5, x2 = x+0.5, alpha=0.6, edgecolor=None, linewidth=0, color = colors[j])
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
        place_field_all: dict,
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
        colors = sns.color_palette("Spectral", len(place_field_all.keys())+2)[2:]
        colors.reverse()
        for j, k in enumerate(place_field_all.keys()):
            lef, rig = set_range(self.core.maze_type, spike_nodes_transform(place_field_all[k], 12))
            lef += 0.5
            rig += 1.5
            ax.fill_betweenx(y=[np.min(behav_time)/1000, np.max(behav_time)/1000], x1=lef, x2 = rig, alpha=0.5, edgecolor=None, linewidth=0, color = colors[j])
        
        CP = correct_paths[int(self.core.maze_type)]
        ax.set_xlim([0, CP.shape[0]])
        ax.set_ylim(np.min(behav_time)/1000, np.max(behav_time)/1000)
        ax.set_yticks(ticks=(np.min(behav_time)/1000, np.max(behav_time)/1000), 
                      labels=[0, round(np.max(behav_time)/1000-np.min(behav_time)/1000, 2)])
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
        if os.path.exists(os.path.join(save_loc, file_name)) == False:
            os.mkdir(os.path.join(save_loc, file_name))
        for i in range(self.core.num):
            if self.core.cell_indices[i] == 0:
                continue
            
            self._add_footprint(
                ax=self.footprint_axes,
                footprint_dir=self.footprints[i],
                cell_id=self.cell_indices[i]-1
            )
            
            self._add_smooth_map(
                ax=self.smooth_map_axes,
                smooth_map=self.core.smooth_maps[i],
                info=self.core.smooth_maps_info[i],
                **smooth_map_kwargs
            )
            
            self._add_trace_map(
                ax=self.trace_map_axes,
                behav_time=cp.deepcopy(self.core.behav_times_list[i]),
                behav_pos=cp.deepcopy(self.core.behav_positions_list[i]),
                spikes=cp.deepcopy(self.core.spikes_list[i]),
                spike_time=cp.deepcopy(self.core.ms_time_behav_list[i]),
                place_field_all=cp.deepcopy(self.core.place_fields_list[i]),
                title="Cell "+str(int(self.core.cell_indices[i])),
                **trace_map_kwargs
            )
            
            self._add_loc_time_curve(
                ax=self.loc_time_axes,
                behav_time=cp.deepcopy(self.core.behav_times_list[i]),
                behav_nodes=cp.deepcopy(self.core.behav_nodes_list[i]),
                spikes=cp.deepcopy(self.core.spikes_list[i]),
                spike_time=cp.deepcopy(self.core.ms_time_behav_list[i]),
                place_field_all=cp.deepcopy(self.core.place_fields_list[i]),
                **loctimecurve_kwargs
            )
        
            print(f"    Plot over, saving figure {i+1}/{self.core.num}...")
            if is_show or file_name is None or save_loc is None:
                plt.tight_layout()
            else:
                plt.savefig(os.path.join(save_loc, file_name, f'{i+1}.png'), dpi=600)
                plt.savefig(os.path.join(save_loc, file_name, f'{i+1}.svg'), dpi=600)
            
            self.footprint_axes.clear()
            self.smooth_map_axes.clear()
            self.trace_map_axes.clear()
            self.loc_time_axes.clear()
            
        plt.close()
    
    @staticmethod
    def visualize_cells(
        index_map: np.ndarray,
        cell_pairs: np.ndarray,
        f: pd.DataFrame,
        footprint_dirs: list,
        mouse: int,
        maze_type: int,
        session: int,
        dates: list[str],
        core: MultiDayCore | None = None,
        save_loc: str | None = None,
        file_name: str | None = None,
        layout_kw: dict = {},
        is_show: bool = False,
        interv_time: float = 80000,
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
                                  'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type', 
                                  'old_map_clear'], 
                                interv_time=interv_time)
            core.get_trace_set(f, file_indices, 
                               keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                     'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                     'DeconvSignal', 'ms_time', 'place_field_all_multiday', 'maze_type', 
                                     'old_map_clear'])
             
        for i in cell_pairs:
            cell_indices = index_map[:, i]
            print("    Cell Indices: ", cell_indices)
            print(f"{i}/{len(cell_pairs)}, cell num = {cell_indices.shape[0]}")
            core2 = MultiDayCore.concat_core(f, file_indices, cell_indices, core=core, interv_time=interv_time)
            Visualizer = MultiDayLayout3(f, footprint_dirs, file_indices, cell_indices, core=core2, **layout_kw)
            Visualizer.visualize(
                save_loc=save_loc, 
                file_name="Line "+str(i+1), 
                is_show=is_show, 
                **kwargs
            )
        del Visualizer
  