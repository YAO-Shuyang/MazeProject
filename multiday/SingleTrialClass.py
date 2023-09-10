import numpy as np
from dataclasses import dataclass
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd


from mylib.maze_graph import correct_paths
from mylib.maze_utils3 import GetDMatrices, mkdir, Clear_Axes, ColorBarsTicks, calc_ms_position, SpikeType
from mylib.divide_laps.lap_split import LapSplit
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d, gaussian_smooth_matrix2d

@dataclass
class SingleLapActivity:
    session: int
    lap: int
    maze_type: int
    mouse: int
    date: int
    
    spikes: np.ndarray
    spike_nodes2d: np.ndarray
    spike_time: np.ndarray
    pos2d: np.ndarray
    
    pos1d: np.ndarray | None = None
    turn_thre: float = 8
    
    def _detect_turns(self) -> bool:
        """_detect_turns

        Returns
        -------
        bool
            Whether the trajectory contains turn events
        
        Raises
        ------
        AttributeError
            The concept of lap only supported maze data.
        """
        
        if self.maze_type == 0:
            raise AttributeError(f"Maze {self.maze_type} does not support linearized position method!")

        if self.pos1d is None:
            self.get_pos1d()
        
        dx = np.ediff1d(self.pos1d)
        back_length_sum = np.nansum(dx[np.where(dx < 0)[0]])
        return back_length_sum >= -self.turn_thre
        
    def _detect_wrong(self) -> bool:
        """_detect_wrong: detect wrong events in the trajectory

        Returns
        -------
        bool
            Whether the trajectory contains wrong events
        
        Raises
        ------
        AttributeError
            The concept of lap only supported maze data.
        """
        if self.maze_type == 0:
            raise AttributeError(f"Maze {self.maze_type} does not support linearized position method!") 
        
        CP = correct_paths[(self.maze_type, 48)]
        
        for i in range(self.spike_nodes2d.shape[0]):
            if self.spike_nodes2d[i] not in CP:
                return False
            
        return True
    
    def is_perfect(self):
        return self._detect_turns() and self._detect_wrong()
    
    def get_velocity2d(self) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = np.append(np.ediff1d(self.pos2d[:, 0]), np.nan), np.append(np.ediff1d(self.pos2d[:, 1]), np.nan)
        dt = np.append(np.ediff1d(self.spike_time), np.nan)
        
        vx, vy = dx/dt*1000, dy/dt*1000
        vx[-1] = vx[-2]
        vy[-1] = vy[-2]
        self.vx, self.vy = vx, vy
        
        vx[vx>10000] = np.nan
        vy[vy>10000] = np.nan
        vx[vx<-10000] = np.nan
        vy[vy<-10000] = np.nan
        return vx, vy
    
    def get_pos1d(self):
        if self.maze_type == 0:
            raise AttributeError(f"Maze {self.maze_type} does not support linearized position method!")
        
        D = GetDMatrices(self.maze_type, 48)
        CP = correct_paths[(self.maze_type, 48)]

        self.pos1d = np.zeros_like(self.spike_nodes2d, dtype=np.float64) * np.nan
        for i in range(self.pos1d.shape[0]):
            if self.spike_nodes2d[i] in CP:
                self.pos1d[i] = D[int(self.spike_nodes2d[i]-1), 0]
                
        self.max_length = np.nanmax(D)
        
        return self.pos1d
    
    def calc_tuning_curve1d(self, bin_size = 2, sigma = 3, folder = 1, M: np.ndarray = None):
        try:
            assert self.pos1d is not None
        except:
            self.get_pos1d()
        
        bin_num = int(self.max_length/bin_size)+1
        self.spike_nodes1d = (self.pos1d / self.max_length)*bin_num//1
        
        median_dt = np.nanmedian(np.ediff1d(self.spike_time))
        dt = np.append(np.ediff1d(self.spike_time), median_dt)
        dt[dt >= 100] = 100
        
        _coords_range = [0, bin_num - 0.0001]

        idx = np.where(np.isnan(self.spike_nodes1d) == False)[0]
        
        occu_time, _, _ = scipy.stats.binned_statistic(
            self.spike_nodes1d[idx],
            dt[idx],
            bins=bin_num,
            statistic="sum",
            range=_coords_range)
        
        occu_time[occu_time<=1] = 1
        
        self.rate_map1d = np.zeros(bin_num, np.float64)
        for i in range(bin_num):
            frames = np.where(self.spike_nodes1d == i)[0]
            self.rate_map1d[i] = np.nansum(self.spikes[frames]) / occu_time[i] * 1000
        
        self.bin_num1d = bin_num
        self.rate_map1d[np.isnan(self.rate_map1d)] = 0
        
        if M is None:
            M = gaussian_smooth_matrix1d(bin_num, sigma=sigma, folder = folder, window=12)
            
        self.smooth_map1d = np.dot(M, self.rate_map1d.T).T
        
        return self.smooth_map1d
    
    def calc_tuning_curve2d(self, M:np.ndarray = None, sigma: int = 3, **kwargs):
        if M is None:
            M = gaussian_smooth_matrix2d(self.maze_type, sigma=sigma, _range = 16, nx=48)
            
        bin_num = 2304
        median_dt = np.nanmedian(np.ediff1d(self.spike_time))
        dt = np.append(np.ediff1d(self.spike_time), median_dt)
        dt[dt >= 100] = 100
        
        _coords_range = [0, bin_num + 0.0001]

        idx = np.where(np.isnan(self.spike_nodes2d) == False)[0]
        
        occu_time, _, _ = scipy.stats.binned_statistic(
            self.spike_nodes2d[idx],
            dt[idx],
            bins=bin_num,
            statistic="sum",
            range=_coords_range)
        
        occu_time[occu_time<=1] = 1
        
        self.rate_map2d = np.zeros(bin_num, np.float64)
        for i in range(bin_num):
            frames = np.where(self.spike_nodes2d == i+1)[0]
            self.rate_map2d[i] = np.nansum(self.spikes[frames]) / occu_time[i] * 1000
        
        self.bin_num2d = bin_num
        self.rate_map2d[np.isnan(self.rate_map2d)] = 0
        
        self.smooth_map2d = np.dot(self.rate_map2d, M.T)
        
        return self.smooth_map2d

class MultiDayLapsActivity:
    def __init__(
        self,
        contents: list[SingleLapActivity] = [],
        laps: np.ndarray | None = np.array([], dtype = np.int64),
        sessions: np.ndarray | None = np.array([], dtype = np.int64),
        is_perfect: np.ndarray | None = np.array([], dtype = np.int64)
    ) -> None:
        self.contents = contents
        self.laps = laps
        self.sessions = sessions
        self.is_perfect = is_perfect
    
    def __len__(self):
        return len(self.contents)
    
    def __iter__(self):
        return iter(self.contents)
    
    def __getitem__(self, item):
        return self.contents[item]
    
    def __setitem__(self, key, value):
        self.contents[key] = value
        
    def __delitem__(self, key):
        del self.contents[key]
    
    def __add__(self, other):
        return MultiDayLapsActivity(self.contents + other.contents, self.laps + other.laps, self.sessions + other.sessions)
    
    def __iadd__(self, other):
        self.contents = self.contents + other.contents
        self.laps += np.concatenate([self.laps, other.laps])
        self.sessions = np.concatenate([self.sessions, other.sessions])
        return self
    
    def __repr__(self):
        return "MultiDayLapsActivity object with " + len(self.contents) + " laps across " + str(len(np.unique(self.sessions)) + " session(s).")
    
    def __str__(self):
        return "MultiDayLapsActivity object with " + len(self.contents) + " laps across " + str(len(np.unique(self.sessions)) + " session(s).")
    
    def __next__(self):
        return next(self.contents), next(self.laps), next(self.sessions), next(self.is_perfect)
    
    def append(self, value: SingleLapActivity):
        self.contents.append(value)
        self.laps = np.append(self.laps, value.lap)
        self.sessions = np.append(self.sessions, value.session)
        is_perfect = 1 if value.is_perfect() else 0
        self.is_perfect = np.append(self.is_perfect, is_perfect)
        
    def pop(self):
        self.contents.pop()
        self.laps = np.delete(self.laps, -1)
        self.sessions = np.delete(self.sessions, -1)
        self.is_perfect = np.delete(self.is_perfect, -1)
    
    def init_info(self):
        self.laps = np.zeros(len(self.contents), np.int64)
        self.sessions = np.zeros(len(self.contents), np.int64)
        self.is_perfect = np.zeros(len(self.contents), np.int64)
        
        for i, trial in enumerate(self.contents):
            self.laps[i] = trial.lap
            self.sessions[i] = trial.session
            self.is_perfect[i] = 1 if trial.is_perfect() else 0
    
    @staticmethod
    def get_data(
        f: pd.DataFrame,
        file_indices: np.ndarray,
        cell_indices: np.ndarray,
        paradigm: str = "CrossMaze"
    ):
        """get_data: get data for multiple days

        Parameters
        ----------
        f : pd.DataFrame
            The dataframe that contains infomation of all recorded sessions
        file_indices : np.ndarray
            The indices of the sessions that you wanted to selected
        cell_indices : np.ndarray
            The indices of the cells that has been identified as the same cell
        paradigm : str, optional
            The paradigm, by default "CrossMaze"

        Returns
        -------
        tuple[MultiDayLapsActivity, MultiDayLapsActivity]
            The MultiDayLapsActivity objects that contain Spikes and 
            DeconvSignals respectively.
        """
        SpikesRes = []
        DeconvRes = []
        
        print("Combinate multiple laps data...")
        for i, d in enumerate(file_indices):
            if os.path.exists(f['Trace File'][d]):
                with open(f['Trace File'][d], 'rb') as handle:
                    trace = pickle.load(handle)
            else:
                print(f"Trace File {f['Trace File'][d]} does not exist.")
                continue
            
            if 'laps' not in trace.keys():
                beg, end = LapSplit(trace, paradigm)
                laps = beg.shape[0]
            else:
                beg, end = trace['lap_begin_index'], trace['lap_end_index']
                laps = trace['laps']   
            
            ms_pos2d = calc_ms_position(trace['correct_pos'], trace['correct_time'], trace['ms_time'])
            spikes_original = trace['Spikes']#SpikeType(trace['DeconvSignal'], threshold=0.5)#
            for l in tqdm(range(laps)):
                t1, t2 = np.where(trace['ms_time'] >= trace['correct_time'][beg[l]])[0][0], np.where(trace['ms_time'] <= trace['correct_time'][end[l]])[0][-1]
                t3, t4 = np.where(trace['ms_time_behav'] >= trace['correct_time'][beg[l]])[0][0], np.where(trace['ms_time_behav'] <= trace['correct_time'][end[l]])[0][-1]
                cell_id = int(cell_indices[i])-1
                
                
                trial = SingleLapActivity(
                    session=i+1,
                    lap=i+1,
                    mouse=int(f['MiceID'][d]),
                    maze_type=int(f['maze_type'][d]),
                    date=int(f['date'][d]),
                    
                    spikes = spikes_original[cell_id, t3:t4+1],#trace['Spikes_original'][cell_id, t1:t2+1],
                    spike_nodes2d=trace['spike_nodes'][t3:t4+1],
                    spike_time=trace['ms_time_behav'][t3:t4+1],
                    pos2d=ms_pos2d[t3:t4+1]
                )
                trial.get_velocity2d()
                trial.get_pos1d()
                #trial.calc_tuning_curve1d()
                #trial.calc_tuning_curve2d()
                SpikesRes.append(trial)
                
                trial = SingleLapActivity(
                    session=i+1,
                    lap=i+1,
                    mouse=int(f['MiceID'][d]),
                    maze_type=int(f['maze_type'][d]),
                    date=int(f['date'][d]),
                    
                    spikes = trace['DeconvSignal'][cell_id, t1:t2+1],
                    spike_nodes2d=trace['spike_nodes_original'][t1:t2+1],
                    spike_time=trace['ms_time'][t1:t2+1],
                    pos2d=ms_pos2d[t1:t2+1]
                )                
                trial.get_velocity2d()
                trial.get_pos1d()
                DeconvRes.append(trial)
        
        SpikesRes = MultiDayLapsActivity(SpikesRes)
        SpikesRes.init_info()
        DeconvRes = MultiDayLapsActivity(DeconvRes)
        DeconvRes.init_info()
        return SpikesRes, DeconvRes

def LapwiseRateMap(trace):
    loc = os.path.join(trace['p'], 'LapwiseRateMap')
    mkdir(loc)    
    
    fig = plt.figure(figsize=(8,4))
    ax = Clear_Axes(plt.axes(), ifyticks=True)
    lap = trace['laps']

    size = 265 if trace['maze_type'] == 1 else 247
    for n in tqdm(range(trace['n_neuron'])):
        smooth_maps1d = np.zeros((trace['laps'], size), np.float64)
        for i in range(trace['laps']):
            beg, end = trace['lap_begin_index'][i], trace['lap_end_index'][i]
    
            indices = np.where((trace['ms_time_behav'] >= trace['correct_time'][beg])&(trace['ms_time_behav'] <= trace['correct_time'][end]))[0]
        
            trial12 = SingleLapActivity(
                1,
                lap=i+1,
                label='0',
                maze_type=trace['maze_type'],
                mouse=trace['MiceID'],
                spikes = trace['Spikes'][n, indices],
                spike_time= trace['ms_time_behav'][indices],
                spike_nodes2d=trace['spike_nodes'][indices]
            )
    
            smooth_maps1d[i, :] = trial12.calc_tuning_curve1d()
        
        im = ax.imshow(smooth_maps1d, cmap = 'jet')
        ax.set_aspect("auto")
        color = 'red' if trace['is_placecell'][n] == 1 else 'black'
        ax.set_title(f"cell {n+1}, SI = {round(trace['SI_all'][n], 3)}", color = color)
        ax.set_yticks(ColorBarsTicks(lap-1, is_auto=True, tick_number=6), ColorBarsTicks(lap-1, is_auto=True, tick_number=6)+1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(smooth_maps1d), is_auto=True, tick_number=5))
        
        plt.savefig(os.path.join(loc, "Cell "+str(n+1)+".png"), dpi=600)
        plt.savefig(os.path.join(loc, "Cell "+str(n+1)+".svg"), dpi=600)
        
        cbar.remove()
        im.remove()


def LapwiseDeconvSignal(
    f: pd.DataFrame,
    file_indices: np.ndarray,
    cell_indices: np.ndarray,
    save_loc: str,
    file_name: str
):
    SpikesRes, DeconvRes = MultiDayLapsActivity.get_data(f, file_indices, cell_indices)  
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(24, 3*len(file_indices)))
    ax1 = Clear_Axes(axes[0], close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
    ax2 = Clear_Axes(axes[1], close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
    ax3 = Clear_Axes(axes[2], close_spines=['top', 'right'], ifyticks=True, ifxticks=True)
    laps = len(SpikesRes)
    
    base_strength, base_v = 0, 0
    colors = sns.color_palette("dark", len(np.unique(SpikesRes.sessions)))
    for i in range(laps):
        spikes, deconv = SpikesRes[i], DeconvRes[i]
        
        x_spikes = spikes.pos1d[np.where(spikes.spikes==1)[0]]
        x_spikes = x_spikes + np.random.rand(len(x_spikes))-0.5
        y_spikes = np.repeat(base_strength, len(x_spikes))
        
        x_deconv = deconv.pos1d
        y_deconv = deconv.spikes + base_strength
        
        dy = np.nanmax(deconv.spikes)*1.2
        
        vx, vy = deconv.vx, deconv.vy
        vmax = max(np.nanmax(vx), np.nanmax(vy))
        vmin = min(np.nanmin(vx), np.nanmin(vy))
        vx = vx + base_v - vmin
        vy = vy + base_v - vmin
        
        if DeconvRes.is_perfect[i] == 1:
            ax1.plot(x_spikes, y_spikes, '|', markeredgewidth=0.6, markersize=2, color = 'red')
            ax1.plot(x_deconv, y_deconv, linewidth=0.8, color = colors[SpikesRes.sessions[i]-1], alpha = 0.5)
            ax2.plot(x_deconv, vx, linewidth=0.8, color = colors[SpikesRes.sessions[i]-1], alpha = 0.5)
            ax3.plot(x_deconv, vy, linewidth=0.8, color = colors[SpikesRes.sessions[i]-1], alpha = 0.5)
        else:
            ax1.plot(x_spikes, y_spikes, '|', markeredgewidth=1, markersize=2, color = 'red')
            ax1.plot(x_deconv, y_deconv, linewidth=0.8, color = 'gray', alpha = 0.5)
            ax2.plot(x_deconv, vx, linewidth=0.8, color = 'gray', alpha = 0.5)
            ax3.plot(x_deconv, vy, linewidth=0.8, color = 'gray', alpha = 0.5)
        

        if deconv.lap == 1 and deconv.session != 1:
            ax1.axhline(y=base_strength-dy*0.5, color='black', linestyle='--', linewidth=0.8)
            ax2.axhline(y=base_v, color='black', linestyle='--', linewidth=0.8)
            
        base_strength += dy
        base_v += (vmax - vmin)*1.05
        
    ax1.set_aspect("auto")
    ax1.set_ylim([-2, base_strength+2])
    
    ax2.set_aspect("auto")
    ax2.set_ylim([-2, base_v+2])
    
    ax3.set_aspect("auto")
    ax3.set_ylim([-2, base_v+2])

    plt.savefig(os.path.join(save_loc, file_name+".png"), dpi=600)
    plt.savefig(os.path.join(save_loc, file_name+".svg"), dpi=600)
    plt.close()
    
    
if __name__ == '__main__':
    import os
    import pickle
    import matplotlib.pyplot as plt
    
    with open(r'E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl', 'rb') as handle:
        trace = pickle.load(handle)
        
    from mylib.local_path import f1, cellReg_09_maze1_2, cellReg_95_maze1
    from mylib.statistic_test import Read_and_Sort_IndexMap  
    
    #file_indices = np.where((f1['MiceID']==11095)&(f1['maze_type']==1)&(f1['Stage'] == 'Stage 2')&(f1['date']>=20220820))[0]      
    file_indices = np.where((f1['MiceID']==10209)&(f1['maze_type']==1)&(f1['Stage'] == 'Stage 2'))[0]      
    index_map = Read_and_Sort_IndexMap(
        path = cellReg_09_maze1_2,
        occur_num=13,
        name_label='SFP2023',#"SFP2022", name_label=
        order=np.array(['20230703', '20230705', '20230707', '20230709', '20230711', '20230713',
                    '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
                    '20230728'])
    )

    """
        order=np.array(["20220820", "20220822", "20220824", "20220826", "20220828", "20220830"])

    """    
    #save_loc = r'E:\Data\FinalResults\Field Analysis\0312 - Multiday DeconvSignal\11095-stage2-maze1-6 sessions'
    save_loc = r'E:\Data\FinalResults\Field Analysis\0312 - Multiday DeconvSignal\10209-stage2-maze1-13 sessions'
    mkdir(save_loc)
    for i in range(index_map.shape[1]):
        LapwiseDeconvSignal(f1, file_indices, index_map[:, i], save_loc=save_loc, file_name=str(i+1))
        
    """
    #LapwiseRateMap(trace)
    

    core = MultiDayCore(keys=['correct_nodes', 'correct_time', 'ms_time_behav', 'Spikes',
                                  'correct_pos', 'smooth_map_all', 'SI_all', 'is_placecell', 
                                  'DeconvSignal', 'ms_time', 'place_field_all', 'maze_type'])
    core = core.get_core(
        f=f1,
        file_indices=file_indices,
        keys=['Spikes', 'spike_nodes', 'ms_time_behav', 'laps', 'lap_begin_index', 'lap_end_index', 'correct_time']
    )
    
    index_map = Read_and_Sort_IndexMap(
        path = cellReg_09_maze1_2,
        occur_num=13,
        name_label='SFP2023',
        order=np.array(['20230703', '20230705', '20230707', '20230709', '20230711', '20230713',
                    '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
                    '20230728'])
    )

    dates = [
        20230703, 20230705, 20230707, 20230709, 20230711, 20230713,
        20230715, 20230717, 20230719, 20230721, 20230724, 20230726,
        20230728
    ]

    for i in range(index_map.shape[1]):
        spikes = np.concatenate([core.res['Spikes'][d][int(index_map[d, i])-1, :] for d in range(index_map.shape[0])])
        spike_nodes = np.concatenate(core.res['spike_nodes'])
        spike_time = np.concatenate(core.res['ms_time_behav'])
        
        size = 265 if trace['maze_type'] == 1 else 247
        fig = plt.figure(figsize=(8,4))
        ax = Clear_Axes(plt.axes(), ifyticks=True)
        lap = trace['laps']
        smooth_maps1d = np.zeros((, size), np.float64)
        for i in range(trace['laps']):
            beg, end = trace['lap_begin_index'][i], trace['lap_end_index'][i]
    
            indices = np.where((trace['ms_time_behav'] >= trace['correct_time'][beg])&(trace['ms_time_behav'] <= trace['correct_time'][end]))[0]
        
            trial12 = SingleLapActivity(
                1,
                lap=i+1,
                label='0',
                maze_type=trace['maze_type'],
                mouse=trace['MiceID'],
                spikes = trace['Spikes'][n, indices],
                spike_time= trace['ms_time_behav'][indices],
                spike_nodes2d=trace['spike_nodes'][indices]
            )
    
            smooth_maps1d[i, :] = trial12.calc_tuning_curve1d()
        
        im = ax.imshow(smooth_maps1d, cmap = 'jet')
        ax.set_aspect("auto")
        ax.set_yticks(ColorBarsTicks(lap-1, is_auto=True, tick_number=6), ColorBarsTicks(lap-1, is_auto=True, tick_number=6)+1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(smooth_maps1d), is_auto=True, tick_number=5))
        plt.show()
    """