import pandas as pd
from scipy.io import loadmat
from mylib.maze_utils3 import *
from matplotlib_venn import venn3, venn3_circles
from mylib.dp_analysis import field_arange, plot_field_arange, BehaviorEventsAnalyzer
from mylib.dp_analysis import plot_1day_line, plot_field_arange_all, FieldDisImage, ImageBase
from mylib.diff_start_point import DSPMazeLapSplit
from mylib.calcium.field_criteria import place_field, place_field_dsp
from mylib.field.in_field import set_range
from mylib import LocTimeCurveAxes

from mylib.behavior.behavevents import BehavEvents
from mylib.divide_laps.lap_split import LapSplit

from numba import jit
#  -------------------------------------------------------- Calsium ----------------------------------------------------------------------------
# In some cases (for example, mice 10019, date 4_20, neuron 23), there're some 'silent neurons' which has indistinct 
# firing activity and as a result, the size of spike_ind is 0 and could not be able to used for further shuffling. 
# So, whenever the size of spike_ind is equal to zero, stop shuffling and return 0, because the silent neuron 
# is actually not a place cell.
def isSilentNeuron(spikes):
    spike_ind = np.where(spikes==1)[0] # spike index
    if spike_ind.shape[0] == 0:
        return 0
    else:
        return 1

def Generate_SilentNeuron(Spikes, threshold = 30):
    spikes_num = np.nansum(Spikes, axis = 1)
    return np.where(spikes_num <= threshold)[0]

# ++++++++++++++++++++++++++++++++++++++++++++++ Shuffle Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# spatial information (all neurons)
# Shuffle ISI: DOI: 10.1523/JNEUROSCI.19-21-09497.1999

def calc_SI(
    spikes: np.ndarray, 
    rate_map: np.ndarray, 
    t_total: float, 
    t_nodes_frac: np.ndarray
) -> np.ndarray:
    mean_rate = np.nansum(spikes, axis = 1) / t_total # mean firing rate
    logArg = (rate_map.T / mean_rate).T;
    logArg[np.where(logArg == 0)] = 1; # keep argument in log non-zero

    IC = np.nansum(t_nodes_frac * rate_map * np.log2(logArg), axis = 1) # information content
    SI = IC / mean_rate; # spatial information (bits/spike)
    return(SI)

def calc_ratemap(Spikes = None, spike_nodes = None, _nbins = 48*48, occu_time = None, Ms = None, is_silent = None):
    Spikes = Spikes
    occu_time = occu_time
    spike_count = np.zeros((Spikes.shape[0], _nbins), dtype = np.float64)
    for i in range(_nbins):
        idx = np.where(spike_nodes == i+1)[0]
        spike_count[:,i] = np.nansum(Spikes[:,idx], axis = 1)

    rate_map_all = spike_count/(occu_time/1000+ 1E-9)
    if is_silent is not None:
        rate_map_all[is_silent,:] = np.nan
    clear_map_all, nanPos = clear_NAN(rate_map_all)
    smooth_map_all = np.dot(clear_map_all, Ms.T)
    return rate_map_all, clear_map_all, smooth_map_all, nanPos

def shuffle_test_isi(SI, spikes, spike_nodes, occu_time, shuffle_n = 1000, Ms = None, silent_cell = None, percent = 95):
    # spikes: numpy.ndarray, shape:(T,), T is the number of frames. The spikes of 1 neuron during recording.
    # spike_nodes: numpy.ndarray, shape:(T,), T is the number of frames. The spatial nodes of each spike recording frame.
    # occu_time: numpy,ndarray, shape:(2304,), is the behaviral occupation time at each spatial bin
    SI_rand = np.zeros(shuffle_n) # 1000 times
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6)
    spike_ind = np.where(spikes==1)[0] # spike index
    isi = np.append(spike_ind[0], np.ediff1d(spike_ind)) # get interspike interval

    # shuffle random variable
    shuffle_isi = np.zeros((shuffle_n, len(isi)), dtype = np.int64)

    for i in range(shuffle_n):
        shuffle_isi[i] = np.random.choice(isi, size = len(isi), replace = False) # shuffle interspike interval
    
    shuffle_spike_ind = np.cumsum(shuffle_isi, axis=1) # shuffled spike index
    spikes_rand = np.zeros((shuffle_n, spikes.shape[0]), dtype = np.int64) 

    for i in range(shuffle_n):
        spikes_rand[i, shuffle_spike_ind[i]] = 1
    
    smooth_map_rand, _, _, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms)
    SI_rand = calc_SI(spikes = spikes_rand, rate_map = smooth_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, percent)
    return is_placecell

def shuffle_test_shift(SI, spikes, spike_nodes, occu_time, shuffle_n = 1000, Ms = None, silent_cell = None, percent = 95):
    # spikes: numpy.ndarray, shape:(T,), T is the number of frames. The spikes of 1 neuron during recording.
    # spike_nodes: numpy.ndarray, shape:(T,), T is the number of frames. The spatial nodes of each spike recording frame.
    # occu_time: numpy,ndarray, shape:(2304,), is the behaviral occupation time at each spatial bin
    SI_rand = np.zeros(shuffle_n)
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6)

    shuffle_shift = np.random.randint(low = 0, high = spikes.shape[0], size = shuffle_n)
    spikes_rand = np.zeros((shuffle_n, spikes.shape[0]), dtype = np.int64)
    for i in range(shuffle_n):
        spikes_rand[i,:] = np.roll(spikes, shift = shuffle_shift[i])

    smooth_map_rand, _, _, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms)
    SI_rand = calc_SI(spikes = spikes_rand, rate_map = smooth_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, percent)
    return is_placecell

# pure random
def shuffle_test_all(SI, spikes, spike_nodes, occu_time, shuffle_n = 1000, Ms = None, silent_cell = None, percent = 95):
    SI_rand = np.zeros(shuffle_n)
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6)
    
    spikes_rand = np.zeros((shuffle_n, spikes.shape[0]), dtype = np.int64)
    spikes_temp = cp.deepcopy(spikes)

    for i in range(shuffle_n):
        np.random.shuffle(spikes_temp)
        spikes_rand[i, :] = cp.deepcopy(spikes_temp)
        
    smooth_map_rand, _, _, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms)
    SI_rand = calc_SI(spikes = spikes_rand, rate_map = smooth_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, percent)
    return is_placecell

def shuffle_test(trace, Ms = None, shuffle_n = 1000, percent = 95, save_loc: str = None, file_name: str = "Shuffle_Venn"):
    n_neuron = trace['n_neuron']
    SI_all = np.zeros(n_neuron, dtype = np.float64)
    is_placecell_isi = np.zeros(n_neuron, dtype = np.int64)
    is_placecell_shift = np.zeros(n_neuron, dtype = np.int64)
    is_placecell_all = np.zeros(n_neuron, dtype = np.int64)

    SI_all = calc_SI(trace['Spikes'], rate_map = trace['rate_map_all'], t_total = trace['t_total'], t_nodes_frac = trace['t_nodes_frac'])

    for i in tqdm(range(n_neuron)):
        if i in trace['SilentNeuron']:
            continue
        is_placecell_isi[i] = shuffle_test_isi(SI = SI_all[i], spikes = trace['Spikes'][i,], spike_nodes=trace['spike_nodes'], 
            occu_time=trace['occu_time_spf'], Ms = Ms, silent_cell = trace['SilentNeuron'], shuffle_n = shuffle_n, percent = percent)
        is_placecell_shift[i] = shuffle_test_shift(SI = SI_all[i], spikes = trace['Spikes'][i,], spike_nodes=trace['spike_nodes'], 
            occu_time=trace['occu_time_spf'], Ms = Ms, silent_cell = trace['SilentNeuron'], shuffle_n = shuffle_n, percent = percent)
        is_placecell_all[i] = shuffle_test_all(SI = SI_all[i], spikes = trace['Spikes'][i,], spike_nodes=trace['spike_nodes'], 
            occu_time=trace['occu_time_spf'], Ms = Ms, silent_cell = trace['SilentNeuron'], shuffle_n = shuffle_n, percent = percent)
    
    trace['SI_all'] = SI_all
    trace['is_placecell_isi'] = is_placecell_isi
    trace['is_placecell_shift'] = is_placecell_shift
    trace['is_placecell_all'] = is_placecell_all
    trace['is_placecell'] = np.zeros(n_neuron, dtype = np.int64)
    subset = Generate_Venn3_Subset(is_placecell_isi, is_placecell_shift, is_placecell_all)
    idx = np.where((is_placecell_all == 1)&(is_placecell_isi == 1)&(is_placecell_shift == 1))[0]
    trace['is_placecell'][idx] = 1
    print("      Number of place cell using time shift shuffle method:" ,np.sum(is_placecell_shift))
    print("      Number of place cell using time all shuffle method:", np.sum(is_placecell_all))
    print("      Number of place cell using time isi shuffle method:", np.sum(is_placecell_isi))
    print("      isi-shift overlapping:", subset[2])
    print("      isi-all overlapping:  ", subset[4])
    print("      shift-all overlapping:", subset[5])
    print("      All 3 overlapping:    ", subset[6])
    print("      Percentage of Place cells:", round(subset[6]/n_neuron*100, 4),"%")
    
    plt.figure(figsize = (6,6))
    if save_loc is None:
        save_loc = join(trace['p'], 'Shuffle_Venn')

    if not exists(save_loc):
        mkdir(save_loc)
        
    venn3(subsets = subset, set_labels = ('Inter Spike Intervle\nShuffle', 'Shift Spikes\nShuffle', 'Pure Random\nShuffle'))
    plt.savefig(join(save_loc, file_name+'.png'), dpi = 600)
    plt.savefig(join(save_loc, file_name+'.svg'), dpi = 600)
    print("    Done.")
    return trace


# ============================================== Draw Figures ====================================================================
# caculate firing rate in correct path and incorrect path
def FiringRateProcess(trace, map_type = 'smooth', spike_threshold = 30, occu_time = None):
    if map_type == 'smooth':
        nx = 48
        rate_map_all = cp.deepcopy(trace['smooth_map_all'])
        if occu_time is None:
            occu_time = cp.deepcopy(trace['occu_time_spf']) if 'occu_time_spf' in trace.keys() else cp.deepcopy(trace['occu_time'])
        spike_nodes = cp.deepcopy(trace['spike_nodes']) 
    elif map_type == 'old':
        nx = 12
        rate_map_all = cp.deepcopy(trace['old_map_clear'])

        if occu_time is None:
            occu_time = occu_time_transform(trace['occu_time_spf'], nx = 12) if 'occu_time_spf' in trace.keys() else occu_time_transform(trace['occu_time'], nx = 12)
        else:
            occu_time = occu_time_transform(occu_time, nx = 12)

        spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx = 12)
        trace['occu_time_old'] = occu_time
        trace['old_nodes'] = spike_nodes
    elif map_type == 'rate':
        nx =48
        rate_map_all = cp.deepcopy(trace['rate_map_clear'])
        if occu_time is None:
            occu_time = cp.deepcopy(trace['occu_time_spf']) if 'occu_time_spf' in trace.keys() else cp.deepcopy(trace['occu_time'])
    else:
        assert False

    Spikes = trace['Spikes']
    maze_type = trace['maze_type']
    silent_idx = trace['SilentNeuron']

    if nx == 48:
        Correct_Graph = Correct_SonGraph1 if maze_type == 1 else Correct_SonGraph2
        Incorrect_Graph = Incorrect_SonGraph1 if maze_type == 1 else Incorrect_SonGraph2
    elif nx == 12:
        Correct_Graph = CorrectPath_maze_1 if maze_type == 1 else CorrectPath_maze_2
        Incorrect_Graph = IncorrectPath_maze_1 if maze_type == 1 else IncorrectPath_maze_2
    

    # Mean Rate, Peak Rate
    n_neuron = Spikes.shape[0]
    mean_rate = np.zeros(n_neuron, dtype = np.float64)
    peak_rate = np.zeros(n_neuron, dtype = np.float64)    
    peak_rate = np.nanmax(rate_map_all, axis = 1)
    mean_rate = np.nansum(Spikes, axis = 1) / np.nansum(occu_time) * 1000
    # set silent cell as np.nan to avoid plotting.
    peak_rate[silent_idx] *= np.nan
    mean_rate[silent_idx] *= np.nan
    trace['mean_rate'] = mean_rate
    trace['peak_rate'] = peak_rate
    return trace

# draw old_map, trace_map, rate_map
def DrawOldMap(trace, ax = None):
    n_neuron = trace['n_neuron']
    old_map_clear = trace['old_map_clear']
    loc = trace['loc']
    
    for k in tqdm(range(n_neuron)):
        im = ax.imshow(np.reshape(old_map_clear[k],[12,12]))
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(old_map_clear[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title(f"Mice {trace['MiceID']}, Cell {k+1}\nSI = {round(trace['SI_all'][k],3)}",
                     color = color,fontsize = 16)
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        cbar.remove()
        im.remove()

def OldMap(trace, isDraw = True):
    Spikes = trace['Spikes']
    spike_nodes = trace['spike_nodes']
    occu_time = trace['occu_time_spf'] if 'occu_time_spf' in trace.keys() else trace['occu_time']
    maze_type = trace['maze_type']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['SilentNeuron'])

    trace['old_map_all'] = old_map_all
    trace['old_map_clear'] = old_map_clear
    trace['old_map_smooth'] = old_map_smooth
    trace['old_map_nanPos'] = old_map_nanPos
    trace['old_nodes'] = old_nodes
    trace['occu_time_old'] = occu_time_old

    if isDraw == False:
        return trace

    p = trace['p']
    loc = os.path.join(p,'OldMap')
    mkdir(loc)
    trace['loc'] = loc

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes())
        
    DrawMazeProfile(maze_type = maze_type, axes = ax, color = 'white', nx = 12, linewidth = 2)

    DrawOldMap(trace, ax = ax)
    plt.close()
    
    return trace

def DrawRateMap(trace, ax = None, cmap = 'jet'):
    n_neuron = len(trace['smooth_map_all'])
    smooth_map_all = trace['smooth_map_all']
    loc = trace['loc']
    
    for k in tqdm(range(n_neuron)):
        # rate map     
        im = ax.imshow(np.reshape(smooth_map_all[k],[48,48]), cmap = cmap)
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks([0, np.nanmax(smooth_map_all[k, :])])
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title(f"Mice {trace['MiceID']}, Cell {k+1}, SI = {round(trace['SI_all'][k],3)}",
                     color = color, fontsize = 12)
        ax.axis([-0.7, 47.7, 47.7, -0.7])
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        cbar.remove()
        im.remove()

def RateMap(trace: dict, cmap = 'jet', **kwargs) -> dict:
    maze_type = trace['maze_type']

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes())

    DrawMazeProfile(maze_type = maze_type, axes = ax, nx = 48, **kwargs)
    
    p = trace['p']
    loc = os.path.join(p,'RateMap')

    trace['loc'] = loc
    mkdir(loc)
    DrawRateMap(trace, ax = ax, cmap=cmap)
    plt.close()
    return trace

def RateMapIncludeIP(trace: dict) -> dict:
    if 'LA' not in trace.key():
        return trace
        
    maze_type = trace['maze_type']

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes())

    DrawMazeProfile(maze_type = maze_type, axes = ax, nx = 48)
    
    p = trace['p']
    loc = os.path.join(p,'RateMapIncludeIP')

    trace['loc'] = loc
    mkdir(loc)
    
    n_neuron = len(trace['LA']['smooth_map_all'])
    smooth_map_all = trace['LA']['smooth_map_all']
    loc = trace['loc']
    
    for k in tqdm(range(n_neuron)):
        # rate map     
        im = ax.imshow(np.reshape(smooth_map_all[k], [48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks([0, np.max(smooth_map_all[k, :])])
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['LA']['is_placecell'][k] == 1 else 'black'
        ax.set_title(f"Mice {trace['MiceID']}, Cell {k+1}, SI = {round(trace['LA']['SI_all'][k],3)}",
                     color = color, fontsize = 12)
        ax.axis([-0.7, 47.7, 47.7, -0.7])
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        cbar.remove()
        im.remove()

# Quarter Map
def DrawQuarterMap(trace, ax = None):
    n_neuron = trace['n_neuron']
    quarter_map_smooth = trace['quarter_map_smooth']
    loc = trace['loc']
    
    for k in tqdm(range(n_neuron)):
        # rate map     
        im = ax.imshow(np.reshape(quarter_map_smooth[k],[24,24]), cmap = 'jet')
        cbar = plt.colorbar(im, ax= ax)
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(quarter_map_smooth[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title(f"Mice {trace['MiceID']}, Cell {k+1}\nSI = {round(trace['SI_all'][k],3)}",
                     color = color, fontsize = 16)
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        cbar.remove()
        im.remove()

def QuarterMap(trace, isDraw = True):
    Spikes = trace['Spikes']
    spike_nodes = trace['spike_nodes']
    occu_time = trace['occu_time_spf'] if 'occu_time_spf' in trace.keys() else trace['occu_time']
    maze_type = trace['maze_type']
    
    quarter_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 24)
    occu_time_quarter = occu_time_transform(occu_time = occu_time, nx = 24)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 24, _range = 4, sigma = 2)
    quarter_map_all, quarter_map_clear, quarter_map_smooth, quarter_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = quarter_nodes, _nbins = 24*24, 
           occu_time = occu_time_quarter, Ms = Ms, is_silent = trace['SilentNeuron'])

    trace['quarter_map_all'] = quarter_map_all
    trace['quarter_map_clear'] = quarter_map_clear
    trace['quarter_map_smooth'] = quarter_map_smooth
    trace['quarter_map_nanPos'] = quarter_map_nanPos
    trace['quarter_nodes'] = quarter_nodes
    trace['occu_time_quarter'] = occu_time_quarter

    if isDraw == False:
        return trace

    p = trace['p']
    loc = os.path.join(p,'QuarterMap')
    mkdir(loc)
    trace['loc'] = loc

    plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes())

    DrawMazeProfile(maze_type = maze_type, axes = ax, color = 'white', nx = 24, linewidth = 2)
    DrawQuarterMap(trace, ax = ax)
    plt.close()
    
    return trace

def get_recent_behavior_time(behav_time: np.ndarray, t: float | int) -> int:
    bef = np.where((behav_time <= t)&(behav_time >= 0))[0]
    aft = np.where((behav_time >= t)&(behav_time >= 0))[0]
    
    if len(bef) == 0 and len(aft) != 0:
        return aft[0]
    elif len(aft) == 0 and len(bef) != 0:
        return bef[-1]
    elif len(aft) == 0 and len(bef) == 0:
        raise ValueError(f"{behav_time} encounter an invalid value {t}")
    else:
        t_bef = behav_time[bef[-1]]
        t_aft = behav_time[aft[0]]
        
        if np.abs(t_aft - t) >= np.abs(t - t_bef):
            return bef[-1]
        else:
            return aft[0]

# trace map
def DrawTraceMap(trace, ax: Axes):
    n_neuron = trace['n_neuron']
    maze_type = trace['maze_type']
    processed_pos, behav_time = Add_NAN(trace['correct_pos'], trace['correct_time'], maze_type = maze_type)
    processed_pos = position_transform(processed_pos = processed_pos, nx = 48, maxWidth = 960)
    ms_time_behav = trace['ms_time_behav']
    Spikes = trace['Spikes']
    loc = trace['loc']
    
    ax.plot(processed_pos[:,0], processed_pos[:,1], color = 'gray', linewidth=0.8)
    ax.invert_yaxis()
    DrawMazeProfile(maze_type = maze_type, axes = ax, color = 'brown', nx = 48, linewidth = 2)
    
    for k in tqdm(range(n_neuron)):
        # spike location
        spike_indices = np.where(Spikes[k,:] == 1)[0]
        time_indices = np.array([get_recent_behavior_time(behav_time, ms_time_behav[j]) for j in spike_indices], np.int64)
        a = []   
            
        b = ax.plot(
            processed_pos[time_indices, 0], 
            processed_pos[time_indices, 1], 
            'o', 
            label='events',
            color='black', 
            markersize=3, 
            markeredgewidth=0
        )
            
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title("Mice "+str(trace['MiceID'])+" , Cell "+str(k+1), 
                     color = color, 
                     fontsize = 16)
        ax.axis([-0.7, 47.7, 47.7, -0.7])
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        
        for s in b:
            s.remove()

def TraceMap(trace: dict) -> dict:
    p = trace['p']
    maze_type = trace['maze_type']
    loc = os.path.join(p,'TraceMap')
    mkdir(loc)
    trace['loc'] = loc

    fig = plt.figure(figsize = (4,4))
    ax = Clear_Axes(plt.axes())

    DrawTraceMap(trace, ax = ax)
    #ax.yaxis_inverted()
    plt.close()
    return trace


def TraceMapIncludeIP(trace: dict) -> dict:
    p = trace['p']
    maze_type = trace['maze_type']
    loc = os.path.join(p,'TraceMapIncludeIP')
    mkdir(loc)
    trace['loc'] = loc

    fig = plt.figure(figsize = (4,4))
    ax = Clear_Axes(plt.axes())

    n_neuron = trace['n_neuron']
    maze_type = trace['maze_type']
    processed_pos, behav_time = Add_NAN(trace['correct_pos'], trace['correct_time'], maze_type = maze_type)
    processed_pos = position_transform(processed_pos = processed_pos, nx = 48, maxWidth = 960)
    ms_time_behav = trace['LA']['ms_time_behav']
    Spikes = trace['LA']['Spikes']
    
    ax.plot(processed_pos[:,0], processed_pos[:,1], color = 'gray', linewidth=0.8)
    ax.invert_yaxis()
    DrawMazeProfile(maze_type = maze_type, axes = ax, color = 'brown', nx = 48, linewidth = 2)
    
    for k in tqdm(range(n_neuron)):
        # spike location
        spike_indices = np.where(Spikes[k,:] == 1)[0]
        time_indices = np.array([get_recent_behavior_time(behav_time, ms_time_behav[j]) for j in spike_indices], np.int64)
        a = []   
            
        b = ax.plot(
            processed_pos[time_indices, 0], 
            processed_pos[time_indices, 1], 
            'o', 
            label='events',
            color='black', 
            markersize=3, 
            markeredgewidth=0
        )
            
        color = 'red' if trace['LA']['is_placecell'][k] == 1 else 'black'
        ax.set_title("Mice "+str(trace['MiceID'])+" , Cell "+str(k+1), 
                     color = color, 
                     fontsize = 12)
        ax.axis([-0.7, 47.7, 47.7, -0.7])
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        
        for s in b:
            s.remove()

# population vector correlation
def PVCorrelationMap(trace):
    # population vector
    PVCorrelation = np.zeros((144,144),dtype = np.float64)
    PVpvalue = np.zeros((144,144),dtype = np.float64)
    
    old_map_all = cp.deepcopy(trace['old_map_all'])
    linear_map_all = LinearizationMap(old_map_all, maze_type = trace['maze_type'])
    linear_map_clear = clear_NAN(linear_map_all)[0] 

    maze_type = trace['maze_type']
    Correct_Graph = CorrectPath_maze_1 if maze_type == 1 else CorrectPath_maze_2
    length = len(Correct_Graph)
    
    loc = os.path.join(trace['p'],'SimpleMazeResults')
    mkdir(loc)
    
    for i in range(143):
        for j in range(i,144):
            PVCorrelation[i,j], PVpvalue[i,j] = pearsonr(linear_map_clear[:,i],linear_map_clear[:,j])
            PVCorrelation[j,i] = PVCorrelation[i,j]
            PVpvalue[j,i] = PVpvalue[i,j]
    
    trace['PVCorrelation'] = PVCorrelation
    trace['PVpvalue'] = PVpvalue
    trace['linear_map_all'] = linear_map_all
    trace['linear_map_clear'] = linear_map_clear
    
    fig = plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes(), xticks = [0,35,71,107,143], yticks = [0,35,71,107,143])
    
    ax.axvline(length-0.5, color = 'black')
    ax.axhline(length-0.5, color = 'black')
    
    im = ax.imshow(PVCorrelation, cmap = 'jet')
    ab = plt.colorbar(im, ax = ax)
    ax.set_title("Population Vector Correlation")
    plt.savefig(os.path.join(loc,"PV_Correlation.svg"),dpi = 600)
    plt.savefig(os.path.join(loc,"PV_Correlation.png"),dpi = 600)
    ab.remove()
    im.remove()
    
    im = ax.imshow(PVpvalue, cmap = 'jet')
    ab = plt.colorbar(im, ax = ax)
    ax.set_title("Population Vector Correlation p-Value")
    plt.savefig(os.path.join(loc,"PV_Correlation_pValue.svg"),dpi = 600)
    plt.savefig(os.path.join(loc,"PV_Correlation_pValue.png"),dpi = 600)
    plt.close()
    
    return trace

def LinearizationMap(old_map_all, maze_type = 1, nx = 12):
    print('        Notes: function "LinerizationMap" is developed for oldmap only, i.e. 12*12')
    if maze_type == 0 or nx != 12:
        print("    WARNING! Open field data do not need linearization!")
        return old_map_all
    
    ordered_path = xorders[maze_type]
    Linear_map_all = cp.deepcopy(old_map_all[:,ordered_path-1])
        
    return Linear_map_all

def CombineMap(trace):
    n_neuron = trace['n_neuron']
    Spikes = trace['Spikes']
    ms_time_behav = trace['ms_time_behav']
    processed_pos_new, behav_time = Add_NAN(trace['correct_pos'], trace['correct_time'], maze_type = trace['maze_type'])
    smooth_map_all = trace['smooth_map_all']
    old_map_clear = trace['old_map_clear']
    quarter_map_smooth = trace['quarter_map_smooth']
    maze_type = trace['maze_type']
    
    loc = os.path.join(trace['p'],'CombineMap')
    mkdir(loc)
    
    fig, axes = plt.subplots(2,2,figsize=(8,6))
    # trace map
    axes[0,0] = Clear_Axes(axes = axes[0][0])
    axes[0,1] = Clear_Axes(axes = axes[0][1])
    axes[1,0] = Clear_Axes(axes = axes[1][0])
    axes[1,1] = Clear_Axes(axes = axes[1][1])
    axes[0,0].plot(processed_pos_new[:,0]/20-0.5,processed_pos_new[:,1]/20-0.5,color = 'gray',zorder = 1, linewidth = 0.7)
    axes[0,1].invert_yaxis()
    axes[1,0].invert_yaxis()
    axes[1,1].invert_yaxis()
    
    # keep trace map 'equal'
    axes[0,0].set_aspect('equal')
    
    DrawMazeProfile(maze_type = maze_type, axes = axes[0,0], nx = 48, linewidth = 2, color = 'brown')
    DrawMazeProfile(maze_type = maze_type, axes = axes[0,1], nx = 48, linewidth = 2)
    DrawMazeProfile(maze_type = maze_type, axes = axes[1,0], nx = 12, linewidth = 2)
    DrawMazeProfile(maze_type = maze_type, axes = axes[1,1], nx = 24, linewidth = 2)

    for k in tqdm(range(n_neuron)):
        ims = []
        cbars = []        
        # TraceMap
        spike_idx = np.where(Spikes[k] == 1)[0]
        Time_idx = np.zeros_like(spike_idx)
        for j in range(len(spike_idx)):
            time = np.where(behav_time <= ms_time_behav[spike_idx[j]])[0]
            if len(time) == 0:
                Time_idx[j] = np.where(behav_time >= ms_time_behav[spike_idx[j]])[0][0]
            else:
                Time_idx[j] = time[-1]
            
        b = axes[0,0].plot(processed_pos_new[Time_idx,0]/20-0.5, processed_pos_new[Time_idx,1]/20-0.5,'o',
                            color = 'black',markersize = 2,zorder = 2)
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        axes[0,0].set_title(f"Mice {trace['MiceID']}, Cell {k+1}", color = color, fontsize = 16)
        
        # ratemap 48 x 48
        im = axes[0,1].imshow(np.reshape(smooth_map_all[k],[48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = axes[0,1])
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(smooth_map_all[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        axes[0,1].set_title(f"SI = {round(trace['SI_all'][k],3)}", color = color, fontsize = 16)
        ims.append(im)
        cbars.append(cbar)  
        
        # OldMap, 12 x 12
        im = axes[1,0].imshow(np.reshape(old_map_clear[k],[12,12]))
        cbar = plt.colorbar(im, ax = axes[1,0])
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(old_map_clear[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        axes[1,0].set_title(f"SI = {round(trace['SI_all'][k],2)}", color = color, fontsize = 16)
        ims.append(im)
        cbars.append(cbar)
        
        # QuarterMap, 24 x 24
        im = axes[1,1].imshow(np.reshape(quarter_map_smooth[k],[24,24]), cmap = 'jet')
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        cbar = plt.colorbar(im, ax = axes[1,1])
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(quarter_map_smooth[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        ims.append(im)
        cbars.append(cbar)
        
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        
        for j in cbars:
            j.remove()
        for j in b:
            j.remove()
        for j in ims:
            j.remove()
    plt.clf()
    plt.close()

def LocTimeCurve(trace):
    ms_time = cp.deepcopy(trace['ms_time_behav'])
    spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx=12)
    
    Graph = NRGs[int(trace['maze_type'])]
    linearized_x = np.zeros_like(trace['spike_nodes'], np.float64)

    for i in range(spike_nodes.shape[0]):
        linearized_x[i] = Graph[int(spike_nodes[i])-1]
        
    linearized_x = linearized_x + np.random.rand(spike_nodes.shape[0]) - 0.5

    fig = plt.figure(figsize=(4,6))
    ax = plt.axes()

    save_loc = os.path.join(trace['p'],'LocTimeCurve')
    mkdir(save_loc)
    
    CP = correct_paths[trace['maze_type']]
    
    t_max = int(np.nanmax(ms_time)/1000)

    for i in tqdm(range(trace['n_neuron'])):
        color = 'red' if trace['is_placecell'][i] == 1 else 'black'
        ax, _, _ = LocTimeCurveAxes(
            ax, 
            behav_time=ms_time,
            spikes=trace['Spikes'][i], 
            spike_time=ms_time, 
            maze_type=trace['maze_type'],
            given_x=linearized_x,
            title='Cell '+str(i+1),
            title_color=color,
        )
        ax.set_xlim([0, len(CP)+1])
        ax.set_ylim([0, t_max])
        
        colors = sns.color_palette("Spectral", len(trace['place_field_all'][i].keys()))
        for j, k in enumerate(trace['place_field_all'][i].keys()):
            lef, rig = set_range(trace['maze_type'], spike_nodes_transform(trace['place_field_all'][i][k], 12))
            lef += 0.5
            rig += 1.5
            ax.fill_betweenx(y=[0, t_max], x1=lef, x2 = rig, alpha=0.3, edgecolor=None, linewidth=0, color = colors[j])
            
        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)
        ax.clear()
            
    return trace

#======================================================================== Lap Analysis ================================================================================
# Cross Lap Correlation for cross maze paradigm
def CrossLapsCorrelation(trace, behavior_paradigm = 'CrossMaze'):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm)
    print('        Begin Index', beg_idx)
    print('        End Index', end_idx)
    _nbins = 2304
    _coords_range = [0, _nbins +0.0001 ]

    n_neuron = trace['n_neuron']
    laps = len(beg_idx)
    rate_map_all = np.zeros((laps, n_neuron, 2304), dtype = np.float64)
    clear_map_all = np.zeros((laps, n_neuron, 2304), dtype = np.float64)
    smooth_map_all = np.zeros((laps, n_neuron, 2304), dtype = np.float64)
    ms_idx_lap = []
    nanPos_all = []
    occu_time_all = np.zeros((laps, 2304), dtype = np.float64)
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 1, _range = 7, nx = 48)

    # Generate lap's rate map
    print("    Generate laps' rate map...")
    for k in tqdm(range(laps)):
        behav_time = cp.deepcopy(trace['correct_time'][beg_idx[k]:end_idx[k]+1])
        idx = np.where((trace['ms_time_behav'] <= behav_time[-1])&(trace['ms_time_behav'] >= behav_time[0]))[0]
        Spikes = cp.deepcopy(trace['Spikes'][:,idx])
        spike_nodes = cp.deepcopy(trace['spike_nodes'][idx])
        dt = cp.deepcopy(trace['dt'][idx])
        ms_idx_lap.append(idx)

        occu_time_all[k,:], _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            dt,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)

        rate_map_all[k,:,:], clear_map_all[k,:,:], smooth_map_all[k,:,:], nanPos = calc_ratemap(Spikes = Spikes, is_silent = trace['SilentNeuron'],
                                                                spike_nodes = spike_nodes, occu_time = occu_time_all[k,:], Ms = Ms)
        nanPos_all.append(nanPos)

    # Calculating cross lap correlation: laps x laps
    LapsCorrelationMatrix = np.zeros((n_neuron,laps,laps), dtype = np.float64)
    LapsCorrelationMean = np.zeros(n_neuron,dtype = np.float64)
    print("    Calculating Cross lap correlation...")
    for n in tqdm(range(n_neuron)):
        for i in range(laps-1):
            for j in range(i+1,laps):
                LapsCorrelationMatrix[n,i,j],_ = pearsonr(smooth_map_all[i,n,:], smooth_map_all[j,n,:])
                LapsCorrelationMatrix[n,j,i] = LapsCorrelationMatrix[n,i,j]
                LapsCorrelationMatrix[n,j,j] = 1
                LapsCorrelationMatrix[n,i,i] = 1

        LapsCorrelationMean[n] = (np.nansum(LapsCorrelationMatrix[n,:,:]) - laps) / (laps**2 - laps)
    
    trace['laps'] = laps
    trace['rate_map_split'] = rate_map_all
    trace['clear_map_split'] = clear_map_all
    trace['smooth_map_split'] = smooth_map_all
    trace['occu_time_split'] = occu_time_all
    trace['nanPos_split'] = nanPos_all
    trace['LapsCorrelationMatrix'] = LapsCorrelationMatrix
    trace['LapsCorrelationMean'] = LapsCorrelationMean
    trace['lap_begin_index'] = np.array(beg_idx, dtype = np.int64)
    trace['lap_end_index'] = np.array(end_idx, dtype = np.int64)
    trace['ms_idx_lap'] = ms_idx_lap

    return trace

def Delete_InterLapSpike(behav_time, ms_time, ms_speed, Spikes, spike_nodes, dt: np.ndarray, trace = None, 
                         behavior_paradigm = 'CrossMaze'):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm) # Get Split TimeIndex Point
    lap = len(beg_idx) # Number of inter-laps
    # behavspike index
    IDX = np.array([], dtype = np.int64)

    for k in range(lap):
        idx = np.where((ms_time >= behav_time[beg_idx[k]])&(ms_time <= behav_time[end_idx[k]]))[0]
        IDX = np.concatenate([IDX, idx])

    return cp.deepcopy(Spikes[:,IDX]), cp.deepcopy(spike_nodes[IDX]), cp.deepcopy(ms_time[IDX]), cp.deepcopy(ms_speed[IDX]), cp.deepcopy(dt[IDX])

def SimplePeakCurve(trace, is_ExistAxes = False, is_GetAxes = False, file_name = None, save_loc = None):
    maze_type = int(trace['maze_type'])
    old_map_all = cp.deepcopy(trace['old_map_clear'])

    # order the ratemap by mazeid
    if maze_type in [1, 2, 3]:
        co_path = correct_paths[maze_type]
        length = co_path.shape[0]
        order = xorders[maze_type]
        old_map_all = old_map_all[:,order-1]
    
    old_map_all = Norm_ratemap(old_map_all)
    old_map_all = sortmap(old_map_all)

    if is_ExistAxes == False:
        plt.figure(figsize = (8,6))
        ax = Clear_Axes(plt.axes(), xticks = np.linspace(0,144,17), ifyticks = True)
    im = ax.imshow(old_map_all, aspect = 'auto')
    ax.invert_yaxis()
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label("Normed Firing Rate / Hz")

    if maze_type in [1,2]:
        ax.axvline(length-0.5, color = 'cornflowerblue', linewidth = 2)

    if is_GetAxes == True:
        return ax
    plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi = 600)
    plt.close()


def plot_spike_monitor(a, b, c, d, save_loc: str or None):
    assert a.shape[0] == b.shape[0] and b.shape[0] == c.shape[0] and c.shape[0] == d.shape[0]
    n = a.shape[0]
    mkdir(save_loc)
    
    Data = {'class': np.concatenate([np.repeat('raw', n), np.repeat('NAN filter', n), np.repeat('speed filtered', n), np.repeat('delete interval', n)]),
            'x2': np.concatenate([np.repeat(1, n), np.repeat(2, n), np.repeat(3, n), np.repeat(4, n)]),
            'x': np.concatenate([np.repeat(1, n)+np.random.rand(n)*0.8-0.4, np.repeat(2, n)+np.random.rand(n)*0.8-0.4, np.repeat(3, n)+np.random.rand(n)*0.8-0.4, np.repeat(4, n)+np.random.rand(n)*0.8-0.4]),
            'number of spikes/cell': np.concatenate([a, b, c, d])}
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'x2', y = 'number of spikes/cell', data = Data, ax = ax)
    sns.scatterplot(x = 'x', y = 'number of spikes/cell', data = Data, hue = 'class', ax = ax, legend=False)
    plt.xticks([1,2,3, 4], ['raw', 'NAN\nfiltered', 'speed\nfiltered', 'delete\ninterval'])
    ax.set_xlabel("")
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_num.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_num.svg'), dpi = 600)
    plt.close()

    Data = {'class': np.concatenate([np.repeat('raw', n), np.repeat('speed filtered', n), np.repeat('speed filted', n), np.repeat('delete interval', n)]),
            'x2': np.concatenate([np.repeat(1, n), np.repeat(2, n), np.repeat(3, n), np.repeat(4, n)]),
            'x': np.concatenate([np.repeat(1, n)+np.random.rand(n)*0.8-0.4, np.repeat(2, n)+np.random.rand(n)*0.8-0.4, np.repeat(3, n)+np.random.rand(n)*0.8-0.4, np.repeat(4, n)+np.random.rand(n)*0.8-0.4]),
            'spikes remain rate / %': np.concatenate([a/a*100, b/a*100, c/a*100, d/a*100])}
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'x2', y = 'spikes remain rate / %', data = Data, ax = ax)
    sns.scatterplot(x = 'x', y = 'spikes remain rate / %', data = Data, hue = 'class', ax = ax, legend=False)
    plt.xticks([1,2,3, 4], ['raw', 'NAN\nfiltered', 'speed\nfiltered', 'delete\ninterval'])
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_xlabel("")
    ax.axis([0, 5, 0, 100])
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_rate.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_rate.svg'), dpi = 600)
    plt.close()
    print("    Figure spikes monitor is done.")

def OldMapSplit(trace = None):
    laps = trace['laps']
    n_neuron = trace['n_neuron']
    beg_idx = trace['lap_begin_index']
    end_idx = trace['lap_end_index']
    Spikes = cp.deepcopy(trace['Spikes'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    
    maze_type = trace['maze_type']
    
    old_split_all = np.zeros((laps,n_neuron,144), dtype = np.float64)
    old_split_clear = np.zeros((laps,n_neuron,144), dtype = np.float64)
    old_split_nanPos = []

    for k in range(laps):
        # Generate Old Map (Splited)
        occu_time = cp.deepcopy(trace['occu_time_split'][k])
        occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
        spike_lap = Spikes[:, beg_idx[k]:end_idx[k]+1]
        spike_nodes_lap = spike_nodes[beg_idx[k]:end_idx[k]+1]

        old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
        Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
        old_split_all[k,:,:], old_split_clear[k,:,:], _, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
                    occu_time = occu_time_old, Ms = Ms, is_silent = trace['SilentNeuron'])
        old_split_nanPos.append(old_map_nanPos)     
        
    trace['old_split_all'] = old_split_all
    trace['old_split_clear'] = old_split_clear
    trace['old_split_nanPos'] = old_split_nanPos
    return trace

def plot_split_trajectory(trace, behavior_paradigm = 'CrossMaze', split_args: dict = {}, **kwargs):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm, **split_args)
    print(beg_idx, end_idx)
    laps = len(beg_idx)
    save_loc = os.path.join(trace['p'], 'behav','laps_trajactory')
    mkdir(save_loc)
    behav_time = trace['correct_time']

    for k in tqdm(range(laps)):
        loc_x, loc_y = trace['correct_pos'][beg_idx[k]:end_idx[k]+1, 0] / 20 - 0.5, trace['correct_pos'][beg_idx[k]:end_idx[k]+1, 1] / 20 - 0.5
        fig = plt.figure(figsize = (6,6))
        ax = Clear_Axes(plt.axes())
        ax.set_title('Frame: '+str(beg_idx[k])+' -> '+str(end_idx[k])+'\n'+'Time:  '+str(behav_time[beg_idx[k]]/1000)+' -> '+str(behav_time[end_idx[k]]/1000))
        plot_trajactory(x = loc_x, y = loc_y, maze_type = trace['maze_type'], nx = 48, is_DrawMazeProfile = True, save_loc = save_loc, is_ExistAxes = True, ax = ax,
                        file_name = 'Lap '+str(k+1), is_inverty = True, **kwargs)

# transform time stamp index of lap-begin and lap-end from behavior to calcium.
def TransformSplitIndex(ms_time = None, correct_time = None, beg_idx = None, end_idx = None):
    beg_idx_ms = np.zeros_like(beg_idx, dtype = np.int64)
    end_idx_ms = np.zeros_like(end_idx, dtype = np.int64)

    for i in range(beg_idx.shape[0]):
        beg_idx_ms[i] = np.where(ms_time >= correct_time[beg_idx[i]])[0][0]
        end_idx_ms[i] = np.where(ms_time <= correct_time[end_idx[i]])[0][-1]

    return beg_idx_ms, end_idx_ms

# To calculate odd_even_map
def odd_even_correlation(trace):
    laps = trace['laps']
    ms_idx = trace['ms_idx_lap']
    if laps == 1:
        return trace
    beg, end = LapSplit(trace, trace['paradigm'])
    
    occu_time_all = cp.deepcopy(trace['occu_time_split'])
    Spikes = cp.deepcopy(trace['Spikes'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    odd_idx = np.where(np.arange(1,laps+1) % 2 == 1)[0]
    evn_idx = np.where(np.arange(1,laps+1) % 2 == 0)[0]
    occu_time_odd = np.nansum(occu_time_all[odd_idx,:], axis = 0)
    occu_time_evn = np.nansum(occu_time_all[evn_idx,:], axis = 0)

    Spikes_odd = np.concatenate([Spikes[:,ms_idx[k]] for k in odd_idx], axis = 1)
    Spikes_evn = np.concatenate([Spikes[:,ms_idx[k]] for k in evn_idx], axis = 1)

    spike_nodes_odd = np.concatenate([spike_nodes[ms_idx[k]] for k in odd_idx])
    spike_nodes_evn = np.concatenate([spike_nodes[ms_idx[k]] for k in evn_idx])

    rate_map_odd, clear_map_odd, smooth_map_odd, nanPos_odd = calc_ratemap(Spikes = Spikes_odd, spike_nodes = spike_nodes_odd, occu_time = occu_time_odd, 
                                                                           Ms = trace['Ms'], is_silent = trace['SilentNeuron'])
    rate_map_evn, clear_map_evn, smooth_map_evn, nanPos_evn = calc_ratemap(Spikes = Spikes_evn, spike_nodes = spike_nodes_evn, occu_time = occu_time_evn, 
                                                                           Ms = trace['Ms'], is_silent = trace['SilentNeuron'])
    
    odd_even_corr = np.zeros(Spikes.shape[0], dtype=np.float64)
    for i in range(Spikes.shape[0]):
        odd_even_corr[i], _ = scipy.stats.pearsonr(smooth_map_odd[i, :], smooth_map_evn[i, :])

    t_total_odd = np.nansum(occu_time_odd) / 1000
    t_total_evn = np.nansum(occu_time_evn) / 1000
    t_nodes_frac_odd = occu_time_odd / 1000 / (t_total_odd + 1E-6)
    t_nodes_frac_evn = occu_time_evn / 1000 / (t_total_evn + 1E-6)
    SI_odd = calc_SI(Spikes_odd, rate_map=smooth_map_odd, t_total=t_total_odd, t_nodes_frac=t_nodes_frac_odd)
    SI_evn = calc_SI(Spikes_evn, rate_map=smooth_map_evn, t_total=t_total_evn, t_nodes_frac=t_nodes_frac_evn)

    appendix = {'rate_map_odd':rate_map_odd, 'clear_map_odd':clear_map_odd, 'smooth_map_odd':smooth_map_odd, 'nanPos_odd':nanPos_odd, 'occu_time_odd':occu_time_odd,
                'rate_map_evn':rate_map_evn, 'clear_map_evn':clear_map_evn, 'smooth_map_evn':smooth_map_evn, 'nanPos_evn':nanPos_evn, 'occu_time_evn':occu_time_evn,
                't_total_odd': t_total_odd, 't_total_evn': t_total_evn, 't_nodes_frac_odd': t_nodes_frac_odd, 't_nodes_frac_evn':t_nodes_frac_evn, 'SI_odd': SI_odd,
                'SI_evn': SI_evn, 'odd_even_corr': odd_even_corr}
    trace.update(appendix)
    
    if 'LA' in trace.keys():
        Spikes = cp.deepcopy(trace['LA']['Spikes'])
        spike_nodes = cp.deepcopy(trace['LA']['spike_nodes'])
        spike_time = cp.deepcopy(trace['LA']['ms_time_behav'])
        dt = np.ediff1d(trace['LA']['ms_time_behav'])
        dt = np.append(dt, np.median(dt))
        dt[dt>100] = 100
    else:
        return trace

    _nbins = 2304
    _coords_range = [0, _nbins +0.0001]
    
    odd_indices = np.concatenate([np.where((spike_time >= beg[i]) & (spike_time <= end[i]))[0] for i in odd_idx])
    evn_indices = np.concatenate([np.where((spike_time >= beg[i]) & (spike_time <= end[i]))[0] for i in evn_idx])
    
    occu_time_odd, _, _ = scipy.stats.binned_statistic(
        spike_nodes[odd_indices],
        dt[odd_indices],
        bins=_nbins,
        statistic="sum",
        range=_coords_range)
    
    occu_time_evn, _, _ = scipy.stats.binned_statistic(
        spike_nodes[evn_indices],
        dt[evn_indices],
        bins=_nbins,
        statistic="sum",
        range=_coords_range
    )

    rate_map_odd, clear_map_odd, smooth_map_odd, nanPos_odd = calc_ratemap(
        Spikes = Spikes[:, odd_indices], 
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[odd_indices], 
        occu_time = occu_time_odd, 
        Ms = trace['Ms']
    )
    
    rate_map_evn, clear_map_evn, smooth_map_evn, nanPos_evn = calc_ratemap(
        Spikes = Spikes[:, evn_indices],
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[evn_indices],
        occu_time = occu_time_evn,
        Ms = trace['Ms']
    )
    
    fir_evn_corr = np.zeros(Spikes.shape[0], dtype=np.float64)
    for i in range(Spikes.shape[0]):
        fir_evn_corr[i], _ = scipy.stats.pearsonr(smooth_map_odd[i, :], smooth_map_evn[i, :])

    t_total_odd = np.nansum(occu_time_odd) / 1000
    t_total_evn = np.nansum(occu_time_evn) / 1000
    t_nodes_frac_odd = occu_time_odd / 1000 / (t_total_odd + 1E-6)
    t_nodes_frac_evn = occu_time_evn / 1000 / (t_total_evn + 1E-6)
    SI_odd = calc_SI(Spikes[:, odd_indices], rate_map=smooth_map_odd, t_total=t_total_odd, t_nodes_frac=t_nodes_frac_odd)
    SI_evn = calc_SI(Spikes[:, evn_indices], rate_map=smooth_map_evn, t_total=t_total_evn, t_nodes_frac=t_nodes_frac_evn)

    appendix = {'rate_map_odd':rate_map_odd, 'clear_map_odd':clear_map_odd, 'smooth_map_odd':smooth_map_odd, 'nanPos_odd':nanPos_odd, 'occu_time_odd':occu_time_odd,
                'rate_map_evn':rate_map_evn, 'clear_map_evn':clear_map_evn, 'smooth_map_evn':smooth_map_evn, 'nanPos_evn':nanPos_evn, 'occu_time_evn':occu_time_evn,
                't_total_odd': t_total_odd, 't_total_evn': t_total_evn, 't_nodes_frac_odd': t_nodes_frac_odd, 't_nodes_frac_evn':t_nodes_frac_evn, 'SI_odd': SI_odd,
                'SI_evn': SI_evn, 'fir_evn_corr': fir_evn_corr}
    trace['LA'].update(appendix)

    return trace

def half_half_correlation(trace):
    beg, end = LapSplit(trace, trace['paradigm'])
    
    mid = int(beg.shape[0]/2)
    t1, t2 = trace['correct_time'][beg[0]], trace['correct_time'][end[mid-1]]
    t3, t4 = trace['correct_time'][beg[mid]], trace['correct_time'][end[-1]]
    
    Spikes = cp.deepcopy(trace['Spikes'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    spike_time = cp.deepcopy(trace['ms_time_behav'])
    dt = np.ediff1d(trace['ms_time_behav'])
    dt = np.append(dt, np.median(dt))
    dt[dt>100] = 100
    
    _nbins = 2304
    _coords_range = [0, _nbins +0.0001]

    occu_time_fir, _, _ = scipy.stats.binned_statistic(
        spike_nodes[np.where((spike_time >= t1) & (spike_time <= t2))[0]],
        dt[np.where((spike_time >= t1) & (spike_time <= t2))[0]],
        bins=_nbins,
        statistic="sum",
        range=_coords_range)
    
    occu_time_sec, _, _ = scipy.stats.binned_statistic(
        spike_nodes[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        dt[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        bins=_nbins,
        statistic="sum",
        range=_coords_range
    )

    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 1, _range = 7, nx = 48)

    rate_map_fir, clear_map_fir, smooth_map_fir, nanPos_fir = calc_ratemap(
        Spikes = Spikes[:, np.where((spike_time >= t1) & (spike_time <= t2))[0]], 
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[np.where((spike_time >= t1) & (spike_time <= t2))[0]], 
        occu_time = occu_time_fir, 
        Ms = Ms
    )
    
    rate_map_sec, clear_map_sec, smooth_map_sec, nanPos_sec = calc_ratemap(
        Spikes = Spikes[:, np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        occu_time = occu_time_sec,
        Ms = Ms
    )
    
    fir_sec_corr = np.zeros(Spikes.shape[0], dtype=np.float64)
    for i in range(Spikes.shape[0]):
        fir_sec_corr[i], _ = scipy.stats.pearsonr(smooth_map_fir[i, :], smooth_map_sec[i, :])

    t_total_fir = np.nansum(occu_time_fir) / 1000
    t_total_sec = np.nansum(occu_time_sec) / 1000
    t_nodes_frac_fir = occu_time_fir / 1000 / (t_total_fir + 1E-6)
    t_nodes_frac_sec = occu_time_sec / 1000 / (t_total_sec + 1E-6)
    SI_fir = calc_SI(Spikes[:, np.where((spike_time >= t1) & (spike_time <= t2))[0]], rate_map=smooth_map_fir, t_total=t_total_fir, t_nodes_frac=t_nodes_frac_fir)
    SI_sec = calc_SI(Spikes[:, np.where((spike_time >= t3) & (spike_time <= t4))[0]], rate_map=smooth_map_sec, t_total=t_total_sec, t_nodes_frac=t_nodes_frac_sec)

    appendix = {'rate_map_fir':rate_map_fir, 'clear_map_fir':clear_map_fir, 'smooth_map_fir':smooth_map_fir, 'nanPos_fir':nanPos_fir, 'occu_time_fir':occu_time_fir,
                'rate_map_sec':rate_map_sec, 'clear_map_sec':clear_map_sec, 'smooth_map_sec':smooth_map_sec, 'nanPos_sec':nanPos_sec, 'occu_time_sec':occu_time_sec,
                't_total_fir': t_total_fir, 't_total_sec': t_total_sec, 't_nodes_frac_fir': t_nodes_frac_fir, 't_nodes_frac_sec':t_nodes_frac_sec, 'SI_fir': SI_fir,
                'SI_sec': SI_sec, 'fir_sec_corr': fir_sec_corr}
    trace.update(appendix)

    if 'LA' in trace.keys():
        Spikes = cp.deepcopy(trace['LA']['Spikes'])
        spike_nodes = cp.deepcopy(trace['LA']['spike_nodes'])
        spike_time = cp.deepcopy(trace['LA']['ms_time_behav'])
        dt = np.ediff1d(trace['LA']['ms_time_behav'])
        dt = np.append(dt, np.median(dt))
        dt[dt>100] = 100
    else:
        return trace
    
    occu_time_fir, _, _ = scipy.stats.binned_statistic(
        spike_nodes[np.where((spike_time >= t1) & (spike_time <= t2))[0]],
        dt[np.where((spike_time >= t1) & (spike_time <= t2))[0]],
        bins=_nbins,
        statistic="sum",
        range=_coords_range)
    
    occu_time_sec, _, _ = scipy.stats.binned_statistic(
        spike_nodes[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        dt[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        bins=_nbins,
        statistic="sum",
        range=_coords_range
    )

    rate_map_fir, clear_map_fir, smooth_map_fir, nanPos_fir = calc_ratemap(
        Spikes = Spikes[:, np.where((spike_time >= t1) & (spike_time <= t2))[0]], 
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[np.where((spike_time >= t1) & (spike_time <= t2))[0]], 
        occu_time = occu_time_fir, 
        Ms = Ms
    )
    
    rate_map_sec, clear_map_sec, smooth_map_sec, nanPos_sec = calc_ratemap(
        Spikes = Spikes[:, np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        is_silent = trace['SilentNeuron'],
        spike_nodes = spike_nodes[np.where((spike_time >= t3) & (spike_time <= t4))[0]],
        occu_time = occu_time_sec,
        Ms = Ms
    )
    
    fir_sec_corr = np.zeros(Spikes.shape[0], dtype=np.float64)
    for i in range(Spikes.shape[0]):
        fir_sec_corr[i], _ = scipy.stats.pearsonr(smooth_map_fir[i, :], smooth_map_sec[i, :])

    t_total_fir = np.nansum(occu_time_fir) / 1000
    t_total_sec = np.nansum(occu_time_sec) / 1000
    t_nodes_frac_fir = occu_time_fir / 1000 / (t_total_fir + 1E-6)
    t_nodes_frac_sec = occu_time_sec / 1000 / (t_total_sec + 1E-6)
    SI_fir = calc_SI(Spikes[:, np.where((spike_time >= t1) & (spike_time <= t2))[0]], rate_map=smooth_map_fir, t_total=t_total_fir, t_nodes_frac=t_nodes_frac_fir)
    SI_sec = calc_SI(Spikes[:, np.where((spike_time >= t3) & (spike_time <= t4))[0]], rate_map=smooth_map_sec, t_total=t_total_sec, t_nodes_frac=t_nodes_frac_sec)

    appendix = {'rate_map_fir':rate_map_fir, 'clear_map_fir':clear_map_fir, 'smooth_map_fir':smooth_map_fir, 'nanPos_fir':nanPos_fir, 'occu_time_fir':occu_time_fir,
                'rate_map_sec':rate_map_sec, 'clear_map_sec':clear_map_sec, 'smooth_map_sec':smooth_map_sec, 'nanPos_sec':nanPos_sec, 'occu_time_sec':occu_time_sec,
                't_total_fir': t_total_fir, 't_total_sec': t_total_sec, 't_nodes_frac_fir': t_nodes_frac_fir, 't_nodes_frac_sec':t_nodes_frac_sec, 'SI_fir': SI_fir,
                'SI_sec': SI_sec, 'fir_sec_corr': fir_sec_corr}
    trace['LA'].update(appendix)
    
    return trace

def calc_ms_speed(behav_speed: np.ndarray, behav_time: np.ndarray, 
                  ms_time: np.ndarray, 
                  time_thre: float = 500) -> np.ndarray:
    """calc_ms_speed 
    Calculate the speed of mice according with the 
      time stamp of the imaging

    Parameters
    ----------
    behav_speed : np.ndarray
        The speed of mice at each recorded frame.
    behav_time : np.ndarray
        The time stamps of the behavioral frames.
    Spikes : np.ndarray
        The deconvolved spikes of each cells recorded.
    ms_time : np.ndarray
        The time stamps of the imaging data.
    time_thre: float
        The tolerable threshold at the time course between behavioral 
          frames and imaging frames.

    Returns
    -------
    np.ndarray
        with the same shape as behav_time.
    """
    assert behav_speed.shape[0] == behav_time.shape[0]

    ms_speed = np.zeros_like(ms_time, dtype = np.float64)
    
    for i in range(ms_speed.shape[0]):
        if ms_time[i] <= behav_time[0]:
            continue
        else:
            idx = np.where(behav_time < ms_time[i])[0][-1]
            
            if ms_time[i] - behav_time[idx] <= time_thre:
                ms_speed[i] = behav_speed[idx]
                
    return ms_speed

def calc_coverage(processed_pos, xbin = 12, ybin = 12, maxHeight = 960, maxWidth = 960):
    x = processed_pos[:, 0] // (maxWidth/xbin)
    y = processed_pos[:, 1] // (maxHeight/ybin)
    id = x + y * xbin + 1
    id = id.astype(np.int64)

    num = 0
    for i in range(1, xbin*ybin+1):
        if len(np.where(id == i)[0]) != 0:
            num += 1
    
    return num/(xbin*ybin)

def coverage_curve(processed_pos, srt = 5, end = 49, save_loc: str or None = None, maze_type = 0):
    coverage = np.zeros(end-srt)
    mkdir(save_loc)
    print("    Calculating coverage curve:")
    for size in tqdm(range(srt, end)):
        coverage[size-srt] = calc_coverage(processed_pos, size, size)

    fig = plt.figure(figsize=(4, 3))
    length = 100 if maze_type == 0 else 96
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks= True, ifyticks=True)

    ax.axvline(length/12, ls = '--', color = 'grey', linewidth = 1)
    ax.axvline(length/24, ls = '--', color = 'grey', linewidth = 1)
    ax.axvline(length/48, ls = '--', color = 'grey', linewidth = 1)
    ax.axhline(coverage[12-srt]*100, ls = '--', color = 'grey', linewidth = 1)
    ax.axhline(coverage[24-srt]*100, ls = '--', color = 'grey', linewidth = 1)
    ax.axhline(coverage[48-srt]*100, ls = '--', color = 'grey', linewidth = 1)
    ax.text(length/12, coverage[12-srt]*100, f"12, {str(round(coverage[12-srt]*100, 2))}%")
    ax.text(length/24, coverage[24-srt]*100, f"24, {str(round(coverage[24-srt]*100, 2))}%")
    ax.text(length/48, coverage[48-srt]*100, f"48, {str(round(coverage[48-srt]*100, 2))}%")

    ax.plot(length/np.arange(srt, end), coverage*100, marker='^', markersize = 1, markerfacecolor='orange', markeredgecolor = 'orange')

    ax.set_ylabel('Coverage / %')
    ax.set_xlabel('bin size / cm')
    ax.set_title('Coverage curve')
    ax.set_xticks(np.linspace(0,20,5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, 'coverage_curve.svg'), dpi = 1800)
    plt.savefig(os.path.join(save_loc, 'coverage_curve.png'), dpi = 1800)
    plt.close()

    return coverage
        
def get_neighbor_bins(id, maze_type: int):
    curr = S2F[id-1] # curr nodes
    G = maze1_graph if maze_type == 1 else maze2_graph
    surr = G[curr]
    surr.append(curr)
    
    bins_vec = []
    for s in surr:
        bins_vec = bins_vec + Father2SonGraph[s]
    
    return np.array(bins_vec, dtype = np.int64)
    

def field_specific_correlation(trace):
    in_field_corr = []
    n = trace['n_neuron']
    place_field_all = trace['place_field_all']
    
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
        
    if trace['laps'] == 1:
        for i in range(n):
            corr = {}
            ks = place_field_all[i].keys()
            for k in ks:
                corr[k] = (np.nan, np.nan)
            in_field_corr.append(corr)
        
        trace['in_field_corr'] = in_field_corr
        return trace
        
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
    
    for i in range(n):
        ks = place_field_all[i].keys()
        corr = {}
        for k in ks:
            idx = get_neighbor_bins(k, trace['maze_type']) - 1
            corr[k] = (pearsonr(trace['smooth_map_odd'][i][idx],
                                         trace['smooth_map_evn'][i][idx])[0],
                                pearsonr(trace['smooth_map_fir'][i][idx],
                                         trace['smooth_map_sec'][i][idx])[0])
        in_field_corr.append(corr)
            
    trace['in_field_corr'] = in_field_corr
    return trace
    
from mylib.field.in_field import InFieldRateChangeModel, set_range
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from mylib import LinearizedRateMapAxes, InstantRateCurveAxes, LocTimeCurveAxes

def get_fatherfield_range(fatherfield: list | np.ndarray, maze_type: int):
    CP = correct_paths[maze_type]
    IP = incorrect_paths[maze_type]
    xorder = xorders[maze_type]

    field_at_cp, field_at_ip = [], []
    cp_idx, ip_idx = [], []
    for i in fatherfield:
        if i in CP:
            field_at_cp.append(i)
            cp_idx.append(np.where(CP == i)[0][0])
        elif i in IP:
            field_at_ip.append(i)
            ip_idx.append(np.where(xorder == i)[0][0])

    cp_idx, ip_idx = np.array(cp_idx, dtype=np.int64), np.array(ip_idx, dtype=np.int64)

    if len(cp_idx) != 0:
        return np.nanmin(cp_idx), np.nanmax(cp_idx)
    else:
        return np.nanmin(ip_idx), np.nanmax(ip_idx)


def get_spike_frame_label(ms_time, spike_nodes, trace = None, behavior_paradigm = 'CrossMaze', **kwargs):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm) # Get Split TimeIndex Point
    lap = len(beg_idx) # Number of inter-laps
    # behav spike index
    frame_labels = np.array([], dtype=np.int64)

    for k in range(lap):
        beg, end = np.where(ms_time >= trace['correct_time'][beg_idx[k]])[0][0], np.where(ms_time <= trace['correct_time'][end_idx[k]])[0][-1]
        labels = BehavEvents.get_frame_labels(spike_nodes[beg:end+1], trace['maze_type'], **kwargs)
        frame_labels = np.concatenate([frame_labels, labels])

    return frame_labels

def ComplexFieldAnalyzer(trace: dict, is_draw: bool = True, shuffle_times = 5000, **kwargs) -> dict:
    Spikes = cp.deepcopy(trace['Spikes'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    old_map_clear = cp.deepcopy(trace['old_map_clear'])
    ms_time = cp.deepcopy(trace['ms_time_behav'])
    n_neuron = trace['n_neuron']

    # Generate place field
    trace['place_field_all'] = place_field(
        trace=trace,
        thre_type=2,
        parameter=0.4,
    )

    place_field_all = cp.deepcopy(trace['place_field_all'])

    maze_type = trace['maze_type']
    save_loc = join(trace['p'], 'ComplexLocTimeCurve')
        
    Graph = NRG[int(maze_type)]

    frame_labels = get_spike_frame_label(
        ms_time=cp.deepcopy(trace['correct_time']), 
        spike_nodes=cp.deepcopy(trace['correct_nodes']),
        trace=trace, 
        behavior_paradigm='CrossMaze',
        window_length = 1
    )
    
    behav_indices = np.where(frame_labels==1)[0]
    old_nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)[behav_indices]
    
    if exists(save_loc):
        shutil.rmtree(save_loc)
        mkdir(save_loc)
    
    place_fields_info = {}
    for i in tqdm(range(n_neuron)):
        for j, k in enumerate(place_field_all[i].keys()):
            father_field = SF2FF(place_field_all[i][k])
            try:
                model = InFieldRateChangeModel.analyze_field(
                    trace, 
                    i, 
                    father_field, 
                    behav_time=cp.deepcopy(trace['correct_time'])[behav_indices], 
                    behav_nodes=old_nodes, 
                    behav_pos=cp.deepcopy(trace['correct_pos'])[behav_indices, :],
                    shuffle_times=shuffle_times,
                    **kwargs
                )
                if is_draw:
                    if os.path.exists(join(save_loc, "Cell "+str(i+1))) == False:
                        os.mkdir(join(save_loc, "Cell "+str(i+1)))
                
                    model.visualize_shuffle_result(save_loc=join(save_loc, "Cell "+str(i+1)), file_name=f'field {j+1} - ')
                place_fields_info[(i, j)] = cp.deepcopy(model.get_info())
            except:
                print(i, j)
                pass

    trace['place_fields_info'] = cp.deepcopy(place_fields_info)
    
    if is_draw == False:    
        return trace
    
    linearized_x = np.zeros_like(old_nodes, np.float64)

    for i in range(old_nodes.shape[0]):
        linearized_x[i] = Graph[int(old_nodes[i])] 
               
    CP = correct_paths[maze_type]
    
    linearized_x = linearized_x + np.random.rand(old_nodes.shape[0]) - 0.5

    MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)
    MRIG = gaussian_smooth_matrix1d(1000, window = 40, sigma=3, folder=0.001)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,7.5), gridspec_kw={'width_ratios':[4,1], 'height_ratios':[1,4]})
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0,1], axes[1,0], axes[1,1]
    ax2 = Clear_Axes(ax2)

    print("Plot figures")
    for k in tqdm(place_fields_info.keys()):

        loc = join(save_loc, "Cell "+str(k[0]+1))
        if os.path.exists(loc) == False and is_draw:
            os.mkdir(loc)

        ax1, a = LinearizedRateMapAxes(
            ax=ax1,
            content=old_map_clear[k[0]],
            maze_type=maze_type,
            M=MTOP
        )
        
        ax1.set_xlim([0, len(CP)+1])

        y_max = np.nanmax(old_map_clear[k[0]])

        ax3, b, c = LocTimeCurveAxes(
            ax=ax3, 
            behav_time=cp.deepcopy(trace['correct_time'][behav_indices]), 
            spikes=Spikes[k[0], :], 
            spike_time=ms_time, 
            maze_type=maze_type, 
            given_x=linearized_x
        )
        ax3.set_xlim([0, len(CP)+1])

        t_max = ms_time[-1]/1000

        lef, rig, fds = np.zeros(len(place_field_all[k[0]].keys()), dtype=np.float64), np.zeros(len(place_field_all[k[0]].keys()), dtype=np.float64), np.zeros(len(place_field_all[k[0]].keys()), dtype=np.int64)
        for j, key in enumerate(place_field_all[k[0]].keys()):
            father_field = SF2FF(place_field_all[k[0]][key])
            lef[j], rig[j] = set_range(maze_type=maze_type, field=father_field)
            lef[j] = lef[j] + 0.5
            rig[j] = rig[j] + 1.5


        colors = sns.color_palette("rainbow", 10) if lef.shape[0] < 10-4 else sns.color_palette("rainbow", lef.shape[0]+4)
        for j in range(lef.shape[0]):
            temp = ax1.plot([lef[j], rig[j]], [-y_max*0.09, -y_max*0.09], color = colors[j+2], linewidth=0.8)
            a = a + temp
          
        shad = ax3.fill_betweenx(y = [0, t_max], x1=lef[k[1]], x2=rig[k[1]], color = colors[k[1]+2], edgecolor=None, alpha=0.5)
            
        ax4 = InstantRateCurveAxes(
            ax=ax4,
            time_stamp=place_fields_info[k]['cal_events_time'],
            content=place_fields_info[k]['cal_events_rate'],
            M=MRIG,
            t_max=t_max*1000,
            title=place_fields_info[k]['ctype']
        )

        fill_list = []
        fill_list.append(shad)
        for err in place_fields_info[k]['err_events']:
            z = ax3.fill_betweenx([err[2]/1000-5, err[3]/1000+5], x1=lef[k[1]], x2=rig[k[1]], color = 'grey', edgecolor = None, alpha=0.8)
            fill_list.append(z)

        plt.savefig(join(loc, "field "+str(k[1]+1)+'.png'), dpi=600)
        plt.savefig(join(loc, "field "+str(k[1]+1)+'.svg'), dpi=600)
            
        ax4.clear()

        for r in a+b+c+fill_list:
            r.remove()

    return trace

def add_perfect_lap(trace: dict) -> dict:
    if trace['maze_type'] == 0:
        return trace
    
    beg, end = LapSplit(trace, trace['paradigm'])
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    D = GetDMatrices(trace['maze_type'], 12)
    CP = cp.deepcopy(correct_paths[trace['maze_type']])
    
    is_perfect = np.ones(beg.shape[0], np.int64)
    
    for i in range(beg.shape[0]):
        for j in range(beg[i]+1, end[i]):
            if behav_nodes[j] not in CP:
                is_perfect[i] = 0
                break
            
            if int(D[behav_nodes[j]-1, 0]*100) < int(D[behav_nodes[j-1]-1, 0]*100):
                is_perfect[i] = 0
                break
    
    trace['is_perfect'] = is_perfect
    return trace

def count_field_number(trace: dict) -> dict:
    if 'place_field_all' not in trace.keys():
        raise ValueError("No 'place_field_all' in trace.")
    
    field_number = np.zeros(trace['Spikes'].shape[0], np.float64)
    for i in tqdm(range(trace['Spikes'].shape[0])):
        if trace['is_placecell'][i] == 0:
            field_number[i] = np.nan
        else:
            num = len(trace['place_field_all'][i].keys())
            field_number[i] = num
    
    print("        Field Number Has Been Counted.")
    trace['place_field_num'] = field_number
    return trace

def field_register(trace: dict, key: str = 'place_field_all') -> dict:
    is_pc = cp.deepcopy(trace['is_placecell'])
    
    # Initial maps
    field_reg = []
    for i in range(is_pc.shape[0]):
        if is_pc[i] == 1:
            for j, k in enumerate(trace[key][i].keys()):
                if trace[key][i][k].shape[0] <= 16:
                    fsc, oec = np.nan, np.nan
                else:
                    fsc = pearsonr(trace['smooth_map_fir'][i, trace[key][i][k]-1], trace['smooth_map_sec'][i, trace[key][i][k]-1])[0]
                    oec = pearsonr(trace['smooth_map_odd'][i, trace[key][i][k]-1], trace['smooth_map_evn'][i, trace[key][i][k]-1])[0]
                rate = trace['smooth_map_all'][i][k-1] if 'smooth_map_all' in trace.keys() else np.nan
                field_reg.append([i, j, k, len(trace[key][i][k]), rate, fsc, oec])
    
    trace['field_reg'] = np.array(field_reg, np.float64)
    return trace

# ------------------------------------------------------------------------ 主程序 -------------------------------------------------------------------------------------
def run_all_mice(p = None, folder = None, behavior_paradigm = 'CrossMaze', v_thre: float = 2.5, **speed_sm_args):
    if behavior_paradigm not in ['CrossMaze','ReverseMaze', 'DSPMaze', 'PinMaze', 'SimpleMaze']:
        raise ValueError(f"{behavior_paradigm} is invalid! Only 'CrossMaze','ReverseMaze', 'SimpleMaze, 'DSPMaze' and 'PinMaze' are valid.")

    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    
    if os.path.exists(os.path.join(p,'trace_behav.pkl')):
        with open(os.path.join(p, 'trace_behav.pkl'), 'rb') as handle:
            trace = pickle.load(handle)
    else:
        warnings.warn(f"{os.path.join(p,'trace_behav.pkl')} is not found!")
        return
    
    trace['p'] = p

    coverage = coverage_curve(trace['processed_pos_new'], maze_type=trace['maze_type'], save_loc=os.path.join(p, 'behav'))
    trace['coverage'] = coverage
    trace = add_perfect_lap(trace)

    # Read File
    print("    A. Read ms.mat File")
    assert(os.path.exists(folder))
    if behavior_paradigm == 'CrossMaze':
        ms_mat = loadmat(folder)
        ms = ms_mat['ms']
        #FiltTraces = np.array(ms['FiltTraces'][0][0]).T
        RawTraces = np.array(ms['RawTraces'][0][0]).T
        DeconvSignal = np.array(ms['DeconvSignals'][0][0]).T
        ms_time = np.array(ms['time'][0])[0,].T[0]

    if behavior_paradigm in ['SimpleMaze', 'ReverseMaze','DSPMaze']:
        with h5py.File(folder, 'r') as f:
            ms_mat = f['ms']
            FiltTraces = np.array(ms_mat['FiltTraces'])
            RawTraces = np.array(ms_mat['RawTraces'])
            DeconvSignal = np.array(ms_mat['DeconvSignals'])
            ms_time = np.array(ms_mat['time'],dtype = np.int64)[0,]

    plot_split_trajectory(trace, behavior_paradigm = behavior_paradigm)

    print("    B. Calculate putative spikes and correlated location from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 3)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # calc ms speed
    behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
    smooth_speed = uniform_smooth_speed(behav_speed, **speed_sm_args)
    
    ms_speed = calc_ms_speed(behav_speed=smooth_speed, behav_time=trace['correct_time'], ms_time=ms_time)

    # Delete NAN value in spike nodes
    print("      - Delete NAN values in data.")
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    Spikes = cp.deepcopy(Spikes_original[:,idx])
    spike_nodes = cp.deepcopy(spike_nodes_original[idx])
    ms_time_behav = cp.deepcopy(ms_time[idx])
    ms_speed_behav = cp.deepcopy(ms_speed[idx])
    dt = np.append(np.ediff1d(ms_time_behav), 33)
    dt[np.where(dt >= 100)[0]] = 100
    
    # Filter the speed (spf: speed filter)
    print(f"      - Filter spikes with speed {v_thre} cm/s.")
    spf_idx = np.where(ms_speed_behav >= v_thre)[0]
    spf_results = [ms_speed_behav.shape[0], spf_idx.shape[0]]
    print(f"        {spf_results[0]} frames -> {spf_results[1]} frames.")
    print(f"        Remain rate: {round(spf_results[1]/spf_results[0]*100, 2)}%")
    print(f"        Delete rate: {round(100 - spf_results[1]/spf_results[0]*100, 2)}%")
    ms_time_behav = ms_time_behav[spf_idx]
    Spikes = Spikes[:, spf_idx]
    spike_nodes = spike_nodes[spf_idx]
    ms_speed_behav = ms_speed_behav[spf_idx]
    dt = dt[spf_idx]
    spike_num_mon2 = np.nansum(Spikes, axis = 1)
    
    # Delete InterLap Spikes
    print("      - Delete the inter-lap spikes.")
    Spikes, spike_nodes, ms_time_behav, ms_speed_behav, dt = Delete_InterLapSpike(behav_time = trace['correct_time'], ms_time = ms_time_behav, 
                                                                              Spikes = Spikes, spike_nodes = spike_nodes, dt = dt, ms_speed=ms_speed_behav,
                                                                              behavior_paradigm = behavior_paradigm, trace = trace)
    n_neuron = Spikes.shape[0]
    spike_num_mon3 = np.nansum(Spikes, axis = 1)
    
    plot_spike_monitor(spike_num_mon1, spike_num_mon2, spike_num_mon3, save_loc = os.path.join(trace['p'], 'behav'))

    print("    C. Calculating firing rate for each neuron and identified their place fields (those areas which firing rate >= 50% peak rate)")
    # Set occu_time <= 50ms spatial bins as nan to avoid too big firing rate
    _nbins = 2304
    _coords_range = [0, _nbins +0.0001 ]

    occu_time, _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            dt,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)

    # Generate silent neuron
    SilentNeuron = Generate_SilentNeuron(Spikes = Spikes, threshold = 30)
    print('       These neurons have spikes less than 30:', SilentNeuron)
    # Calculating firing rate
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 2, _range = 7, nx = 48)
    rate_map_all, rate_map_clear, smooth_map_all, nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = spike_nodes,
                                                                        _nbins = 48*48, occu_time = occu_time, Ms = Ms, is_silent = SilentNeuron)   
    
    print("    D. Shuffle test for spatial information of each cells to identified place cells. Shuffle method including 1) inter spike intervals(isi), 2) rigid spike shifts, 3) purely random rearrangement of spikes.")
    # total occupation time
    t_total = np.nansum(occu_time)/1000
    # time fraction at each spatial bin
    t_nodes_frac = occu_time / 1000 / (t_total+ 1E-6)

    # Save all variables in a dict
    trace_ms = {'Spikes_original':Spikes_original, 'spike_nodes_original':spike_nodes_original, 'ms_speed_original': ms_speed, 'RawTraces':RawTraces,'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'Spikes':Spikes, 'spike_nodes':spike_nodes, 'ms_time_behav':ms_time_behav, 'ms_speed_behav':ms_speed_behav, 'n_neuron':n_neuron, 
                't_total':t_total, 'dt': dt, 't_nodes_frac':t_nodes_frac, 'SilentNeuron':SilentNeuron, 'rate_map_all':rate_map_all, 'rate_map_clear':rate_map_clear, 
                'smooth_map_all':smooth_map_all, 'nanPos':nanPos, 'Ms':Ms, 'ms_folder':folder, 'occu_time_spf': occu_time, 'speed_filter_results': spf_results}
    trace.update(trace_ms)

    # Generate place field
    trace['place_field_all'] = place_field(
        trace=trace,
        thre_type=2,
        parameter=0.4,
    )

    # Shuffle test
    trace = shuffle_test(trace, trace['Ms'])
    plot_field_arange(trace, save_loc=os.path.join(trace['p'], 'PeakCurve'))

    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("    Plotting:")
    print("      1. Ratemap")
    #RateMap(trace)
    
    print("      2. Tracemap")
    #TraceMap(trace)      
      
    print("      3. Quarter_map")
    trace = QuarterMap(trace, isDraw = False)
    
    print("      4. Oldmap")
    trace = OldMap(trace, isDraw=False)
    
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("      5. PeakCurve")
    mkdir(os.path.join(trace['p'], 'PeakCurve'))
    SimplePeakCurve(trace, file_name = 'PeakCurve', save_loc = os.path.join(trace['p'], 'PeakCurve'))
    
    print("      6. Combining tracemap, rate map(48 x 48), old map(12 x 12) and quarter map(24 x 24)")
    CombineMap(trace)
    
    if trace['maze_type'] != 0:
        # LocTimeCurve
        print("      7. LocTimeCurve:")
        LocTimeCurve(trace) 
        print("    Analysis:")
        print("      A. Calculate Population Vector Correlation")
        #population vector correaltion
        trace = PVCorrelationMap(trace)

    
    # Firing Rate Processing:
    print("      B. Firing rate Analysis")
    #trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = 30)
    
    # Cross-Lap Analysis
    if behavior_paradigm in ['ReverseMaze', 'CrossMaze', 'DSPMaze']:
        print("      C. Cross-Lap Analysis")
        trace = CrossLapsCorrelation(trace, behavior_paradigm = behavior_paradigm)
        print("      D. Old Map Split")
        trace = OldMapSplit(trace)
        print("      E. Calculate Odd-even Correlation")
        trace = odd_even_correlation(trace)
        print("      F. Calculate Half-half Correlation")
        trace = half_half_correlation(trace)
        print("      G. In Field Correlation")
        trace = field_specific_correlation(trace)
    
    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")
    
    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)


if __name__ == '__main__':
    import pickle

    with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
