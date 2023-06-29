import pandas as pd
from scipy.io import loadmat
from mylib.maze_utils3 import *
from matplotlib_venn import venn3, venn3_circles
from mylib.dp_analysis import field_arange, plot_field_arange, BehaviorEvents, BehaviorEventsAnalyzer
from mylib.dp_analysis import plot_1day_line, plot_field_arange_all, FieldDisImage, ImageBase
from mylib.diff_start_point import DSPMazeLapSplit
from mylib.calcium.reverse import ReverseMazeLapSplit

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

def calc_SI(spikes, rate_map, t_total, t_nodes_frac):
    mean_rate = np.nansum(spikes, axis = 1) / t_total # mean firing rate
    logArg = (rate_map.T / mean_rate).T;
    logArg[np.where(logArg == 0)] = 1; # keep argument in log non-zero

    IC = np.nansum(t_nodes_frac * rate_map * np.log2(logArg), axis = 1) # information content
    SI = IC / mean_rate; # spatial information (bits/spike)
    return(SI)

def calc_ratemap(Spikes = None, spike_nodes = None, _nbins = 48*48, occu_time = None, Ms = None, is_silent = None):
    if is_silent is None:
        print("    ERROR! no silent cell information.")
        return 
    
    Spikes = Spikes
    occu_time = occu_time
    spike_count = np.zeros((Spikes.shape[0], _nbins), dtype = np.float64)
    for i in range(_nbins):
        idx = np.where(spike_nodes == i+1)[0]
        spike_count[:,i] = np.nansum(Spikes[:,idx], axis = 1)

    rate_map_all = spike_count/(occu_time/1000+ 1E-9)
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
    
    _, _, smooth_map_rand, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms, 
                                                                                    is_silent = silent_cell)
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

    _, _, smooth_map_rand, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms, 
                                                                                    is_silent = silent_cell)
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
        
    _, _, smooth_map_rand, _ = calc_ratemap(Spikes = spikes_rand, spike_nodes = spike_nodes, occu_time = occu_time, Ms = Ms, 
                                                                                    is_silent = silent_cell)
    SI_rand = calc_SI(spikes = spikes_rand, rate_map = smooth_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, percent)
    return is_placecell

def shuffle_test(trace, Ms = None, shuffle_n = 1000, percent = 95):
    n_neuron = trace['n_neuron']
    SI_all = np.zeros(n_neuron, dtype = np.float64)
    is_placecell_isi = np.zeros(n_neuron, dtype = np.int64)
    is_placecell_shift = np.zeros(n_neuron, dtype = np.int64)
    is_placecell_all = np.zeros(n_neuron, dtype = np.int64)

    SI_all = calc_SI(trace['Spikes'], rate_map = trace['smooth_map_all'], t_total = trace['t_total'], t_nodes_frac = trace['t_nodes_frac'])

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
    venn3(subsets = subset, set_labels = ('Inter Spike Intervle\nShuffle', 'Shift Spikes\nShuffle', 'Pure Random\nShuffle'))
    mkdir(os.path.join(trace['p'], 'Shuffle_Venn'))
    plt.savefig(os.path.join(trace['p'], 'Shuffle_Venn', 'Shuffle_Venn.png'), dpi = 600)
    plt.savefig(os.path.join(trace['p'], 'Shuffle_Venn', 'Shuffle_Venn.svg'), dpi = 600)
    print("    Done.")
    return trace


# ============================================== Draw Figures ====================================================================

# To judge that does a cell's field(s) (main field and, if existed, subfield(s)) locate at both correct path and incorrect path
def FieldOnBothPath(place_field = {}, maze_type = 1, correct_path = None):
    if correct_path is None:
        print("    Error! arg 'correct_path' is required, or it'll raise up an calculating error! Reported by funciton FieldOnBothPath.")
        return 0

    count = np.array([0,0], dtype = np.int64)
    for k in place_field.keys():
        if k in correct_path:
            count[0] = 1
        else:
            count[1] = 1

    # if sum(ocunt) = 2, there must be at least 2 fields locate at correct and incorrect path, respectively.
    if np.sum(count) == 2:
        return 1
    else:
        return 0
        

# According to smoothed rate map, we divide cell populations into two groups: Main Field on Correct Path Cells and Main Field on Incorrect Path cells.
def MainFieldOnWhichPath(trace = None,):
    # Initiate
    maze_type = trace['maze_type']
    rate_map_all = trace['smooth_map_all']
    place_field_all = trace['place_field_all']
    n_neuron = rate_map_all.shape[0]
    silent_idx = trace['SilentNeuron']

    # value = 1 -> main field on correct path, value = 0 -> main field on incorrect path.
    main_field_all = np.zeros(n_neuron, dtype = np.int64)
    # if there're place field on both path, value -> 1, else value -> 0.
    is_FieldOnBoth = np.zeros(n_neuron, dtype = np.int64)

    # Generate max idx
    max_idx = np.argmax(rate_map_all, axis = 1)
    
    correct_path = correct_paths[maze_type]
    # main_field identified
    for n in range(n_neuron):
        # skip silent index.
        if n in silent_idx:
            continue
        
        # if max_idx[n] in correct path, set main_field_all[n] as 1, else keep it as 0.
        if max_idx[n] in correct_path:
            main_field_all[n] = 1

        is_FieldOnBoth[n] = FieldOnBothPath(place_field = place_field_all[n], maze_type = maze_type, correct_path = correct_path)
        
    trace['is_FieldOnBoth'] = is_FieldOnBoth
    trace['main_field_all'] = main_field_all
    return trace


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
            occu_time = occu_time_transform(trace['occu_time_spf'], nx = 12)
        else:
            occu_time = occu_time_transform(occu_time, nx = 12)

        spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx = 12)
        trace['occu_time_old'] = occu_time
        trace['old_nodes'] = spike_nodes
    elif map_type == 'rate':
        nx =48
        rate_map_all = cp.deepcopy(trace['rate_map_clear'])
        if occu_time is None:
            occu_time = cp.deepcopy(trace['occu_time_spf'])
    else:
        assert False

    Spikes = trace['Spikes']
    maze_type = trace['maze_type']
    silent_idx = trace['SilentNeuron']

    # Divide cell population into 2 groups: Main field on correct path cell and main field on incorrect path cell
    trace = MainFieldOnWhichPath(trace = trace)

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

    if maze_type == 0:
        return trace

    cor_nodes = []
    inc_nodes = [] #np.setdiff1d(all_nodes,cor_nodes)
    for k in range(spike_nodes.shape[0]):
        if spike_nodes[k] in Correct_Graph:
            cor_nodes.append(k)
        else:
            inc_nodes.append(k)
    cor_nodes = np.array(cor_nodes, dtype = np.int64)
    inc_nodes = np.array(inc_nodes, dtype = np.int64)
    #cor_nodes = sum(cor_nodes,[])
    #all_nodes = np.array(range(1,Spikes.shape[1]+1))
    
    peak_rate_onpath = np.zeros(n_neuron, dtype = np.float64)
    mean_rate_onpath = np.zeros(n_neuron, dtype = np.float64)

    main_field_all = trace['main_field_all']
    cor_time = np.nansum(occu_time[Correct_Graph-1])
    inc_time = np.nansum(occu_time[Incorrect_Graph-1])
    for n in range(n_neuron):
        # if n is silent, set it as np.nan to avoid effects.
        if n in silent_idx:
            peak_rate_onpath[n] = np.nan
            mean_rate_onpath[n] = np.nan
            continue

        if main_field_all[n] == 1:
            peak_rate_onpath[n] = np.nanmax(rate_map_all[n,Correct_Graph-1])
            mean_rate_onpath[n] = np.nansum(Spikes[n, cor_nodes]) / cor_time * 1000
        else:
            peak_rate_onpath[n] = np.nanmax(rate_map_all[n,Incorrect_Graph-1])
            mean_rate_onpath[n] = np.nansum(Spikes[n, inc_nodes]) / inc_time * 1000
            
    trace['peak_rate_on_path'] = peak_rate_onpath
    trace['mean_rate_on_path'] = mean_rate_onpath
        
    # old version, Before Nov 20th, 2022.
    '''
    cor_mean_rate = np.zeros(n_neuron, dtype = np.float64)
    inc_mean_rate = np.zeros(n_neuron, dtype = np.float64)
    cor_peak_rate = np.zeros(n_neuron, dtype = np.float64)
    inc_peak_rate = np.zeros(n_neuron, dtype = np.float64)

    cor_mean_rate = np.nansum(Spikes[:,cor_nodes], axis = 1) / np.nansum(occu_time[Correct_Graph-1]) * 1000
    inc_mean_rate = np.nansum(Spikes[:,inc_nodes], axis = 1) / np.nansum(occu_time[Incorrect_Graph-1]) * 1000
    cor_peak_rate = np.nanmax(rate_map_all[:,Correct_Graph-1], axis = 1)
    inc_peak_rate = np.nanmax(rate_map_all[:,Incorrect_Graph-1], axis = 1)
    # print(round(peak_rate[k],4),"  ",round(mean_rate[k],4),"  ",round(cor_mean_rate[k],4),"  ",round(inc_mean_rate[k],4))
    
    
    trace['cor_mean_rate'] = cor_mean_rate
    trace['inc_mean_rate'] = inc_mean_rate
    trace['cor_peak_rate'] = cor_peak_rate
    trace['inc_peak_rate'] = inc_peak_rate
    trace['cor_nodes'] = cor_nodes
    trace['inc_nodes'] = inc_nodes
    '''
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

def DrawRateMap(trace, ax = None):
    n_neuron = len(trace['rate_map_all'])
    smooth_map_all = trace['smooth_map_all']
    loc = trace['loc']
    
    for k in tqdm(range(n_neuron)):
        # rate map     
        im = ax.imshow(np.reshape(smooth_map_all[k],[48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_ticks(ColorBarsTicks(peak_rate=np.nanmax(smooth_map_all[k]), is_auto=True, tick_number=4))
        cbar.set_label('Firing Rate / Hz')
        # maze profile
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title(f"Mice {trace['MiceID']}, Cell {k+1}, SI = {round(trace['SI_all'][k],3)}",
                     color = color, fontsize = 12)
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        cbar.remove()
        im.remove()

def RateMap(trace):
    maze_type = trace['maze_type']

    fig = plt.figure(figsize = (4,3))
    ax = Clear_Axes(plt.axes())

    DrawMazeProfile(maze_type = maze_type, axes = ax, nx = 48)
    
    p = trace['p']
    loc = os.path.join(p,'RateMap')

    trace['loc'] = loc
    mkdir(loc)
    DrawRateMap(trace, ax = ax)
    plt.close()

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

# trace map
def DrawTraceMap(trace, ax = None):
    n_neuron = trace['n_neuron']
    maze_type = trace['maze_type']
    processed_pos, behav_time = Add_NAN(trace['correct_pos'], trace['correct_time'], maze_type = maze_type)
    processed_pos = position_transform(processed_pos = processed_pos, nx = 48, maxWidth = 960)
    ms_time_behav = trace['ms_time_behav']
    Spikes = trace['Spikes']
    loc = trace['loc']
    
    a = ax.plot(processed_pos[:,0], processed_pos[:,1], color = 'gray')
    DrawMazeProfile(maze_type = maze_type, axes = ax, color = 'brown', nx = 48, linewidth = 2)
    
    for k in tqdm(range(n_neuron)):
        # spike location
        spike_idx = np.where(Spikes[k,:] == 1)[0]
        Time_idx = np.zeros_like(spike_idx)
        for j in range(len(spike_idx)):
            time = np.where(behav_time <= ms_time_behav[spike_idx[j]])[0]
            if len(time) == 0:
                Time_idx[j] = np.where(behav_time >= ms_time_behav[spike_idx[j]])[0][0]
            else:
                Time_idx[j] = time[-1]
            
        b = ax.plot(processed_pos[Time_idx, 0], 
                    processed_pos[Time_idx, 1], 'o',
                            color = 'black', markersize = 2)
        color = 'red' if trace['is_placecell'][k] == 1 else 'black'
        ax.set_title("Mice "+str(trace['MiceID'])+" , Cell "+str(k+1),color = color,fontsize = 16)
        plt.savefig(os.path.join(loc,str(k+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(loc,str(k+1)+'.png'),dpi = 600)
        for s in range(len(b)):
            b[s].remove()

    plt.close()
    plt.clf()

def TraceMap(trace):
    p = trace['p']
    maze_type = trace['maze_type']
    loc = os.path.join(p,'TraceMap')
    mkdir(loc)
    trace['loc'] = loc

    fig = plt.figure(figsize = (4,4))
    ax = Clear_Axes(plt.axes())
    ax.invert_yaxis()

    DrawTraceMap(trace, ax = ax)
    #ax.yaxis_inverted()
    plt.close()


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


def LocTimeCurve(trace, curve_type = 'Deconv', threshold = 3, isDraw = True):
    # curve_type: str, determines the resource of spikes
    # threshold: float, std * threshold to determine the spikes.
    
    ms_time = cp.deepcopy(trace['ms_time_behav'])
    ms_time_original = cp.deepcopy(trace['ms_time'])
    # y axis value of trajactories (time: ms)
    behav_time = cp.deepcopy(trace['correct_time'])
    correct_nodes = cp.deepcopy(trace['correct_nodes'])
    old_nodes_behav = spike_nodes_transform(spike_nodes = correct_nodes, nx = 12)

    # reorder old_nodes with the order of correct path and incorrect path
    if trace['maze_type'] == 0:
        return

    length = len(correct_paths[trace['maze_type']])
    order = xorders[trace['maze_type']]

    reorder = np.zeros_like(order)
    for i in range(reorder.shape[0]):
        reorder[i] = np.where(order == i+1)[0] + 1
    # reorder old_nodes
    old_nodes_order = reorder[old_nodes_behav-1]

    if curve_type == 'Deconv':
        transient = 'DeconvSignal'
    elif curve_type == 'Raw':
        transient == 'RawTraces'
    else:
        print("    ValueError! Only 'Deconv' and 'Raw' are valid value for curve_type! Report by LocTimeCurve()")
        return
    
    # Generate Spikes and delete spikes
    Spikes = SpikeType(Transients = trace[transient], threshold = threshold) # Calculating spikes in different ways
    idx = np.zeros(ms_time.shape[0], dtype = np.int64)
    for i in range(ms_time.shape[0]):
        idx[i] = np.where(ms_time_original == ms_time[i])[0][0]
    Spikes = Spikes[:,idx]

    # x axis value of trajactories (location: spatial bins)
    rand_nodes = np.random.rand(old_nodes_behav.shape[0]) - 0.5 + old_nodes_order
    
    # find a spike time stamp:
    spike_time = np.zeros_like(ms_time)
    spike_location = np.zeros_like(ms_time)
    for i in range(spike_time.shape[0]):
        # project ms_time onto behav_time
        idx_left = np.where(behav_time <= ms_time[i])[0][-1]
        idx_right = np.where(behav_time >= ms_time[i])[0][0]
        idx = idx_left if ms_time[i] - behav_time[idx_left] <= behav_time[idx_right] - ms_time[i] else idx_right
        # x axis value of spikes (location: bin)
        spike_location[i] = rand_nodes[idx]  
        # y axis value of spikes (time: ms)
        spike_time[i] = behav_time[idx]  
    
    if isDraw == True:
        trace['loc'] = os.path.join(trace['p'],'LocTimeCurve',curve_type+'_'+str(threshold))
        mkdir(trace['loc'])
        DrawLocTimeCurve(behav_time = behav_time, rand_nodes = rand_nodes, spike_time = spike_time, spike_location = spike_location, 
                        Spikes = Spikes, length = length, trace = trace, curve_type = curve_type, threshold = threshold)


def DrawLocTimeCurve(behav_time = None, rand_nodes = None, spike_time = None, spike_location = None, Spikes =None, 
                     length = None, trace = None, curve_type = 'Raw', threshold = 3):
    # Plotting gray trajactory shadow
    fig = plt.figure(figsize = (3,4))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks=True, ifyticks=True)
    ax.plot(rand_nodes,behav_time/1000,'.', color = 'gray', markersize = 1)
    ax.axis([0,145,0,np.nanmax(behav_time/1000)+10])
    ax.set_xticks(np.linspace(0,144,9))
    ax.set_xlabel('Maze ID (Linearized)')
    ax.set_ylabel('Time / s')
    ax.axvline(length+0.5,color = 'orange')
    
    for n in tqdm(range(trace['n_neuron'])):
        # Generate Spikes x,y location
        idx = np.where(Spikes[n,:] == 1)[0]
        x = spike_location[idx]
        y = spike_time[idx] / 1000

        # plotting Spikes
        if 'is_placecell' not in trace.keys():
            color = 'black'
        else:
            color = 'red' if trace['is_placecell'][n] == 1 else 'black'
        a = ax.plot(x,y,'|',color = 'red', markersize = 3, markeredgewidth = 0.3)
        ax.set_title(curve_type+', threshold = '+str(threshold)+'\nCell '+str(n+1), color = color)
        plt.savefig(os.path.join(trace['loc'],'Cell '+str(n+1)+'.svg'),dpi = 600)
        plt.savefig(os.path.join(trace['loc'],'Cell '+str(n+1)+'.png'),dpi = 600)
        for j in a:
            j.remove()

    plt.clf()
    plt.close()

# =============================================== Lap Split Analysis (For Cross Maze, Reveres Maze, DSP Maze (Different Start Point)) =================================
def LapSplit(trace, behavior_paradigm = 'CrossMaze', **kwargs):
    if behavior_paradigm == 'CrossMaze':
        return CrossMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm == 'ReverseMaze':
        return ReverseMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm == 'DSPMaze':
        return DSPMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm == 'SimpleMaze':
        return np.array([0], dtype = np.int64), np.array([trace['correct_time'].shape[0]-1], dtype = np.int64)

def get_check_area(maze_type: int, start_point: int, check_length: int = 5):
    if start_point > 144 or start_point < 1:
        raise ValueError("Only values belong to set {1, 2, ..., 144} are valid! But "+f"{start_point} was inputt.")

    area = [start_point]
    graph = maze1_graph if maze_type == 1 else maze2_graph
    surr = graph[start_point]
    prev = 1

    StepExpand = {1: [start_point]}
    while prev <= check_length:
        StepExpand[prev+1] = []
        for j in StepExpand[prev]:
            for k in graph[j]:
                if k not in area:
                    area.append(k)
                    StepExpand[prev+1].append(k)
        prev += 1

    return area

# This function has been proved to be suited for all session for cross maze paradigm.
def CrossMazeLapSplit(trace, check_length = 5, mid_length = 5):
    if trace['maze_type'] in [1,2,3]:
        beg_idx = []
        end_idx = []
        if len(np.where(np.isnan(trace['correct_nodes']))[0]) != 0:
            print('Error! correct_nodes contains NAN value!')
            return [],[]

        # Define start area, end area and check area with certain sizes that are defaultly set as 5(check_length).
        co_path = correct_paths[trace['maze_type']]
        start_area = get_check_area(maze_type=trace['maze_type'], start_point=1, check_length=check_length)
        end_area = get_check_area(maze_type=trace['maze_type'], start_point=144, check_length=check_length)
        mid = co_path[int(len(co_path) * 0.5)]
        check_area = get_check_area(maze_type=trace['maze_type'], start_point=mid, check_length=mid_length)

        behav_nodes = cp.deepcopy(trace['correct_nodes'])
        behav_nodes = spike_nodes_transform(spike_nodes = behav_nodes, nx = 12)
        switch = 0
        check = 0

        # Check if lap-start or lap-end point frame by frame
        for k in range(behav_nodes.shape[0]):
            # Define checking properties for check area. If recorded point change from end area to start area without passing by the check area, we identify it 
            # as the start of a new lap
            
            if behav_nodes[k] in check_area:
                # Enter check area from end area.
                if switch == 2:
                    check = 0

                # Enter check area from start area.
                if switch == 1: 
                    check = 1

                # Abnormal case: that mice does not occur at start area before they enter the check area.
                if switch == 0:
                    check = 1
                    switch = 1 # Assume that switch state must be belong to {1,2}
                    beg_idx.append(0)

            if behav_nodes[k] in start_area:
                # if switch = 0
                if switch == 0:
                    beg_idx.append(k)
                    switch = 1
                if switch == 2 and check == 1:
                    end_idx.append(k-1)
                    beg_idx.append(k)
                    switch = 1

            if behav_nodes[k] in end_area:
                switch = 2   # state 2: at end area
                check = 1    # check = 1 represents mice have passed the check area

        if switch == 2:
            end_idx.append(behav_nodes.shape[0]-1)
            
        if len(beg_idx) != len(end_idx):
            # Abort when swith = 1
            beg_idx.pop()
        return np.array(beg_idx, dtype = np.int64), np.array(end_idx, dtype = np.int64)

    elif trace['maze_type'] == 0:
        behav_nodes = trace['correct_nodes']
        unit = int(behav_nodes.shape[0] / 2)
        beg_idx = [0, unit]
        end_idx = [unit-1, behav_nodes.shape[0]-1]
        return np.array(beg_idx, dtype = np.int64), np.array(end_idx, dtype = np.int64)
    else:
        print("    Error in maze_type! Report by mylib.maze_utils3.CrossMazeLapSplit")
        return np.array([], dtype = np.int64), np.array([], dtype = np.int64)



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
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 2, _range = 7, nx = 48)

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


def plot_spike_monitor(a, b, c, save_loc: str or None):
    assert a.shape[0] == b.shape[0] and b.shape[0] == c.shape[0]
    n = a.shape[0]
    mkdir(save_loc)
    
    Data = {'class': np.concatenate([np.repeat('raw', n), np.repeat('speed filted', n), np.repeat('delete interval', n)]),
            'x2': np.concatenate([np.repeat(1, n), np.repeat(2, n), np.repeat(3, n)]),
            'x': np.concatenate([np.repeat(1, n)+np.random.rand(n)*0.8-0.4, np.repeat(2, n)+np.random.rand(n)*0.8-0.4, np.repeat(3, n)+np.random.rand(n)*0.8-0.4]),
            'number of spikes/cell': np.concatenate([a, b, c])}
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'x2', y = 'number of spikes/cell', data = Data, ax = ax)
    sns.scatterplot(x = 'x', y = 'number of spikes/cell', data = Data, hue = 'class', ax = ax, legend=False)
    plt.xticks([1,2,3], ['raw', 'speed\nfilted', 'delete\ninterval'])
    ax.set_xlabel("")
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_num.png'), dpi = 600)
    plt.savefig(os.path.join(save_loc, 'spikes_monitor_num.svg'), dpi = 600)
    plt.close()

    Data = {'class': np.concatenate([np.repeat('raw', n), np.repeat('speed filted', n), np.repeat('delete interval', n)]),
            'x2': np.concatenate([np.repeat(1, n), np.repeat(2, n), np.repeat(3, n)]),
            'x': np.concatenate([np.repeat(1, n)+np.random.rand(n)*0.8-0.4, np.repeat(2, n)+np.random.rand(n)*0.8-0.4, np.repeat(3, n)+np.random.rand(n)*0.8-0.4]),
            'spikes remain rate / %': np.concatenate([a/a*100, b/a*100, c/a*100])}
    fig = plt.figure(figsize = (4, 3))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(x = 'x2', y = 'spikes remain rate / %', data = Data, ax = ax)
    sns.scatterplot(x = 'x', y = 'spikes remain rate / %', data = Data, hue = 'class', ax = ax, legend=False)
    plt.xticks([1,2,3], ['raw', 'speed\nfilted', 'delete\ninterval'])
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_xlabel("")
    ax.axis([0, 4, 0, 100])
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

def plot_split_trajactory(trace, behavior_paradigm = 'CrossMaze', split_args: dict = {}, **kwargs):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm, **split_args)
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
    return trace

def half_half_correlation(trace):
    laps = trace['laps']

    if laps == 1:
        return trace 
    
    mid = int((laps+1)/2) if laps % 2 == 1 else int(laps/2)
    ms_idx = trace['ms_idx_lap']

    occu_time_all = cp.deepcopy(trace['occu_time_split'])
    Spikes = cp.deepcopy(trace['Spikes'])
    spike_nodes = cp.deepcopy(trace['spike_nodes'])
    fir_idx = np.arange(0, mid)
    sec_idx = np.arange(mid, laps)
    occu_time_fir = np.nansum(occu_time_all[fir_idx,:], axis = 0)
    occu_time_sec = np.nansum(occu_time_all[sec_idx,:], axis = 0)

    Spikes_fir = np.concatenate([Spikes[:,ms_idx[k]] for k in fir_idx], axis = 1)
    Spikes_sec = np.concatenate([Spikes[:,ms_idx[k]] for k in sec_idx], axis = 1)

    spike_nodes_fir = np.concatenate([spike_nodes[ms_idx[k]] for k in fir_idx])
    spike_nodes_sec = np.concatenate([spike_nodes[ms_idx[k]] for k in sec_idx])

    rate_map_fir, clear_map_fir, smooth_map_fir, nanPos_fir = calc_ratemap(Spikes = Spikes_fir, spike_nodes = spike_nodes_fir, occu_time = occu_time_fir, 
                                                                           Ms = trace['Ms'], is_silent = trace['SilentNeuron'])
    rate_map_sec, clear_map_sec, smooth_map_sec, nanPos_sec = calc_ratemap(Spikes = Spikes_sec, spike_nodes = spike_nodes_sec, occu_time = occu_time_sec, 
                                                                           Ms = trace['Ms'], is_silent = trace['SilentNeuron'])
    fir_sec_corr = np.zeros(Spikes.shape[0], dtype=np.float64)
    for i in range(Spikes.shape[0]):
        fir_sec_corr[i], _ = scipy.stats.pearsonr(smooth_map_fir[i, :], smooth_map_sec[i, :])

    t_total_fir = np.nansum(occu_time_fir) / 1000
    t_total_sec = np.nansum(occu_time_sec) / 1000
    t_nodes_frac_fir = occu_time_fir / 1000 / (t_total_fir + 1E-6)
    t_nodes_frac_sec = occu_time_sec / 1000 / (t_total_sec + 1E-6)
    SI_fir = calc_SI(Spikes_fir, rate_map=smooth_map_fir, t_total=t_total_fir, t_nodes_frac=t_nodes_frac_fir)
    SI_sec = calc_SI(Spikes_sec, rate_map=smooth_map_sec, t_total=t_total_sec, t_nodes_frac=t_nodes_frac_sec)

    appendix = {'rate_map_fir':rate_map_fir, 'clear_map_fir':clear_map_fir, 'smooth_map_fir':smooth_map_fir, 'nanPos_fir':nanPos_fir, 'occu_time_fir':occu_time_fir,
                'rate_map_sec':rate_map_sec, 'clear_map_sec':clear_map_sec, 'smooth_map_sec':smooth_map_sec, 'nanPos_sec':nanPos_sec, 'occu_time_sec':occu_time_sec,
                't_total_fir': t_total_fir, 't_total_sec': t_total_sec, 't_nodes_frac_fir': t_nodes_frac_fir, 't_nodes_frac_sec':t_nodes_frac_sec, 'SI_fir': SI_fir,
                'SI_sec': SI_sec, 'fir_sec_corr': fir_sec_corr}
    trace.update(appendix)
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

    ms_speed = np.zeros_like(ms_time)
    
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

    # Aligned wireless miniscope data with behavioral data. Only for wireless miniscope
    if behavior_paradigm == 'ReverseMaze':
        # need Aligned Information
        align_frame = 679
        ms_time += align_frame * 33

    plot_split_trajactory(trace, behavior_paradigm = behavior_paradigm)

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

    # Generate place field
    place_field_all = place_field(n_neuron = n_neuron, smooth_map_all = smooth_map_all, maze_type = trace['maze_type'])
    
    
    print("    D. Shuffle test for spatial information of each cells to identified place cells. Shuffle method including 1) inter spike intervals(isi), 2) rigid spike shifts, 3) purely random rearrangement of spikes.")
    # total occupation time
    t_total = np.nansum(occu_time)/1000
    # time fraction at each spatial bin
    t_nodes_frac = occu_time / 1000 / (t_total+ 1E-6)

    # Save all variables in a dict
    trace_ms = {'Spikes_original':Spikes_original, 'spike_nodes_original':spike_nodes_original, 'ms_speed_original': ms_speed, 'RawTraces':RawTraces,'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'Spikes':Spikes, 'spike_nodes':spike_nodes, 'ms_time_behav':ms_time_behav, 'ms_speed_behav':ms_speed_behav, 'n_neuron':n_neuron, 
                't_total':t_total, 'dt': dt, 't_nodes_frac':t_nodes_frac, 'SilentNeuron':SilentNeuron, 'rate_map_all':rate_map_all, 'rate_map_clear':rate_map_clear, 
                'smooth_map_all':smooth_map_all, 'nanPos':nanPos, 'Ms':Ms, 'place_field_all':place_field_all, 'ms_folder':folder, 'occu_time_spf': occu_time, 'speed_filter_results': spf_results}
    trace.update(trace_ms)

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
        LocTimeCurve(trace, curve_type = 'Deconv', threshold = 3) 
        print("    Analysis:")
        print("      A. Calculate Population Vector Correlation")
        #population vector correaltion
        trace = PVCorrelationMap(trace)

    
    # Firing Rate Processing:
    print("      B. Firing rate Analysis")
    trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = 30)
    
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
    import time
    import pandas as pd
    import os
    import warnings
    import h5py
    import pickle
    from scipy.io import loadmat

    i = 23
    f = pd.read_excel(r"G:\YSY\Reverse_maze\Reverse_maze_paradigm.xlsx", sheet_name='calcium')
    work_flow = r'G:\YSY\Reverse_maze'

    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])
    session = int(f['session'][i])

    totalpath = work_flow
    p = os.path.join(totalpath, str(MiceID), str(date),"session "+str(session))

    if os.path.exists(os.path.join(p,'trace_behav.pkl')):
        with open(os.path.join(p, 'trace_behav.pkl'), 'rb') as handle:
            trace = pickle.load(handle)
    else:
        warnings.warn(f"{os.path.join(p,'trace_behav.pkl')} is not exist!")
    
    trace['p'] = p    
    f.loc[i, 'Path'] = p
    #coverage = coverage_curve(trace['processed_pos_new'], maze_type=trace['maze_type'], save_loc=os.path.join(p, 'behav'))
    #trace['coverage'] = coverage

    # Read File
    print("    A. Read ms.mat File")
    ms_path = os.path.join(folder, 'ms.mat')
    if os.path.exists(ms_path) == False:
        warnings.warn(f"{ms_path} is not exist!")

    if behavior_paradigm == ['ReverseMaze', 'CrossMaze']:
        ms_mat = loadmat(ms_path)
        ms = ms_mat['ms']
        #FiltTraces = np.array(ms['FiltTraces'][0][0]).T
        RawTraces = np.array(ms['RawTraces'][0][0]).T
        DeconvSignal = np.array(ms['DeconvSignals'][0][0]).T
        ms_time = np.array(ms['time'][0])[0,].T[0]
        
    if behavior_paradigm in ['DSPMaze']:
        with h5py.File(ms_path, 'r') as f:
            ms_mat = f['ms']
            FiltTraces = np.array(ms_mat['FiltTraces'])
            RawTraces = np.array(ms_mat['RawTraces'])
            DeconvSignal = np.array(ms_mat['DeconvSignals'])
            ms_time = np.array(ms_mat['time'],dtype = np.int64)[0,]

    plot_split_trajactory(trace, behavior_paradigm = behavior_paradigm, split_args={})