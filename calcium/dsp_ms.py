import copy as cp
import os
import pickle
import time
import warnings
from os.path import exists, join

import h5py
import scipy.stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns
from mylib.behavior.behavevents import BehavEvents
from mylib.maze_graph import correct_paths, NRGs, Father2SonGraph, CP_DSPs
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile, clear_NAN, mkdir, SpikeNodes, SpikeType, GetDMatrices
from mylib.maze_utils3 import plot_trajactory, spike_nodes_transform, SmoothMatrix, occu_time_transform
from mylib.preprocessing_ms import coverage_curve, calc_speed_with_smooth, calc_ratemap, place_field_dsp
from mylib.preprocessing_ms import plot_spike_monitor, calc_ms_speed, OldMap
from mylib.preprocessing_ms import calc_SI
from mylib.divide_laps.lap_split import LapSplit
from mylib.calcium.axes.peak_curve import get_y_order
from scipy.io import loadmat
from tqdm import tqdm
from mylib import RateMapAxes, TraceMapAxes, PeakCurveAxes, LocTimeCurveAxes
from mylib.calcium.firing_rate import calc_rate_map_properties
from mylib.dsp.neural_traj import get_neural_trajectory, segmented_neural_trajectory

# 11/20/25, 准备新增两个实验。
# 我们将correct/incorrect imaging videos合在一起重新跑了CNMF-E。从而，代码需要随之修改。
DSPPalette = ["#A9CCE3", "#82C3C5", '#9C8FBC', "#D9A6A9", "#DCC8A4", '#647D91', "#C06C84", "#007a8c",
              "#DF9E26", "#15C4CA", "#4A1CBF"]

def classify_lap(
    behav_nodes: np.ndarray, 
    beg_idx: np.ndarray, 
    maze_type: int
):
    """Classify which route the given laps correspond to.

    Parameters
    ----------
    behav_nodes : np.ndarray
        The position bins of a given route
    beg_idx : np.ndarray, (n_laps,)
        The starting indices of each lap    
    maze_type : int
        The type of maze. 1: Maze A; 2: Maze B; 4: Maze A modified.

    Returns
    -------
    lap_type : np.ndarray (n_laps,)
        The classified route type for each lap.
        
    Examples
    --------
    >>> lap_type = classify_lap(behav_nodes, beg_idx, maze_type=1)
    >>> print(lap_type)
    array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 4, 4, 4, 5, 5, 6, 6, 6, 0, 0])
    
    Raises
    ------
    NotImplementedError
        If the maze type is not supported.
    ValueError
        If a lap's starting node cannot be identified in the classification set.
    """
    if maze_type == 1:
        route_class = {
            0: [1, 2, 13, 14, 25, 26],
            1: [23, 22, 34, 33],
            2: [66, 65, 64, 63],
            3: [99, 87, 88, 76],
            4: [8, 7, 6, 18],
            5: [93, 105, 106, 94],
            6: [135, 134, 133, 121]
        }
    elif maze_type == 2:
        route_class = {
            0: [1, 2, 14, 13, 3, 15],
            1: [7, 19, 18, 30, 31],
            2: [23, 24, 22, 34],
            3: [105, 106, 94, 82],
            4: [113, 112, 124, 125],
            5: [62, 50, 38, 51, 39, 52],
            6: [139, 127, 128, 140]
        }
    elif maze_type == 4:
        route_class = {
            0: [1, 2, 13, 14, 25, 26],
            1: [23, 22, 34, 33],
            2: [66, 65, 64, 63],
            3: [99, 87, 88, 76],
            4: [8, 7, 6, 18],
            5: [93, 105, 106, 94],
            6: [135, 134, 133, 121, 110], # 110 here is to handle an exception from mouse 10266, SB, trial 111.
            7: [138, 126, 127, 115],
            8: [30, 31, 19, 20],
            9: [45, 46, 47, 48],
            10: [62, 50, 51, 52, 39, 40]
        }
    else:
        raise NotImplementedError(
            f"Maze type {maze_type} is not implemented for lap classification!"
        )

    lap_type = np.zeros_like(beg_idx, np.int64) 
    laps = beg_idx.shape[0]
    
    for i in range(laps):
        is_find = False
        for k in route_class.keys():
            if behav_nodes[beg_idx[i]] in route_class[k]:
                lap_type[i] = k
                is_find = True
                break

        if not is_find:
            raise ValueError(
                f"Fail to identify which node the lap starts! "
                f"Lap {i+1}, Node {behav_nodes[beg_idx[i]]}, "
                f"Index {beg_idx[i]}, Classification set {route_class}"
            )

    return lap_type
    
def plot_split_trajectory(trace: dict):
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    behav_pos = cp.deepcopy(trace['correct_pos'])
    behav_time = cp.deepcopy(trace['behav_time'])

    save_loc = os.path.join(trace['p'], 'behav','laps_trajactory')
    mkdir(save_loc)

    laps = beg_idx.shape[0]

    for k in tqdm(range(laps)):
        loc_x, loc_y = behav_pos[beg_idx[k]:end_idx[k]+1, 0] / 20 - 0.5, behav_pos[beg_idx[k]:end_idx[k]+1, 1] / 20 - 0.5
        fig = plt.figure(figsize = (6,6))
        ax = Clear_Axes(plt.axes())
        ax.set_title('Frame: '+str(beg_idx[k])+' -> '+str(end_idx[k])+'\n'+'Time:  '+str(behav_time[beg_idx[k]]/1000)+' -> '+str(behav_time[end_idx[k]]/1000))
        DrawMazeProfile(maze_type=trace['maze_type'], nx = 48, color='black', axes=ax)
        ax.invert_yaxis()
        ax.plot(loc_x, loc_y, 'o', color = 'red', markeredgewidth = 0, markersize = 3)

        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.png'), dpi=600)
        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.svg'), dpi=600)
        plt.close()

def split_calcium_data(
    lap_idx: np.ndarray,
    trace: dict, 
    ms_time: np.ndarray
):
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    beg_idx_ms, end_idx_ms = np.zeros_like(beg_idx), np.zeros_like(end_idx)
    for i in range(beg_idx.shape[0]):
        beg_idx_ms[i] = np.where(ms_time >= trace['correct_time'][beg_idx[i]])[0][0]
        end_idx_ms[i] = np.where(ms_time <= trace['correct_time'][end_idx[i]])[0][-1]
    
    idx = np.concatenate([np.arange(beg_idx_ms[i], end_idx_ms[i]+1) for i in lap_idx])

    return idx 

def get_son_area(area: np.ndarray):
    return np.concatenate([Father2SonGraph[i] for i in area])

def RoutewiseCorrelation(trace: dict):
    maze_type = trace['maze_type']
    CPs = CP_DSPs[maze_type]
    if maze_type != 4:
        n_nodes = 10
    elif maze_type == 4:
        n_nodes = 11 if 'node 15' not in trace.keys() else 16
    corr_mat = np.ones((n_nodes, n_nodes), np.float64)
    
    for i in range(n_nodes - 1):
        for j in range(i+1, n_nodes):
            corr = np.zeros(trace['n_neuron'], np.float64) * np.nan
            idx = np.arange(trace['n_neuron'])
            """
            idx = np.where(
                (trace[f'node {i}']['is_placecell'] == 1) |
                (trace[f'node {j}']['is_placecell'] == 1)
            )[0]
            """
            bins = get_son_area(np.intersect1d(
                CPs[trace[f'node {i}']['Route']],
                CPs[trace[f'node {j}']['Route']],
            )) - 1
            
            for k in idx:
                corr[k] = np.corrcoef(
                    trace['node '+str(i)]['smooth_map_all'][k, bins], 
                    trace['node '+str(j)]['smooth_map_all'][k, bins]
                )[0,1]
            
            corr_mat[i, j] = corr_mat[j, i] = np.nanmean(corr)
    
    trace['route_wise_corr'] = corr_mat
    return trace

def calc_pvc(ratemap1, ratemap2, bins):
    corr = np.zeros(bins.shape[0])
    
    for i in range(bins.shape[0]):
        corr[i] = np.corrcoef(ratemap1[:, bins[i]], ratemap2[:, bins[i]])[0,1]
    
    return np.nanmean(corr)

# New version on 11/20/25
def LocTimeCurve(trace: dict) -> dict:
    mouse = trace['MiceID'],
    date = trace['date']
    maze_type = trace['maze_type']
    cell_list = range(trace['n_neuron'])
    save_loc = join(trace['p'], 'LocTimeCurve')
        
    mkdir(save_loc)
    if 'node 10' not in trace.keys():
        colors, n_nodes_all = ([
            DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
            DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]
        ], 10)
    elif 'node 11' not in trace.keys() and 'node 10' in trace.keys():
        colors, n_nodes_all = ([
            DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[7], DSPPalette[0],
            DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]
        ], 11)
    elif 'node 15' in trace.keys():
        colors, n_nodes_all = ([
            DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[7], DSPPalette[0],
            DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0],
            DSPPalette[0], DSPPalette[8], DSPPalette[9], DSPPalette[10], DSPPalette[0]
        ], 16)
    
    fig = plt.figure(figsize=(3, 4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)      
    linearized_xs = [
        NRGs[maze_type][spike_nodes_transform(trace[f'node {_i}']['spike_nodes'], 12).astype(np.int64) - 1] + 
        np.random.rand(trace[f'node {_i}']['spike_nodes'].shape[0]) - 0.5
        for _i in range(n_nodes_all)
    ]
    
    for cell in tqdm(cell_list):
        clear_items = []
        for i in range(n_nodes_all):
            ax, a, b = LocTimeCurveAxes(
                ax,
                behav_time=trace[f'node {i}']['ms_time_behav'],
                given_x=linearized_xs[i],
                spikes=trace[f'node {i}']['Spikes'][cell, :],
                spike_time=trace[f'node {i}']['ms_time_behav'],
                maze_type=maze_type,
                line_kwargs={'linewidth': 0.4, 'color': colors[i]},
                bar_kwargs={'markeredgewidth': 0.5, 'markersize': 2, 'color': 'k'},
                is_include_incorrect_paths=True
            )
            clear_items = clear_items + a + b
        
        if n_nodes_all in [10, 11]:
            t1 = trace['node 4']['ms_time_behav'][-1]/1000 if n_nodes_all == 10 else trace['node 5']['ms_time_behav'][-1]/1000
            t2 = trace['node 5']['ms_time_behav'][0]/1000 if n_nodes_all == 10 else trace['node 6']['ms_time_behav'][0]/1000
            t3 = trace['node 9']['ms_time_behav'][-1]/1000 if n_nodes_all == 10 else trace['node 10']['ms_time_behav'][-1]/1000
            ax.set_yticks([0, t1, t2, t3], [0, t1, 0, t3-t2])
        else:
            t1 = trace['node 4']['ms_time_behav'][-1]/1000
            t2 = trace['node 5']['ms_time_behav'][0]/1000
            t3 = trace['node 9']['ms_time_behav'][-1]/1000
            t4 = trace['node 10']['ms_time_behav'][0]/1000
            t5 = trace['node 15']['ms_time_behav'][0]/1000
            ax.set_yticks([0, t1, t2, t3, t4, t5], [0, t1, 0, t3-t2, 0, t5-t4])
            
        plt.savefig(os.path.join(save_loc, f'Cell {cell+1}.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, f'Cell {cell+1}.svg'), dpi=600)
        for items in clear_items:
            items.remove()
    
    plt.close()
    return trace

def run_all_mice_DLC(
    i: int, 
    f: pd.DataFrame, 
    work_flow: str, 
    v_thre: float = 0, 
    cam_degree = 0, 
    speed_sm_args = {}
):
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])

    if behavior_paradigm not in ['DSPMaze']:
        raise ValueError(f"This is code for reverse maze and hairpin maze specifically! But {behavior_paradigm} is got.")

    totalpath = work_flow
    p = os.path.join(totalpath, str(MiceID), str(date))

    if os.path.exists(os.path.join(p,'trace_behav.pkl')):
        with open(os.path.join(p, 'trace_behav.pkl'), 'rb') as handle:
            trace = pickle.load(handle)
    else:
        raise FileNotFoundError(f"{os.path.join(p,'trace_behav.pkl')} is not exist!")
    
    trace['p'] = p
    f.loc[i, 'Path'] = p
    coverage = coverage_curve(trace['processed_pos_new'], maze_type=trace['maze_type'], save_loc=os.path.join(p, 'behav'))
    trace['coverage'] = coverage

    # Read File
    print("    A. Read ms.mat File")
    ms_path = os.path.join(folder, 'ms.mat')
    if os.path.exists(ms_path) == False:
        warnings.warn(f"{ms_path} is not exist!")

    try:
        ms_mat = loadmat(ms_path)
        ms = ms_mat['ms']
        #FiltTraces = np.array(ms['FiltTraces'][0][0]).T
        RawTraces = np.array(ms['RawTraces'][0][0]).T
        DeconvSignal = np.array(ms['DeconvSignals'][0][0]).T
        ms_time = np.array(ms['time'][0])[0,].T[0]
    except:
        with h5py.File(ms_path, 'r') as f:
            ms_mat = f['ms']
            FiltTraces = np.array(ms_mat['FiltTraces'])
            RawTraces = np.array(ms_mat['RawTraces'])
            DeconvSignal = np.array(ms_mat['DeconvSignals'])
            ms_time = np.array(ms_mat['time'],dtype = np.int64)[0,]
    
    #plot_split_trajectory(trace)
    beg, end = LapSplit(trace, trace['paradigm']) 
    lap_type = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg, maze_type=maze_type)
    print(lap_type)
    
    print("    B. Calculate putative spikes and correlated location from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 2)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # Filter the speed
    behav_speed = calc_speed_with_smooth(
        behav_positions = trace['correct_pos']/10, 
        behav_time = trace['correct_time'], 
        lap_beg_time=trace['lap beg time'],
        lap_end_time=trace['lap end time'],
        smooth_window=5 # frames
    )
    trace['correct_speed_smoothed'] = behav_speed
    trace['correct_speed_raw'] = calc_speed_with_smooth(
        behav_positions = trace['correct_pos']/10, 
        behav_time = trace['correct_time'], 
        lap_beg_time=trace['lap beg time'],
        lap_end_time=trace['lap end time'],
        smooth_window=1
    )

    ms_speed = calc_ms_speed(
        behav_speed=trace['correct_speed_smoothed'], 
        behav_time=trace['correct_time'], 
        ms_time=ms_time
    )

    # Delete NAN value in spike nodes
    print("      - Delete NAN values in data.")
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    Spikes = cp.deepcopy(Spikes_original[:,idx])
    spike_nodes = cp.deepcopy(spike_nodes_original[idx])
    ms_time_behav = cp.deepcopy(ms_time[idx])
    ms_speed_behav = cp.deepcopy(ms_speed[idx])
    dt = np.append(np.ediff1d(ms_time_behav), 33)
    dt[np.where(dt >= 100)[0]] = 100
    spike_num_mon2 = np.nansum(Spikes, axis = 1)
    
    # Filter the speed
    print(f"     - Filter spikes with speed {v_thre} cm/s.")
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
    spike_num_mon3 = np.nansum(Spikes, axis = 1)

    # Delete InterLap Spikes
    n_neuron = Spikes.shape[0]
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 1, _range = 7, nx = 48)
    print("      - Calculate and shuffle sub-ratemap")
    spike_num_mon4 = np.zeros(n_neuron, dtype=np.int64)
    
    d_type = np.ediff1d(lap_type)
    lap_intervals = trace['correct_time'][beg[1:]] - trace['correct_time'][end[:-1]]
    idx = np.where(
        (d_type != 0) |
        ((d_type == 0) & (lap_type[:-1] == 0) & (lap_intervals > 99000))
    )[0]
    if maze_type != 4:
        if idx.shape[0] != 9:
            warnings.warn(
                f"Breakpoints should be 9 but {idx.shape[0]} were found! "
                f"Automatedly set the second value as the division of route 1b and 1c."
                f"Note that this would potentially result in erroneous division of route categories."
            )
            idx = np.where(d_type != 0)[0] # Between routes
            # Between route 1b and 1c
            idx_0 = np.where((d_type == 0) & (lap_type[:-1] == 0) & (lap_intervals > 99000))[0]
            # There should be 9 breakpoints to separate all the laps into 10 groups.
            # 这里主要是两个异常值的处理，仅限于截止至2024年8月1日的10224/27 2023-10-11 session 1
            # (correct session) 中route 1b存在两个lap之间时间超过了99s，从而使得我们上述的判别
            # route 1b 1c之间的breakpoints无法辨别具体的位置。考虑到这两处异常均位于session 1，我们
            # 直接去除所找到的第一个值，保留第二个值作为1b1c的gap。
            # 因此对于任何新的数据，如果两个lap之间的时间间隔超过了99s，并且位于session 2，此处仍会
            # 报错并错误的分类，需注意！
            idx = np.insert(idx, 4, idx_0[1])
    else:
        if idx.shape[0] not in [10, 15]:
            raise ValueError(
                f"Breakpoints should be 10 or 15 for Maze 1 modified but {idx.shape[0]} were found! "
            )
            
    seg_beg, seg_end = np.concatenate([[0], idx+1]), np.concatenate([idx, [d_type.shape[0]-1]])
    trace['n_neuron'] = n_neuron
    n_nodes_total = idx.shape[0]+1
    for n in range(n_nodes_total):
        print(f"        Node {n} -----------------------------------------------------------")
        lap_idx = np.arange(seg_beg[n], seg_end[n] + 1)
        idx = split_calcium_data(lap_idx=lap_idx, trace=trace, ms_time=ms_time_behav)
        spike_num_mon4 += np.nansum(Spikes[:, idx], axis = 1)

        trace['node '+str(n)] = calc_rate_map_properties(
            trace['maze_type'],
            ms_time_behav[idx],
            Spikes[:, idx],
            spike_nodes[idx],
            ms_speed_behav[idx],
            dt[idx],
            Ms,
            trace['p'],
            behavior_paradigm = trace['paradigm'],
            kwargs = {'file_name': 'Place cell shuffle [trans]', 'shuffle_n': 1},
            spike_num_thre=5,
            placefield_kwargs={"thre_type": 2, "parameter": 0.2, 'events_num_crit': 5},
            is_shuffle=False,
            is_calc_fields=False
        )
        trace['node '+str(n)]['Route'] = lap_type[seg_beg[n]]

    plot_spike_monitor(spike_num_mon1, spike_num_mon2, spike_num_mon3, spike_num_mon4, save_loc = os.path.join(trace['p'], 'behav'))

    """
    print("Total Place Fields:")
    trace['place_field_all'] = place_field_dsp(
        trace, thre_type=2, parameter=0.4, events_num_crit=10, need_events_num=True, split_thre=0.2, reactivate_num=5
    )
    """
    print("    C. Calculating firing rate for each neuron and identified their place fields (those areas which firing rate >= 50% peak rate)")
    # Set occu_time <= 50ms spatial bins as nan to avoid too big firing rate

    trace_ms = {'Spikes_original':Spikes_original, 'route_labels': lap_type,
                'spike_nodes_original':spike_nodes_original, 
                'ms_time_behav':ms_time_behav, 'ms_speed_behav':ms_speed_behav,
                'Spikes':Spikes, 'spike_nodes':spike_nodes,
                'ms_speed_original': ms_speed, 'RawTraces':RawTraces,
                'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'ms_folder':folder, 
                'speed_filter_results': spf_results, 
                'n_neuron': Spikes_original.shape[0], 'Ms':Ms}

    trace.update(trace_ms)
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)

    print("    Plotting:")
    print("      1. Old Maps")
    for n in range(n_nodes_total):
        trace[f'node {n}'] = OldMap(trace[f'node {n}'], isDraw=False)
        
    print("      2. Neural Trajectory (Deprecated)")
    #trace = get_neural_trajectory(trace)
    #trace = segmented_neural_trajectory(trace)
    
    print("      3. Routewise Correlation")
    trace = RoutewiseCorrelation(trace)
    
    print("      4. Loc-time curve")
    #trace = LocTimeCurve(trace)
    #trace = LocTimeCurve(trace)

    print("      5. Population Vector Correlation (Deprecated)")
    #MazeSegmentsPVCorrelation(trace)

    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")

    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)

if __name__ == '__main__':
    from mylib.local_path import f2
    
    for i in range(35+6, len(f2)):
        if f2['MiceID'][i] == 10209:
            continue

        run_all_mice_DLC(
            i=i,
            f=f2, 
            work_flow=r"D:\Data\Dsp_maze"
        )
    """
    with  open(f2['Trace File'][34], 'rb') as handle:
        trace = pickle.load(handle)
    trace['p'] = join(r"D:\Data\Dsp_maze", str(int(trace['MiceID'])), str(int(trace['date'])))
    LocTimeCurve(trace)
    """