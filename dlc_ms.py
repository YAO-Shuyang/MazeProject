import os
import pickle
from scipy.io import loadmat
import numpy as np
import time
import h5py
import copy as cp
import warnings
import pandas as pd
import scipy.stats
from mylib.preprocessing_ms import plot_split_trajactory, Delete_InterLapSpike, Generate_SilentNeuron, calc_ratemap, place_field
from mylib.preprocessing_ms import shuffle_test, RateMap, QuarterMap, OldMap, SimplePeakCurve, TraceMap, LocTimeCurve, PVCorrelationMap
from mylib.preprocessing_ms import CrossLapsCorrelation, OldMapSplit, FiringRateProcess, calc_ms_speed, plot_spike_monitor
from mylib.preprocessing_ms import half_half_correlation, odd_even_correlation, coverage_curve, CombineMap, plot_field_arange
from mylib.maze_utils3 import SpikeType, SpikeNodes, SmoothMatrix, mkdir

def run_all_mice_DLC(i: int, f: pd.DataFrame, work_flow: str, 
                     v_thre: float = 2.5, cam_degree = 0):#p = None, folder = None, behavior_paradigm = 'CrossMaze'):
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
        return
    
    trace['p'] = p    
    f.loc[i, 'Path'] = p
    coverage = coverage_curve(trace['processed_pos_new'], maze_type=trace['maze_type'], save_loc=os.path.join(p, 'behav'))
    trace['coverage'] = coverage

    # Read File
    print("    A. Read ms.mat File")
    ms_path = os.path.join(folder, 'ms.mat')
    if os.path.exists(ms_path) == False:
        warnings.warn(f"{ms_path} is not exist!")
        return

    if behavior_paradigm == 'CrossMaze':
        ms_mat = loadmat(ms_path)
        ms = ms_mat['ms']
        #FiltTraces = np.array(ms['FiltTraces'][0][0]).T
        RawTraces = np.array(ms['RawTraces'][0][0]).T
        DeconvSignal = np.array(ms['DeconvSignals'][0][0]).T
        ms_time = np.array(ms['time'][0])[0,].T[0]
    if behavior_paradigm in ['ReverseMaze','DSPMaze']:
        with h5py.File(ms_path, 'r') as f:
            ms_mat = f['ms']
            FiltTraces = np.array(ms_mat['FiltTraces'])
            RawTraces = np.array(ms_mat['RawTraces'])
            DeconvSignal = np.array(ms_mat['DeconvSignals'])
            ms_time = np.array(ms_mat['time'],dtype = np.int64)[0,]

    plot_split_trajactory(trace, behavior_paradigm = behavior_paradigm, split_args={})

    print("    B. Calculate putative spikes and correlated location from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 5)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # Filter the speed
    ms_speed = calc_ms_speed(behav_speed=trace['smooth_speed'], behav_time=trace['correct_time'], 
                             ms_time=ms_time)

    # Delete NAN value in spike nodes
    print("      - Delete NAN values in data.")
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    Spikes = cp.deepcopy(Spikes_original[:,idx])
    spike_nodes = cp.deepcopy(spike_nodes_original[idx])
    ms_time_behav = cp.deepcopy(ms_time[idx])
    ms_speed_behav = cp.deepcopy(ms_speed[idx])
    dt = np.append(np.ediff1d(ms_time_behav), 33)
    dt[np.where(dt >= 100)[0]] = 100
    
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
    Ms = SmoothMatrix(maze_type = maze_type, sigma = 2, _range = 7, nx = 48)
    rate_map_all, rate_map_clear, smooth_map_all, nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = spike_nodes, 
                                                                        _nbins = 48*48, occu_time = occu_time, Ms = Ms, is_silent = SilentNeuron)

    # Generate place field
    place_field_all = place_field(n_neuron = n_neuron, smooth_map_all = smooth_map_all, maze_type = maze_type)
    
    
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
    
    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")
    
    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)