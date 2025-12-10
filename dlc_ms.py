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
import gc
from mylib.preprocessing_ms import plot_split_trajectory, Generate_SilentNeuron, calc_ratemap, place_field, Delete_InterLapSpike
from mylib.preprocessing_ms import shuffle_test, RateMap, QuarterMap, OldMap, SimplePeakCurve, TraceMap, LocTimeCurve, PVCorrelationMap
from mylib.preprocessing_ms import CrossLapsCorrelation, OldMapSplit, FiringRateProcess, calc_ms_speed, plot_spike_monitor
from mylib.preprocessing_ms import half_half_correlation, odd_even_correlation, coverage_curve, CombineMap, plot_field_arange, field_register
from mylib.preprocessing_ms import calc_speed, uniform_smooth_speed, field_specific_correlation, Clear_Axes, DrawMazeProfile, add_perfect_lap
from mylib.preprocessing_ms import get_spike_frame_label, ComplexFieldAnalyzer, RateMapIncludeIP, TraceMapIncludeIP, count_field_number
from mylib.field.within_field import within_field_half_half_correlation, within_field_odd_even_correlation
from mylib.maze_utils3 import SpikeType, SpikeNodes, SmoothMatrix, mkdir
from mylib.calcium.firing_rate import calc_rate_map_properties
import matplotlib.pyplot as plt

def run_all_mice_DLC(i: int, f: pd.DataFrame, work_flow: str, 
                     v_thre: float = 2.5, cam_degree = 0, speed_sm_args = {}):#p = None, folder = None, behavior_paradigm = 'CrossMaze'):
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
    familiarity = f['Familiarity'][i]
    trace = add_perfect_lap(trace)

    # Read File
    print("    A. Read ms.mat File")
    ms_path = os.path.join(folder, 'ms.mat')
    if os.path.exists(ms_path) == False:
        warnings.warn(f"{ms_path} is not exist!")
        return

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

    plot_split_trajectory(trace, behavior_paradigm = behavior_paradigm, split_args={})

    print("    B. Calculate putative spikes (events) and their related position from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 3)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # Filter the speed
    if 'smooth_speed' not in trace.keys():
        behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
        trace['smooth_speed'] = uniform_smooth_speed(behav_speed, **speed_sm_args)
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
    print("      - Delete the correct track spikes.")
    Spikes, spike_nodes, ms_time_behav, ms_speed_behav, dt = Delete_InterLapSpike(behav_time = trace['correct_time'], ms_time = ms_time_behav, 
                                                                              Spikes = Spikes, spike_nodes = spike_nodes, dt = dt, ms_speed=ms_speed_behav,
                                                                              behavior_paradigm = behavior_paradigm, trace = trace)

    n_neuron = Spikes.shape[0]
    spike_num_mon4 = np.nansum(Spikes, axis = 1)

    plot_spike_monitor(spike_num_mon1, spike_num_mon2, spike_num_mon3, spike_num_mon4, save_loc = os.path.join(trace['p'], 'behav'))

    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 1, _range = 7, nx = 48)
    
    if maze_type != 0:
        print("      - Delete the inter-lap spikes.")
        frame_labels = get_spike_frame_label(
            ms_time=ms_time_behav, 
            spike_nodes=spike_nodes,
            trace=trace, 
            behavior_paradigm=behavior_paradigm
        )
        assert frame_labels.shape[0] == spike_nodes.shape[0]

        # cis direction
        idx = np.where(frame_labels == 1)[0]
        
        if maze_type != 0:
            print("Consider all place fields either on correct track or incorrect track.")
            additional_trace =  calc_rate_map_properties(
                trace['maze_type'],
                ms_time_behav,
                Spikes,
                spike_nodes,
                ms_speed_behav,
                dt,
                Ms,
                trace['p'],
                behavior_paradigm = behavior_paradigm,
                kwargs = {'file_name': 'Place cell shuffle [cis]'}
            )
            trace['LA'] = additional_trace
            print("    Draw rate map that includes activities on incorrect path.")
            #RateMapIncludeIP(trace)
            print("    Draw trace map that includes activities on incorrect path.")
            #TraceMapIncludeIP(trace)
    else:
        idx = np.arange(ms_time_behav.shape[0])
        
    trace_ms = calc_rate_map_properties(
        trace['maze_type'],
        ms_time_behav[idx],
        Spikes[:, idx],
        spike_nodes[idx],
        ms_speed_behav[idx],
        dt[idx],
        Ms,
        trace['p'],
        behavior_paradigm = behavior_paradigm,
        kwargs = {'file_name': 'Place cell shuffle [cis]'}
    )

    trace_ms2 = {'Spikes_original':Spikes_original, 'spike_nodes_original':spike_nodes_original, 'ms_speed_original': ms_speed, 'RawTraces':RawTraces,'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'Ms':Ms, 'ms_folder':folder, 'speed_filter_results': spf_results, 'familiarity': familiarity}
    trace.update(trace_ms2)
    trace.update(trace_ms)

    if trace['maze_type'] != 0:
        print("      7. LocTimeCurve:")
    #    LocTimeCurve(trace) 

    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    
    print("    Plotting:")
    print("      1. Oldmap")
    trace = OldMap(trace, isDraw=False)
    # Generate place field
    print("      2. Complex Field Analyzer")
    #if maze_type != 0:
        #trace = ComplexFieldAnalyzer(trace)

    print("      3. Ratemap")
    #RateMap(trace)
    
    print("      4. Tracemap")
    #TraceMap(trace)      
      
    print("      5. Quarter_map")
    trace = QuarterMap(trace, isDraw = False)
    
    #if maze_type != 0:
        #LocTimeCurve(trace) 
    #trace = OldMap(trace, isDraw=False)
    
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("      5. PeakCurve")
    mkdir(os.path.join(trace['p'], 'PeakCurve'))
    #SimplePeakCurve(trace, file_name = 'PeakCurve', save_loc = os.path.join(trace['p'], 'PeakCurve'))
    
    print("      6. Combining tracemap, rate map(48 x 48), old map(12 x 12) and quarter map(24 x 24)")
    #CombineMap(trace)
    
    if trace['maze_type'] != 0:
        # LocTimeCurve
        print("      7. LocTimeCurve:")
        #LocTimeCurve(trace) 
        print("    Analysis:")
        print("      A. Calculate Population Vector Correlation")
        #population vector correaltion
        trace = PVCorrelationMap(trace)
    # Firing Rate Processing:
    print("      B. Firing rate Analysis")
    trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = 10)
    # Cross-Lap Analysis

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
    
    trace['FSCList'] = within_field_half_half_correlation(
        trace['smooth_map_fir'],
        trace['smooth_map_sec'],
        trace['place_field_all']
    )
    
    trace['OECList'] = within_field_odd_even_correlation(
        trace['smooth_map_odd'],
        trace['smooth_map_evn'],
        trace['place_field_all']
    )
    
    trace = field_register(trace)

    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")
    
    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)
    
    del trace
    gc.collect()
    
    
if __name__ == '__main__':
    
    from mylib.local_path import f1
    work_flow = r"D:\Data\Cross_maze"
    run_all_mice_DLC(728, f=f1, work_flow=work_flow)