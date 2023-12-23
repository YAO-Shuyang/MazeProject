import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import pandas as pd
import pickle
import gc

from mylib.maze_utils3 import mkdir, DrawMazeProfile, spike_nodes_transform, Clear_Axes, correct_paths, maze_graphs
from mylib.multiday.field_tracker import field_register
from mylib.statistic_test import GetMultidayIndexmap
from mylib.divide_laps.lap_split import LapSplit
from mylib.multiday.core import MultiDayCore
from mylib.local_path import f1, f_CellReg_day
import copy as cp
        
def field_reallocate(field: dict, maze_type: int):
    CP = correct_paths[(int(maze_type), 48)]
    G = maze_graphs[(int(maze_type), 48)]
    
    field_area = cp.deepcopy(CP)
    shuffle_field = {}
    for i, k in enumerate(field.keys()):
        CENTER = np.random.choice(field_area, size = 1)[0]
        LENGTH = len(field[k])
        
        Area = [CENTER]
        step = 0
        StepExpand = {0: [CENTER]}
        while len(Area) < LENGTH:
            StepExpand[step+1] = []
            for c in StepExpand[step]:
                surr = G[c]
                for j in surr:
                    if j in field_area and j not in Area:
                        StepExpand[step+1].append(j)
                        Area.append(j)
        
            # Generate field successfully! 
            if len(StepExpand[step+1]) == 0:
                break
            step += 1
    
        shuffle_field[CENTER] = np.array(Area, dtype=np.int64)
        
        field_area = np.intersect1d(field_area, shuffle_field[CENTER])
    return shuffle_field
            
def match_fields(
    index_map1: np.ndarray,
    index_map2: np.ndarray,
    place_fields_num1: np.ndarray,
    place_fields_num2: np.ndarray,
    is_placecell_x: np.ndarray,
    is_placecell_y: np.ndarray,
    place_fields_x: list,
    place_fields_y: list,
    field_pool_x: list,
    field_pool_y: list,
    maze_type: int,
    overlap_thre: float = 0.6
):
    n_neuron = index_map1.shape[0]
    
    dyn_behav = np.zeros((7, n_neuron))
    #turn_on = np.zeros(n_neuron)
    #turn_off = np.zeros(n_neuron)
    #kept_on = np.zeros(n_neuron)
    #kept_off = np.zeros(n_neuron)
    
    #field_on = np.zeros(n_neuron)
    #field_off = np.zeros(n_neuron)
    #field_kep = np.zeros(n_neuron)
    
    size_std = []
    for i in range(n_neuron):
        if is_placecell_x[i] == 0 and is_placecell_y[i] == 0:
            dyn_behav[3, i] = 1
            continue
        elif is_placecell_x[i] == 0 and is_placecell_y[i] == 1:
            dyn_behav[0, i] = 1
            continue
        elif is_placecell_x[i] == 1 and is_placecell_y[i] == 0:
            dyn_behav[1, i] = 1
            continue
        else:
            dyn_behav[2, i] = 1
        
        for k1 in place_fields_x[i].keys():
            is_matched = False
            field1 = place_fields_x[i][k1]
            for k2 in place_fields_y[i].keys():
                field2 = place_fields_y[i][k2]
                overlap = np.intersect1d(field1, field2)
                
                if len(overlap)/len(field1) > overlap_thre or len(overlap)/len(field2) > overlap_thre:
                    size_std.append(np.abs(len(field1)-len(field2))/(len(field1)+len(field2)))
                    is_matched = True
                    break
            
            if is_matched:
                dyn_behav[6, i] += 1
            else:
                dyn_behav[5, i] += 1
        
        dyn_behav[4, i] = place_fields_num1[i] - dyn_behav[6, i]
    
    shuffle_std = []
    # Shuffle test
    dyn_behav_shuf = np.zeros((3, n_neuron, 10))
    for i in range(n_neuron):
        if is_placecell_x[i] == 0 or is_placecell_y[i] == 0:
            continue
        
        for n in range(10):
            fields1 = field_reallocate(place_fields_x[i], maze_type=maze_type)
            fields2 = field_reallocate(place_fields_y[i], maze_type=maze_type)
        
            for k1 in fields1.keys():
                is_matched = False
                field1 = fields1[k1]
                for k2 in fields2.keys():
                    field2 = fields2[k2]
                    overlap = np.intersect1d(field1, field2)
                    if len(overlap)/len(field1) > overlap_thre or len(overlap)/len(field2) > overlap_thre:
                        shuffle_std.append(np.abs(len(field1)-len(field2))/(len(field1)+len(field2)))
                        is_matched = True
                        break
            
                if is_matched:
                    dyn_behav_shuf[2, i, n] += 1
                else:
                    dyn_behav_shuf[1, i, n] += 1
        
            dyn_behav_shuf[0, i, n] = place_fields_num1[i] - dyn_behav_shuf[2, i, n]
    
    return dyn_behav, np.mean(dyn_behav_shuf, axis=2)

def field_tracker(trace: dict, overlap_thre: float = 0.6):
    n_sessions = trace['n_sessions']
    
    Data = {
        "Start Session": [],
        "Interval": [],
        "Cell Pair Number": [],
        "Turn-On": [],
        "Turn-On Proportion": [],
        "Turn-Off": [],
        "Turn-Off Proportion": [],
        "Kept-On": [],
        "Kept-On Proportion": [],
        "Kept-Off": [],
        "Kept-Off Proportion": [],
        "Prev Field Number": [],
        "Next Field Number": [],
        "Field-On": [],
        "Field-On Proportion": [],
        "Field-Off": [],
        "Field-Off Proportion": [],
        "Field-Kept": [],
        "Field-Kept Proportion": [],
        "Data Type": [],
    }
    
    index_map = trace['index_map']
    is_placecell = trace['is_placecell']
    field_number = trace['place_field_numbers']
    
    
    for i in range(n_sessions-1):
        if i == 5 and trace['MiceID'] == 10212 and trace['Stage'] in ['Stage 1', 'Stage 1+2']:
            continue
        for j in range(i+1, n_sessions):
            if j == 5 and trace['MiceID'] == 10212 and trace['Stage'] in ['Stage 1', 'Stage 1+2']:
                continue
            
            print(f"Session {i+1} vs Session {j+1}, Interval {j-i}: Tracking fields...")
            idx = np.where((index_map[i, :] != 0) & (index_map[j, :] != 0))[0]
            if len(idx) <= 10:
                print(f"Only {len(idx)} cell pairs are found")
                continue
                
            print(f"    Cell Pairs: {len(idx)}")
            dyn_behav, dyn_behav_shuf = match_fields(
                index_map1 = index_map[i, idx],
                index_map2 = index_map[j, idx],
                place_fields_num1 = trace['place_field_numbers'][i, idx],
                place_fields_num2 = trace['place_field_numbers'][j, idx],
                is_placecell_x = is_placecell[i, idx],
                is_placecell_y = is_placecell[j, idx],
                place_fields_x = [trace['place_field_all'][i][k] for k in idx],
                place_fields_y = [trace['place_field_all'][j][k] for k in idx],
                field_pool_x = trace['field_pool'][i],
                field_pool_y = trace['field_pool'][j],
                maze_type=trace['maze_type'],
                overlap_thre=overlap_thre
            )
            
            pc_idx = np.where((index_map[i, idx] != 0) & 
                              (index_map[j, idx] != 0) & 
                              (is_placecell[i, idx] == 1) &
                              (is_placecell[j, idx] == 1))[0]
            print("  Info of Field-tracking ---------------------------------")
            print(f"  PC-PC Pairs:            {np.sum(dyn_behav[2, :])}/{len(idx)}, Prop. {round(np.sum(dyn_behav[2, :])/len(idx)*100, 3)}%")
            print(f"  PC-nPC Turn-off Pairs:  {np.sum(dyn_behav[1, :])}/{len(idx)}, Prop. {round(np.sum(dyn_behav[1, :])/len(idx)*100, 3)}%")
            print(f"  nPC-PC Turn-on Pairs:   {np.sum(dyn_behav[0, :])}/{len(idx)}, Prop. {round(np.sum(dyn_behav[0, :])/len(idx)*100, 3)}%")
            print(f"  nPC-nPC Kept-off Pairs: {np.sum(dyn_behav[3, :])}/{len(idx)}, Prop. {round(np.sum(dyn_behav[3, :])/len(idx)*100, 3)}%")

            field_num_prev = np.sum(field_number[i, idx[pc_idx]])
            field_num_next = np.sum(field_number[j, idx[pc_idx]])
            print(f"  -----------Tracking fields {field_num_prev}->{field_num_next}-------------")
            print(f"  Field On:   {np.sum(dyn_behav[4, :])}/{field_num_next}, Prop. {round(np.sum(dyn_behav[4, :])/field_num_next*100, 3)}%")
            print(f"    - Shuffle {np.sum(dyn_behav_shuf[0, :])}/{field_num_prev}, Prop. {round(np.sum(dyn_behav_shuf[0, :])/field_num_prev*100, 3)}%")
            print(f"  Field Off:  {np.sum(dyn_behav[5, :])}/{field_num_prev}, Prop. {round(np.sum(dyn_behav[5, :])/field_num_prev*100, 3)}%")
            print(f"    - Shuffle {np.sum(dyn_behav_shuf[1, :])}/{field_num_prev}, Prop. {round(np.sum(dyn_behav_shuf[1, :])/field_num_prev*100, 3)}%")
            print(f"  Field Kept: {np.sum(dyn_behav[6, :])}/{field_num_prev}, Prop. {round(np.sum(dyn_behav[6, :])/field_num_prev*100, 3)}%")
            print(f"    - Shuffle {np.sum(dyn_behav_shuf[2, :])}/{field_num_prev}, Prop. {round(np.sum(dyn_behav_shuf[2, :])/field_num_prev*100, 3)}%", end="\n\n")
            
            Data["Start Session"] = Data["Start Session"] + [i+1, i+1]
            Data["Interval"] = Data["Interval"] + [j-i, j-i]
            Data["Cell Pair Number"] = Data["Cell Pair Number"] + [len(idx), len(idx)]
            Data["Turn-On"] = Data["Turn-On"] + [np.sum(dyn_behav[0, :]), np.sum(dyn_behav[0, :])]
            Data["Turn-On Proportion"] = Data["Turn-On Proportion"] + [np.sum(dyn_behav[0, :])/len(idx)*100, np.sum(dyn_behav[0, :])/len(idx)*100]
            Data["Turn-Off"] = Data["Turn-Off"] + [np.sum(dyn_behav[1, :]), np.sum(dyn_behav[1, :])]
            Data["Turn-Off Proportion"] = Data["Turn-Off Proportion"] + [np.sum(dyn_behav[1, :])/len(idx)*100, np.sum(dyn_behav[1, :])/len(idx)*100]
            Data["Kept-On"] = Data["Kept-On"] + [np.sum(dyn_behav[2, :]), np.sum(dyn_behav[2, :])]
            Data["Kept-On Proportion"] = Data["Kept-On Proportion"] + [np.sum(dyn_behav[2, :])/len(idx)*100, np.sum(dyn_behav[2, :])/len(idx)*100]
            Data["Kept-Off"] = Data["Kept-Off"] + [np.sum(dyn_behav[3, :]), np.sum(dyn_behav[3, :])]
            Data["Kept-Off Proportion"] = Data["Kept-Off Proportion"] + [np.sum(dyn_behav[3, :])/len(idx)*100, np.sum(dyn_behav[3, :])/len(idx)*100]
            
            Data['Prev Field Number'] = Data['Prev Field Number'] + [field_num_prev, field_num_prev]
            Data['Next Field Number'] = Data['Next Field Number'] + [field_num_next, field_num_next]
            Data['Field-On'] = Data['Field-On'] + [np.sum(dyn_behav[4, :]), np.sum(dyn_behav[0, :])]
            Data['Field-On Proportion'] = Data['Field-On Proportion'] + [np.sum(dyn_behav[4, :])/field_num_prev*100, np.sum(dyn_behav_shuf[0, :])/field_num_prev*100]
            Data['Field-Off'] = Data['Field-Off'] + [np.sum(dyn_behav[5, :]), np.sum(dyn_behav[1, :])]
            Data['Field-Off Proportion'] = Data['Field-Off Proportion'] + [np.sum(dyn_behav[5, :])/field_num_prev*100, np.sum(dyn_behav_shuf[1, :])/field_num_prev*100]
            Data['Field-Kept'] = Data['Field-Kept'] + [np.sum(dyn_behav[6, :]), np.sum(dyn_behav[2, :])]
            Data['Field-Kept Proportion'] = Data['Field-Kept Proportion'] + [np.sum(dyn_behav[6, :])/field_num_prev*100, np.sum(dyn_behav_shuf[2, :])/field_num_prev*100]
            Data['Data Type'] = Data['Data Type'] + ['Data', 'Shuffle']
    
    del trace        
    for k in Data.keys():
        Data[k] = np.array(Data[k])
    
    return Data
    

def run_all_mice_multiday(
    i: int,
    f: pd.DataFrame = f_CellReg_day,
    overlap_thre: float = 0.75
):
    if f['maze_type'][i] == 0:
        return f
    
    line = i
    cellreg_dir = f['cellreg_folder'][i]
    mouse = int(f['MiceID'][i])
    stage = f['Stage'][i]
    session = int(f['session'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = f['paradigm'][i]
    
    index_map = GetMultidayIndexmap(
        mouse,
        stage=stage,
        session=session,
        i = i,
        occu_num=2
    )
    # Initial basic elements
    n_neurons = index_map.shape[1]
    n_sessions = index_map.shape[0]    

    # Get information from daily trace.pkl
    core = MultiDayCore(
        keys = ['correct_nodes', 'correct_time', 'correct_pos', 
                'ms_time_behav', 'Spikes', 'spike_nodes', 'lap beg time', 'lap end time',
                'smooth_map_all', 'SI_all', 'is_placecell', 
                'DeconvSignal', 'ms_time', 'spike_nodes_original', 
                'place_field_all_multiday', 'is_perfect']
    )
    file_indices = np.where((f1['MiceID'] == mouse) & (f1['Stage'] == stage) & (f1['session'] == session))[0]
    
    if mouse in [11095, 11092]:
        file_indices = file_indices[3:]
    
    if stage == 'Stage 1+2':
        file_indices = np.where((f1['MiceID'] == mouse) & (f1['session'] == session) & ((f1['Stage'] == 'Stage 1') | (f1['Stage'] == 'Stage 2')))[0]
        
    print(file_indices, mouse, stage, session)
    res = core.get_trace_set(f=f1, file_indices=file_indices, keys=['correct_nodes', 'correct_time', 'correct_pos', 
                'ms_time_behav', 'Spikes', 'spike_nodes', 'lap beg time', 'lap end time',
                'smooth_map_all', 'SI_all', 'is_placecell', 
                'DeconvSignal', 'ms_time', 'spike_nodes_original', 
                'place_field_all_multiday', 'is_perfect'])
    
    lap_id, session_id, is_perfect = [], [], []
    
    # Generate a global time frame for each cell
    assert len(res['correct_time']) == len(res['ms_time_behav']) and len(res['correct_time']) == len(res['ms_time'])
    for i in range(len(res['correct_time'])-1):
        dt = np.max([np.max(res['correct_time'][i]), np.max(res['ms_time_behav'][i]), np.max(res['ms_time'][i])]) + 50000
        res['correct_time'][i+1] = res['correct_time'][i+1] + dt
        res['ms_time_behav'][i+1] = res['ms_time_behav'][i+1] + dt
        res['ms_time'][i+1] = res['ms_time'][i+1] + dt
        res['lap beg time'][i+1] = res['lap beg time'][i+1] + dt
        res['lap end time'][i+1] = res['lap end time'][i+1] + dt
        lap_id.append(np.arange(1, len(res['lap beg time'][i])+1))
        session_id.append(np.repeat(i+1, len(res['lap beg time'][i])))
        is_perfect.append(res['is_perfect'][i])
        
    lap_id.append(np.arange(1, len(res['lap beg time'][-1])+1))
    session_id.append(np.repeat(session, len(res['lap beg time'][-1])))
    is_perfect.append(res['is_perfect'][-1])
    lap_id = np.concatenate(lap_id)
    session_id = np.concatenate(session_id)
    is_perfect = np.concatenate(is_perfect)
    laps_info = np.zeros((session_id.shape[0], 3), dtype=np.int64)
    laps_info[:, 0] = session_id
    laps_info[:, 1] = lap_id
    laps_info[:, 2] = is_perfect
    
        
    SI_all = np.zeros((n_sessions, n_neurons), dtype=np.float64)*np.nan
    is_placecell = np.zeros((n_sessions, n_neurons), dtype=np.float64)*np.nan
    smooth_map_all = np.zeros((n_sessions, n_neurons, 2304), dtype=np.float64)*np.nan
    place_field_numbers = np.zeros((n_sessions, n_neurons), dtype=np.float64)*np.nan
    place_field_all = []
    
    behav_len = np.zeros(n_sessions, dtype=np.int64)
    ms_len = np.zeros(n_sessions, dtype=np.int64)
    ms_ori_len = np.zeros(n_sessions, dtype=np.int64)
    
    for i in range(n_sessions):
        behav_len[i] = len(res['correct_time'][i])
        ms_len[i] = len(res['ms_time_behav'][i])
        ms_ori_len[i] = len(res['ms_time'][i])
    
    ms_sum = np.sum(ms_len)
    ms_ori_sum = np.sum(ms_ori_len)
    
    behav_len = np.append([0], np.cumsum(behav_len))
    ms_len = np.append([0], np.cumsum(ms_len))
    ms_ori_len = np.append([0], np.cumsum(ms_ori_len))
    
    DeconvSignal = np.zeros((n_neurons, ms_ori_sum), dtype=np.float64)*np.nan
    Spikes = np.zeros((n_neurons, ms_sum), dtype=np.float64)*np.nan
    behav_indices, ms_indices, ms_ori_indices = [], [], []
    
    field_pool = [[] for _ in range(n_sessions)]
    place_field_all = [[] for _ in range(n_sessions)]
    for j in range(n_neurons):
        behav_idx, ms_idx, ms_ori_idx = [], [], []
        for i in range(n_sessions):
            cell_id = int(index_map[i, j])
            if cell_id != 0:
                Spikes[j, ms_len[i]:ms_len[i+1]] = res['Spikes'][i][cell_id-1, :]
                DeconvSignal[j, ms_ori_len[i]:ms_ori_len[i+1]] = res['DeconvSignal'][i][cell_id-1, :]
                SI_all[i, j] = res['SI_all'][i][cell_id-1]
                is_placecell[i, j] = res['is_placecell'][i][cell_id-1]
                smooth_map_all[i, j, :] = res['smooth_map_all'][i][cell_id-1, :]
                
                if is_placecell[i, j] == 1:
                    place_field_numbers[i, j] = len(res['place_field_all_multiday'][i][cell_id-1].keys())
                    
                    for k in res['place_field_all_multiday'][i][cell_id-1].keys():
                        field_pool[i].append(res['place_field_all_multiday'][i][cell_id-1][k])

                behav_idx.append(np.arange(behav_len[i], behav_len[i+1]))
                ms_idx.append(np.arange(ms_len[i], ms_len[i+1]))
                ms_ori_idx.append(np.arange(ms_ori_len[i], ms_ori_len[i+1]))
                place_field_all[i].append(res['place_field_all_multiday'][i][cell_id-1])
                
            else:
                place_field_all[i].append(None)
        
        behav_idx = np.concatenate(behav_idx)
        ms_idx = np.concatenate(ms_idx)
        ms_ori_idx = np.concatenate(ms_ori_idx)
        
        behav_indices.append(behav_idx)
        ms_indices.append(ms_idx)
        ms_ori_indices.append(ms_ori_idx)
                
    for k in ['correct_nodes', 'correct_time', 'lap beg time', 'lap end time', 
              'ms_time_behav', 'spike_nodes', 'ms_time', 'spike_nodes_original']:
        res[k] = np.concatenate(res[k])
    
    res['correct_pos'] = np.concatenate(res['correct_pos'], axis=0)
    
    # Documentation for trace variables
    # SI_all: spatial information of each neuron, 
    #       shape: (n_sessions, n_neurons)
    # is_placecell: whether each neuron is place cell,
    #       shape: (n_sessions, n_neurons)
    # smooth_map_all: spatial information of each neuron, 
    #       shape: (n_sessions, n_neurons, 2304)
    # place_field_all: place fields of each neuron, 
    #       list[list[dict]], shape: (n_sessions, n_neurons, n_fields), where n_fields is varied.
    # place_field_numbers: number of place fields of each neuron, 
    #       shape: (n_sessions, n_neurons)
    # DeconvSignal: deconvolved signal of each neuron, 
    #       shape: (n_neurons, T3)
    # spike_nodes_original: original spike nodes of each neuron, 
    #       shape: (T3, )
    # ms_time: ms time of each neuron,
    #       shape: (T3, )
    # Spikes: spikes of each neuron,
    #       shape: (n_neurons, T2)
    # spike_nodes: spike nodes of each neuron,
    #       shape: (T2, )
    # ms_time_behav: ms time of each neuron, 
    #       shape: (T2, )
    
    # behav_indices: indices of behavior frames of each neuron, list[n_neurons, ], varied
    # ms_indices: indices of ms frames of each neuron, list[n_neurons, ], varied
    # ms_ori_indices: indices of ms original frames of each neuron, list[n_neurons, ], varied
    # n_neurons: number of neurons
    # n_sessions: number of sessions
    # maze_type: maze type
    # correct_nodes: correct nodes of each neuron, 
    #       shape: (T1, )
    # correct_time: correct time of each neuron,
    #       shape: (T1, )
    # correct_pos: correct position of each neuron,
    #       shape: (T1, 2)
    # lap beg time: begin time of each lap,
    #       shape: (n_laps, )
    # lap end time: end time of each lap,
    #       shape: (n_laps, )
    # laps_info: information of each lap,
    #       shape: (n_laps, 2)
    # index_map: index map of each neuron, 
    #       shape: (n_sessions, n_neurons)
    # field_pool: field pool of each neuron,

    trace = {"MiceID": mouse, "Stage": stage, "session": session, "maze_type": maze_type, "paradigm": behavior_paradigm,
             "SI_all": SI_all, "is_placecell": is_placecell, "smooth_map_all": smooth_map_all, "place_field_all": place_field_all, 
             "place_field_numbers": place_field_numbers,
              "n_neurons": n_neurons, "n_sessions": n_sessions, "maze_type": maze_type,
             "lap beg time": res['lap beg time'], "lap end time": res['lap end time'], "laps_info": laps_info, 
             "index_map": index_map.astype(np.int64), "field_pool": field_pool}

    trace['res'] = field_tracker(trace, overlap_thre = overlap_thre)
    print("Field Register...")
    trace = field_register(trace)
    with open(os.path.join(os.path.dirname(os.path.dirname(cellreg_dir)), "trace_mdays.pkl"), 'wb') as handle:
        print(type(trace['is_placecell']), os.path.join(os.path.dirname(os.path.dirname(cellreg_dir)), "trace_mdays.pkl"))
        pickle.dump(trace, handle)
        
    del res
    del trace
    gc.collect()


if __name__ == "__main__":
    idx = np.where((f_CellReg_day['maze_type'] != 0))[0]
    #run_all_mice_multiday(i=12, overlap_thre=0.7)

    for i in range(37, len(f_CellReg_day)):
        if f_CellReg_day['include'][i] == 0 or f_CellReg_day['maze_type'][i] == 0:
            continue
    
        print(i)
        run_all_mice_multiday(i, f_CellReg_day, overlap_thre=0.6)
        
        """
        with open(f_CellReg_day['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        trace = field_register(trace)
        
        with open(f_CellReg_day['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
        """
    """
    """
        