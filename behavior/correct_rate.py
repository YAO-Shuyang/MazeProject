from mylib.maze_utils3 import spike_nodes_transform
from mylib.divide_laps.lap_split import LapSplit
from mylib.maze_graph import DPs, correct_paths, StartPoints, EndPoints, maze_graphs, DP_DSP, CP_DSP
import numpy as np
import copy as cp
from tqdm import tqdm

def calc_behavioral_score(trace: dict) -> tuple[int, int]:
    maze_type = trace['maze_type']
    behav_nodes = spike_nodes_transform(cp.deepcopy(trace['correct_nodes']), nx=12)
    behav_time = cp.deepcopy(trace['correct_time'])
    
    DP = cp.deepcopy(DPs[int(maze_type)])
    CP = cp.deepcopy(correct_paths[int(maze_type)])
    G = cp.deepcopy(maze_graphs[(int(maze_type), 12)])
    start_point, end_point = StartPoints[int(maze_type)], EndPoints[int(maze_type)]
    
    pass_num = 0
    err_num = 0
    weight = []
    
    for i in DP:
    
        if i == start_point or i == end_point or i not in CP: # Do not consider the correct rate at decision point
            continue
    
        behav_indices = np.where(behav_nodes == i)[0]
        in_field_time = behav_time[behav_indices].astype(np.float64)
        in_field_node = behav_nodes[behav_indices].astype(np.float64)
        dt = np.ediff1d(in_field_time)
        intervals = np.where(dt >= 200)[0]
        in_field_time = np.insert(in_field_time, intervals+1, np.nan)
        in_field_node = np.insert(in_field_node, intervals+1, np.nan)
    
        interval_pos = np.concatenate([[-1], np.where(np.isnan(in_field_time))[0], [in_field_time.shape[0]]])
    
        for j in range(interval_pos.shape[0]-1):
            beg = interval_pos[j] + 1
            end = interval_pos[j+1]
        
            behav_beg = np.where(behav_time == in_field_time[beg])[0][0]
            behav_end = np.where(behav_time == in_field_time[end-1])[0][0]
            
            if behav_beg == 0 or behav_end == behav_time.shape[0]-1:
                continue
            
            prev = behav_nodes[behav_beg-1]
            next = behav_nodes[behav_end+1]
            
            if prev not in CP:
                continue
            
            dis_to_start_curr = np.where(CP==i)[0][0]
            dis_to_start_prev = np.where(CP==prev)[0][0]
            
            if dis_to_start_prev >= dis_to_start_curr:
                continue
            
            pass_num += 1
            
            weight.append(len(G[i])-1)
            
            if next not in CP or prev == next:
                err_num += 1

    standard_err_rate = 1 - pass_num/sum(weight)

    return err_num, pass_num, standard_err_rate

def lapwise_behavioral_score(trace: dict):
    maze_type = trace['maze_type']
    behav_nodes = spike_nodes_transform(cp.deepcopy(trace['correct_nodes']), nx=12)
    behav_time = cp.deepcopy(trace['correct_time'])
    
    DP = cp.deepcopy(DPs[int(maze_type)])
    CP = cp.deepcopy(correct_paths[int(maze_type)])
    start_point, end_point = StartPoints[int(maze_type)], EndPoints[int(maze_type)]
    
    pass_nums = np.zeros(17)
    err_nums = np.zeros(17)
        
    for n, i in enumerate(DP):
    
        if i == start_point or i == end_point or i not in CP: # Do not consider the correct rate at decision point
            continue
    
        behav_indices = np.where(behav_nodes == i)[0]
        in_field_time = behav_time[behav_indices].astype(np.float64)
        in_field_node = behav_nodes[behav_indices].astype(np.float64)
        dt = np.ediff1d(in_field_time)
        intervals = np.where(dt >= 200)[0]
        in_field_time = np.insert(in_field_time, intervals+1, np.nan)
        in_field_node = np.insert(in_field_node, intervals+1, np.nan)
    
        interval_pos = np.concatenate([[-1], np.where(np.isnan(in_field_time))[0], [in_field_time.shape[0]]])
    
        for j in range(interval_pos.shape[0]-1):
            beg = interval_pos[j] + 1
            end = interval_pos[j+1]
        
            behav_beg = np.where(behav_time == in_field_time[beg])[0][0]
            behav_end = np.where(behav_time == in_field_time[end-1])[0][0]
        
            if behav_beg == 0 or behav_end == behav_time.shape[0]-1:
                continue
            
            prev = behav_nodes[behav_beg-1]
            next = behav_nodes[behav_end+1]
            
            if prev not in CP:
                continue
            
            dis_to_start_curr = np.where(CP==i)[0][0]
            dis_to_start_prev = np.where(CP==prev)[0][0]
            
            if dis_to_start_prev >= dis_to_start_curr:
                continue
            
            pass_nums[n] += 1
            
            if next not in CP or prev == next:
                err_nums[n] += 1

    return pass_nums, err_nums

def calc_behavioral_score_dsp(
    route: int,
    behav_nodes, 
    behav_time
) -> tuple[int, int]:
    # DSP is confined to Maze A (maze 1)
    assert behav_nodes is not None
    assert behav_time is not None
    
    DP = cp.deepcopy(DP_DSP[route])
    CP = cp.deepcopy(CP_DSP[route])
    G = cp.deepcopy(maze_graphs[(1, 12)])
    start_point, end_point = CP[0], EndPoints[1]
    
    pass_num = 0
    err_num = 0
    weight = []
    
    for i in DP:
    
        if i == start_point or i == end_point or i not in CP: # Do not consider the correct rate at decision point
            continue
    
        behav_indices = np.where(behav_nodes == i)[0]
        in_field_time = behav_time[behav_indices].astype(np.float64)
        in_field_node = behav_nodes[behav_indices].astype(np.float64)
        dt = np.ediff1d(in_field_time)
        intervals = np.where(dt >= 200)[0]
        in_field_time = np.insert(in_field_time, intervals+1, np.nan)
        in_field_node = np.insert(in_field_node, intervals+1, np.nan)
    
        interval_pos = np.concatenate([[-1], np.where(np.isnan(in_field_time))[0], [in_field_time.shape[0]]])
    
        for j in range(interval_pos.shape[0]-1):
            beg = interval_pos[j] + 1
            end = interval_pos[j+1]
        
            behav_beg = np.where(behav_time == in_field_time[beg])[0][0]
            behav_end = np.where(behav_time == in_field_time[end-1])[0][0]
            
            if behav_beg == 0 or behav_end == behav_time.shape[0]-1:
                continue
            
            prev = behav_nodes[behav_beg-1]
            next = behav_nodes[behav_end+1]
            
            if prev not in CP:
                continue
            
            dis_to_start_curr = np.where(CP==i)[0][0]
            dis_to_start_prev = np.where(CP==prev)[0][0]
            
            if dis_to_start_prev >= dis_to_start_curr:
                continue
            
            pass_num += 1
            
            weight.append(len(G[i])-1)
            
            if next not in CP or prev == next:
                err_num += 1

    standard_err_rate = 1 - pass_num/sum(weight)

    return err_num, pass_num, standard_err_rate