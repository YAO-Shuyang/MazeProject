import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import copy as cp

from mylib.multiday.core import MultiDayCore
import scipy.stats
from mylib import LocTimeCurveAxes
from mylib.maze_graph import NRG, maze_graphs, correct_paths
from mylib.maze_utils3 import SmoothMatrix, mkdir, Clear_Axes, DrawMazeProfile, spike_nodes_transform, GetDMatrices
from mylib.preprocessing_ms import shuffle_test_all, shuffle_test_isi, shuffle_test_shift, calc_SI, RateMap

def get_singleday_data(
    f: pd.DataFrame,
    file_indices: np.ndarray,
) -> dict:
    core = MultiDayCore(
        keys=['correct_time', 'correct_pos', 'correct_nodes', 'ms_time', 'ms_time_behav', 'dt',
              'Spikes', 'spike_nodes', 'DeconvSignal', 'is_placecell', 'maze_type']
    )
    return core.get_trace_set(
        f=f, 
        file_indices=file_indices, 
        keys=['correct_time', 'correct_pos', 'correct_nodes', 'ms_time', 'ms_time_behav', 'dt',
              'Spikes', 'spike_nodes', 'DeconvSignal', 'is_placecell', 'maze_type']
    )

def get_placecell_identity(
    index_map: np.ndarray,
    data: dict
) -> np.ndarray:
    is_placecells = np.zeros_like(index_map, dtype=np.float64) * np.nan
    index_map = index_map.astype(np.int64)
    
    for i in range(index_map.shape[0]):
        idx = np.where(index_map[i, :]!=0)[0]
        is_placecells[i, idx] = data['is_placecell'][i][index_map[i, idx]-1]
        
    return is_placecells
    

def concat_multiday_cell(
    data: dict,
    indices: list,
    cell_indices: np.ndarray
) -> dict:
    res = {}
    
    res['behav_label'] = np.concatenate([np.repeat(idx+1, data['correct_time'][idx].shape[0]) for idx in indices])
    res['ms_behav_label'] = np.concatenate([np.repeat(idx+1, data['ms_time_behav'][idx].shape[0]) for idx in indices])
    res['ms_label'] = np.concatenate([np.repeat(idx+1, data['ms_time'][idx].shape[0]) for idx in indices])
    
    for k in ['correct_time', 'correct_nodes', 'ms_time', 'ms_time_behav', 'dt',
               'spike_nodes']:
        res[k] = np.concatenate([data[k][idx] for idx in indices])
        
    for k in ['correct_pos']:
        res[k] = np.concatenate([data[k][idx] for idx in indices], axis=0)
    
    for k in ['Spikes', 'DeconvSignal']:
        res[k] = np.concatenate([data[k][indices[i]][cell_indices[i], :] for i in range(cell_indices.shape[0])])
        
    return res


def main_for_entirecells(
    f: pd.DataFrame,
    file_indices: np.ndarray,
    index_map: np.ndarray,
    shuffle_n: int = 1000,
    save_loc: str = None
):
    index_map = index_map.astype(np.int64)
    data = get_singleday_data(
        f=f,
        file_indices=file_indices
    )
    
    is_placecells = get_placecell_identity(
        index_map=index_map,
        data=data
    )
    print(index_map.shape)
    # The number of place cells should more than 5, or the cell list is excluded for analysis.
    placecell_num = np.nansum(is_placecells, axis=0)
    index_map = index_map[:, placecell_num >= 5]
    is_placecells = is_placecells[:, placecell_num >= 5]
    
    SI_all = np.zeros(index_map.shape[1], np.float64)
    is_placecell_isi = np.zeros(index_map.shape[1], np.int64)
    is_placecell_shift = np.zeros(index_map.shape[1], np.int64)
    is_placecell_all = np.zeros(index_map.shape[1], np.int64)
    is_placecell = np.zeros(index_map.shape[1], np.int64)
    entire_map_all = np.zeros((index_map.shape[1], 2304), np.float64)
    mean_rate_all = np.zeros(index_map.shape[1], np.float64)
    t_total = np.zeros(index_map.shape[1], np.float64)
    t_nodes_frac = np.zeros((index_map.shape[1], 2304), np.float64)
    info = []

    n_neuron = index_map.shape[1]
    
    Ms = SmoothMatrix(maze_type = data['maze_type'][0], sigma = 2, _range = 7, nx = 48)

    _nbins = 2304
    _coords_range = [0, _nbins +0.0001 ]

    print("1. Calculate rate map...")
    for n in tqdm(range(n_neuron)):
        indices = np.where(index_map[:, n] != 0)[0]
        res = concat_multiday_cell(
            data=data,
            indices=indices,
            cell_indices=index_map[indices, n]-1
        )
        
        occu_time, _, _ = scipy.stats.binned_statistic(
            res['spike_nodes'],
            res['dt'],
            bins=_nbins,
            statistic="sum",
            range=_coords_range
        )
        
        res['occu_time'] = occu_time
        
        spike_count = np.zeros(_nbins, dtype = np.float64)
        for i in range(_nbins):
            idx = np.where(res['spike_nodes'] == i+1)[0]
            spike_count[i] = np.nansum(res['Spikes'][idx])

        entire_map_all[n, :] = spike_count/(occu_time/1000+ 1E-9)
        t_total[n] = np.nansum(occu_time)/1000
        t_nodes_frac[n, :] = occu_time / 1000 / (t_total[n]+ 1E-6)
        mean_rate_all[n] = np.nansum(res['Spikes'])/t_total[n]
        
        info.append(res)
        
    entire_map_all[np.isnan(entire_map_all)] = 0
    smooth_map_all = np.dot(entire_map_all, Ms.T)
    
    # Calculate SI
    print("2. Calculate SI...")
    logArg = (entire_map_all.T / mean_rate_all.T).T;
    logArg[np.where(logArg == 0)] = 1; # keep argument in log non-zero
    IC = np.nansum(t_nodes_frac * entire_map_all * np.log2(logArg), axis = 1) # information content
    SI_all = IC / mean_rate_all; # spatial information (bits/spike)
    
    print("3. Shuffle test...")
    for i in tqdm(range(n_neuron)):
        is_placecell_isi[i] = shuffle_test_isi(SI = SI_all[i], spikes = info[i]['Spikes'], spike_nodes=info[i]['spike_nodes'], 
            occu_time=info[i]['occu_time'], Ms = Ms, shuffle_n = shuffle_n)
        """
        is_placecell_shift[i] = shuffle_test_shift(SI = SI_all[i], spikes = info[i]['Spikes'], spike_nodes=info[i]['spike_nodes'], 
            occu_time=info[i]['occu_time'], Ms = Ms, shuffle_n = shuffle_n)
        is_placecell_all[i] = shuffle_test_all(SI = SI_all[i], spikes = info[i]['Spikes'], spike_nodes=info[i]['spike_nodes'], 
            occu_time=info[i]['occu_time'], Ms = Ms, shuffle_n = shuffle_n)
        """
        
    #is_placecell = is_placecell_all + is_placecell_shift + is_placecell_isi
    #is_placecell = np.where(is_placecell == 3, 1, 0)
    
    trace = {'index_map': index_map, 'is_placecells': is_placecells, 'n_neuron': index_map.shape[1],
             'mean_rate_all': mean_rate_all, 't_total': t_total, 't_nodes_frac': t_nodes_frac, 'SI_all': SI_all,
             'is_placecell': is_placecell_isi, 'rate_map_all': entire_map_all, 'smooth_map_all': smooth_map_all,
             'info': info, 'p': save_loc, 'MiceID': f['MiceID'][file_indices[0]], 'maze_type': f['maze_type'][file_indices[0]]}
    
    mkdir(save_loc)
    with open(os.path.join(save_loc, 'trace_multiday.pkl'), 'wb') as handle:
        pickle.dump(trace, handle)
    
    return trace

def TraceMap(trace: dict) -> dict:
    fig = plt.figure(figsize=(6,4))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(maze_type=trace['maze_type'], axes=ax, nx=48)

    
def LocTimeCurve(trace):
    D = GetDMatrices(trace['maze_type'], nx=48)
    """

    """
    fig = plt.figure(figsize=(4,6))
    ax = plt.axes()

    save_loc = os.path.join(trace['p'],'LocTimeCurve')
    mkdir(save_loc)
    
    CP = correct_paths[trace['maze_type']]
    Graph = cp.deepcopy(NRG[int(trace['maze_type'])])
    
    for i in tqdm(range(trace['n_neuron'])):
        ms_time = cp.deepcopy(trace['info'][i]['ms_time_behav'])
        spike_nodes = spike_nodes_transform(trace['info'][i]['spike_nodes'], nx=12).astype(np.int64)
        
        linearized_x = np.zeros_like(spike_nodes, np.float64)
    
        for k in range(spike_nodes.shape[0]):
            linearized_x[k] = Graph[spike_nodes[k]]
        
        linearized_x = linearized_x + np.random.rand(spike_nodes.shape[0]) - 0.5
        
        color = 'red' if trace['is_placecell'][i] == 1 else 'black'
        ax, a1, b1 = LocTimeCurveAxes(
            ax, 
            behav_time=ms_time,
            spikes=trace['info'][i]['Spikes'], 
            spike_time=trace['info'][i]['ms_time_behav'], 
            maze_type=trace['maze_type'],
            given_x=linearized_x,
            title='Cell '+str(i+1),
            title_color=color,
        )
        ax.set_xlim([0, len(CP)+1])

        plt.savefig(os.path.join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(os.path.join(save_loc, str(i+1)+'.svg'), dpi = 600)
        a = a1 + b1
        for j in a:
            j.remove()
            
    return trace
    
if __name__ == '__main__':
    import pickle
    
    with open(r"E:\Data\Cross_maze\10209\Multiday-Stage 2-Maze 1\trace_multiday.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    trace = LocTimeCurve(trace)
    
    from mylib.statistic_test import Read_and_Sort_IndexMap  
    from mylib.local_path import f1, cellReg_09_maze1_2, cellReg_95_maze1
    #file_indices = np.where((f1['MiceID']==11095)&(f1['maze_type']==1)&(f1['Stage'] == 'Stage 2')&(f1['date']>=20220820))[0]      
    file_indices = np.where((f1['MiceID']==10209)&(f1['maze_type']==1)&(f1['Stage'] == 'Stage 2'))[0]      
    index_map = Read_and_Sort_IndexMap(
        path = cellReg_09_maze1_2,
        occur_num=6,
        name_label='SFP2023',#"SFP2022", name_label=
        order=np.array(['20230709', '20230711', '20230713',
                    '20230715', '20230717', '20230719', '20230721', '20230724', '20230726', 
                    '20230728'])
    )
    
    trace = main_for_entirecells(
        f1,
        file_indices,
        index_map,
        save_loc=r"E:\Data\Cross_maze\10209\Multiday-Stage 2-Maze 1"
    )

    
    
    
    