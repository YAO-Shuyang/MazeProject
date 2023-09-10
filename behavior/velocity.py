import numpy as np
from mylib.maze_utils3 import Clear_Axes, spike_nodes_transform, GetDMatrices, uniform_smooth_speed, maze_graphs
from mylib.divide_laps.lap_split import LapSplit
import matplotlib.pyplot as plt
from mazepy.behav.graph import Graph
from tqdm import tqdm

def get_linearized_pos(behav_pos: np.ndarray, G: Graph):
    dis = np.zeros_like(behav_pos[:, 0], np.float64)
    for i in tqdm(range(behav_pos.shape[0])):
        dis[i] = G.shortest_distance((behav_pos[i, 0]/80, behav_pos[i, 1]/80), (0.01, 0.03))*8
    return dis[:-1]

from numba import jit
def cumulative_dis(dx, dy):
    dl = np.sqrt(dx**2 + dy**2)
    return np.cumsum(dl)

def visualize_velocity(behav_pos1: np.ndarray, behav_time1: np.ndarray, behav_nodes1: np.ndarray, 
                       behav_pos2: np.ndarray, behav_time2: np.ndarray, behav_nodes2: np.ndarray, 
                       behav_pos3: np.ndarray, behav_time3: np.ndarray, behav_nodes3: np.ndarray, 
                       behav_pos4: np.ndarray, behav_time4: np.ndarray, behav_nodes4: np.ndarray, D: np.ndarray):
    
    x1, x2, x3 = get_linearized_pos(behav_pos1, D), get_linearized_pos(behav_pos2, D), get_linearized_pos(behav_pos3, D)
    x4 = get_linearized_pos(behav_pos4, D)
    behav_time1 = behav_time1 - behav_time1[0]
    behav_time2 = behav_time2 - behav_time2[0]
    behav_time3 = behav_time3 - behav_time3[0]
    behav_time4 = behav_time4 - behav_time4[0]
    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8,6))
    axx, axy = axes[0], axes[1]

    dt = np.ediff1d(behav_time1)
    dx = np.ediff1d(behav_pos1[:, 0])
    dy = np.ediff1d(behav_pos1[:, 1])
    path_length = np.nansum(np.sqrt(dx**2+dy**2))
    #x1 = cumulative_dis(dx, dy) / path_length
    
    axx.plot(x1, dx/dt*1000, label = '10209')
    axy.plot(x1, dy/dt*1000)
    
    #axx, axy = axes[1], axes[5]

    dt = np.ediff1d(behav_time2)
    dx = np.ediff1d(behav_pos2[:, 0])
    dy = np.ediff1d(behav_pos2[:, 1])
    path_length = np.nansum(np.sqrt(dx**2+dy**2))
    #x2 = cumulative_dis(dx, dy) / path_length
    
    axx.plot(x2, dx/dt*1000, label = '10212')
    axy.plot(x2, dy/dt*1000)
    
    #axx, axy = axes[2], axes[6]

    dt = np.ediff1d(behav_time3)
    dx = np.ediff1d(behav_pos3[:, 0])
    dy = np.ediff1d(behav_pos3[:, 1])
    path_length = np.nansum(np.sqrt(dx**2+dy**2))
    #x3 = cumulative_dis(dx, dy) / path_length
    
    axx.plot(x3, dx/dt*1000, label = '11095')
    axy.plot(x3, dy/dt*1000)
    
    
    #axx, axy = axes[3], axes[7]
    dt = np.ediff1d(behav_time4)
    dx = np.ediff1d(behav_pos4[:, 0])
    dy = np.ediff1d(behav_pos4[:, 1])
    path_length = np.nansum(np.sqrt(dx**2+dy**2))
    #x3 = cumulative_dis(dx, dy) / path_length
    
    axx.plot(x4, dx/dt*1000, label = '10224')
    axy.plot(x4, dy/dt*1000)
    axx.legend()
    plt.show()
    
if __name__ == '__main__':
    import pickle
    
    with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", 'rb') as handle:
        trace1 = pickle.load(handle)

    with open(r"E:\Data\Cross_maze\10212\20230728\session 2\trace.pkl", 'rb') as handle:
        trace2 = pickle.load(handle)
        
    with open(r"E:\Data\Cross_maze\11095\20220830\session 2\trace.pkl", 'rb') as handle:
        trace3 = pickle.load(handle)
        
    with open(r"E:\Data\Cross_maze\10224\20230822\session 2\trace_behav.pkl", 'rb') as handle:
        trace4 = pickle.load(handle)
    
    trace4['lap_begin_index'], trace4['lap_end_index'] = LapSplit(trace4)

    beg1, end1 = trace1['lap_begin_index'][28-1], trace1['lap_end_index'][28-1]
    beg2, end2 = trace2['lap_begin_index'][14-1], trace2['lap_end_index'][14-1]
    beg3, end3 = trace3['lap_begin_index'][10-1], trace3['lap_end_index'][10-1]
    beg4, end4 = trace4['lap_begin_index'][17-1], trace4['lap_end_index'][17-1]
    D = Graph(12, 12, maze_graphs[(1, 12)])
    visualize_velocity(trace1['correct_pos'][beg1:end1, :], trace1['correct_time'][beg1:end1], trace1['correct_nodes'][beg1:end1],  
                       trace2['correct_pos'][beg2:end2, :], trace2['correct_time'][beg2:end2], trace2['correct_nodes'][beg2:end2], 
                       trace3['correct_pos'][beg3:end3, :], trace3['correct_time'][beg3:end3], trace3['correct_nodes'][beg3:end3], 
                       trace4['correct_pos'][beg4:end4, :], trace4['correct_time'][beg4:end4], trace4['correct_nodes'][beg4:end4],  D)