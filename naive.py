import numpy as np
from mylib.maze_graph import *
from mylib.maze_utils3 import spike_nodes_transform

def get_graph_adjacent_matrix(
    maze_type: int = 1,
    nx: int = 12
):
    B = np.zeros((nx**2, nx**2)) * np.nan
    G = maze_graphs[(maze_type, nx)]
    for i in range(nx**2):
        for j in G[i+1]:
            B[i, j-1] = 0
    
    return B

def get_exploration_progress(
    behav_nodes: np.ndarray,
    maze_type: int = 1,
    nx: int = 12
):
    behav_nodes = spike_nodes_transform(spike_nodes=behav_nodes, nx = nx)
    behav_progress = np.zeros_like(behav_nodes)

    diff = np.ediff1d(behav_nodes)
    shift = np.where(diff != 0)[0]
    COUNT = np.zeros((nx**2, nx**2), int)
    
    for i in range(shift.shape[0]):
        pass
        
    return np.sum(B[behav_nodes])