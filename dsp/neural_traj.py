from mazepy.datastruc.neuact import SpikeTrain, NeuralTrajectory
from mazepy.datastruc.variables import Variable1D
from mazepy.datastruc.kernel import GaussianKernel1d
import numpy as np
import copy as cp
# Perform PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis as FA
import matplotlib.pyplot as plt
import pandas as pd
from mylib.maze_utils3 import spike_nodes_transform, mkdir
import seaborn as sns
from mylib.maze_graph import NRG
import umap.umap_ as umap
import os

def get_neural_trajectory(trace: dict):
    time_beg, time_end = trace['lap beg time'], trace['lap end time']
    
    neuro_trajs = []
    time_trajs = []
    pos_trajs = []
    route_ids = []
    lap_ids = []
    
    idx = np.where(np.isnan(trace['spike_nodes_original']) == False)[0]
    ms_time = trace['ms_time'][idx]
    spike_nodes = trace['spike_nodes_original'][idx]
    Spikes = trace['Spikes_original'][:, idx]
    speed = trace['ms_speed_original'][idx]
    
    idx = np.where(speed >= 2.5)[0]
    ms_time = ms_time[idx]
    spike_nodes = spike_nodes[idx]
    Spikes = Spikes[:, idx]
    speed = speed[idx]
    
    for i in range(time_beg.shape[0]):
        idx = np.where(
            (ms_time >= time_beg[i]) &
            (ms_time <= time_end[i])
        )[0]
        
        spike_train = SpikeTrain(
            Spikes[:, idx],
            ms_time[idx],
            spike_nodes[idx]
        )
        
        neuro_traj = spike_train.calc_neural_trajectory(
            t_window=500,
            step_size=50
        )
        
        kernel = GaussianKernel1d(n = 20, sigma=0.2)
        neuro_traj.smooth(kernel)
        neuro_trajs.append(neuro_traj.to_array())
        time_trajs.append(neuro_traj.time)
        pos_trajs.append(neuro_traj.variable)
        route_ids.append(np.repeat(trace['route_labels'][i], neuro_traj.shape[1]))
        lap_ids.append(np.repeat(i, neuro_traj.shape[1]))
        
    neuro_trajs = np.concatenate(neuro_trajs, axis=1)
    time_trajs = np.concatenate(time_trajs)
    pos_trajs = np.concatenate(pos_trajs)
    route_ids = np.concatenate(route_ids)
    lap_ids = np.concatenate(lap_ids)
    
    idx = np.where(
        pos_trajs < 2350
    )[0]
    
    trace['neural_traj'] = neuro_trajs[:, idx]
    trace['time_traj'] = time_trajs[idx]
    trace['pos_traj'] = pos_trajs[idx]
    trace['traj_route_ids'] = route_ids[idx]
    trace['traj_lap_ids'] = lap_ids[idx]
    
    return trace

def pca_dim_reduction(trace: dict, n_components=20, UMAPi: int = 0, UMAPj: int = 1):
    if 'neural_traj' not in trace.keys():
        get_neural_trajectory(trace)
        
    neural_traj = trace['neural_traj']
    pos_traj = trace['pos_traj']
    time_traj = trace['time_traj']
    route_ids = trace['traj_route_ids']
    lap_ids = trace['traj_lap_ids']
    
    pos_traj = spike_nodes_transform(pos_traj, 12)
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]
        
    pca = umap.UMAP(n_components=n_components)  # Reduce to 2 dimensions for visualization
    #reduced_data = pca.fit_transform(neural_traj[:, idx].T)
    reduced_data = pca.fit_transform(neural_traj.T)
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6*2, 4))
    reduced_data_temp = reduced_data[:, [UMAPi, UMAPj]]
    
    PC1, PC2 = reduced_data_temp[:, 0], reduced_data_temp[:, 1] # df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    colors = sns.color_palette("rainbow", 7)[::-1]
    idx = np.where(np.ediff1d(lap_ids) != 0)[0]
    beg, end = np.concatenate([[0], idx+1]), np.concatenate([idx, [lap_ids.shape[0]-1]])
    
    # Plot the reduced data, color-coding by the related variable
    color_a = sns.color_palette("rainbow", 144)
    color_b = sns.color_palette("rainbow", 7)
    for i in range(PC1.shape[0]-1):
        if lap_ids[i] != lap_ids[i+1]:
            continue

        axes[0].plot(
            PC1[i:i+2], 
                    PC2[i:i+2], 
                    color=color_a[int(pos_traj_reord[i])-1], 
                    linewidth=0.5,
                    alpha=1
        )
        
        axes[1].plot(
            PC1[i:i+2], 
            PC2[i:i+2], 
            color=color_b[int(route_ids[i])], 
            linewidth=0.5,
            alpha=1
        )
            
    mkdir(os.path.join(trace['p'], 'neural_traj', f"UMAP_DIM{n_components}"))
    plt.savefig(os.path.join(
        trace['p'], 'neural_traj', f"UMAP_DIM{n_components}", 'UMAP'+str(UMAPi+1)+str(UMAPj+1)+'.png'
    ))
    axes[0].clear()
    axes[1].clear()
        
    plt.close()
            
if __name__ == '__main__':
    import pickle
    from mylib.maze_utils3 import spike_nodes_transform
    with open(r"E:\Data\Dsp_maze\10224\20231012\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    for dim in np.arange(2, 26):
        for i in range(dim-1):
            for j in range(i+1, dim):
                print(dim, i, j)
                pca_dim_reduction(trace, n_components=dim, UMAPi=i, UMAPj=j)