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
from mylib.maze_graph import NRG, CP_DSP
import umap.umap_ as umap
import os

DSPPalette = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9', '#F2E2C5', '#647D91', '#C06C84']

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

def umap_dim_reduction(trace: dict, n_components=20):
    if 'neural_traj' not in trace.keys():
        get_neural_trajectory(trace)
        
    neural_traj = trace['neural_traj']
            
    pca = umap.UMAP(n_components=n_components)  # Reduce to 2 dimensions for visualization
    #reduced_data = pca.fit_transform(neural_traj[:, idx].T)
    reduced_data = pca.fit_transform(neural_traj.T)
    
    trace['reduced_data'] = reduced_data
    trace['UMAPn'] = n_components
    return trace

def pca_dim_reduction(trace: dict, n_components=20):
    if 'neural_traj' not in trace.keys():
        get_neural_trajectory(trace)
        
    neural_traj = trace['neural_traj']
        
    pca = PCA(n_components=n_components)  # Reduce to 2 dimensions for visualization
    #reduced_data = pca.fit_transform(neural_traj[:, idx].T)
    reduced_data = pca.fit_transform(neural_traj.T)
    
    trace['reduced_data_pca'] = reduced_data
    trace['PCAn'] = n_components
    return trace

def fa_dim_reduction(trace: dict, n_components=20):
    if 'neural_traj' not in trace.keys():
        get_neural_trajectory(trace)
        
    neural_traj = trace['neural_traj']
        
    fa = FA(n_components=n_components)  # Reduce to 2 dimensions for visualization
    #reduced_data = pca.fit_transform(neural_traj[:, idx].T)
    reduced_data = fa.fit_transform(neural_traj.T)
    
    trace['reduced_data_fa'] = reduced_data
    trace['FAn'] = n_components
    return trace

def visualize_neurotraj(
    trace: dict,
    n_components=20,
    component_i: int = 0,
    component_j: int = 1,
    method: str = "UMAP",
    is_show: bool = False,
    palette: str = 'default'
):
    if method not in ["UMAP", "PCA", "FA"]:
        raise ValueError("method should be one of ['UMAP', 'PCA', 'FA']")
    
    print(f"Dimensional reduction with {method} - {n_components} components.")
    
    if method == "UMAP":
        trace = umap_dim_reduction(trace, n_components)
        reduced_data = trace['reduced_data']
        n_components = trace['UMAPn']
    elif method == "PCA":
        trace = pca_dim_reduction(trace, n_components)
        reduced_data = trace['reduced_data_pca']
        n_components = trace['PCAn']
    elif method == "FA":
        trace = fa_dim_reduction(trace, n_components)
        reduced_data = trace['reduced_data_fa']
        n_components = trace['FAn']
            
    lap_ids = trace['traj_lap_ids']
    route_ids = trace['traj_route_ids']
    pos_traj = trace['pos_traj']
              
    print(
        f"Visualizing neural trajectory - component {component_i}"
        f" and {component_j}.", 
        end='\n\n'
    )
    
    # Convert to major bin
    pos_traj = spike_nodes_transform(pos_traj, 12).astype(np.float64)
    
    # Set bin at the incorrect track as NAN
    for i in range(pos_traj.shape[0]):
        if pos_traj[i] not in CP_DSP[route_ids[i]]:
            pos_traj[i] = np.nan
    
    # Delete NAN 
    idx = np.where(np.isnan(pos_traj) == False)[0]
    pos_traj = pos_traj[idx].astype(np.int64)
    lap_ids = lap_ids[idx]
    route_ids = route_ids[idx]
    reduced_data = reduced_data[idx, :]
    
    # Convert to graph
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]
        
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6*2, 4))
    reduced_data_temp = reduced_data[:, [component_i, component_j]]
    
    PC1, PC2 = reduced_data_temp[:, 0], reduced_data_temp[:, 1] # df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    idx = np.where(np.ediff1d(lap_ids) != 0)[0]
    beg = np.concatenate(
        [[0], idx]
    )
    end = np.concatenate(
        [idx+1, [lap_ids.shape[0]]]
    )
    linewidths = np.full_like(lap_ids, 0.1).astype(np.float64)
    for i in range(beg.shape[0]):
        linewidths[beg[i]:end[i]] = np.linspace(0.1, 1, end[i]-beg[i])

    # Plot the reduced data, color-coding by the related variable
    color_a = sns.color_palette("rainbow", 144)
    color_b = sns.color_palette(palette, 7) if palette != 'default' else DSPPalette
    
    for i in range(PC1.shape[0]-1):
        if lap_ids[i] != lap_ids[i+1]:
            continue

        axes[0].plot(
            PC1[i:i+2], 
            PC2[i:i+2], 
            color=color_a[int(pos_traj_reord[i])-1], 
            linewidth=linewidths[i],
            alpha=0.5
        )
        
        axes[1].plot(
            PC1[i:i+2], 
            PC2[i:i+2], 
            color=color_b[int(route_ids[i])], 
            linewidth=linewidths[i],
            alpha=0.5
        )
    
    if is_show:
        plt.show()
    else:
        mkdir(os.path.join(trace['p'], 'neural_traj', f"UMAP_DIM{n_components}"))
        plt.savefig(os.path.join(
            trace['p'], 'neural_traj', f"UMAP_DIM{n_components}", 'UMAP'+str(component_i+1)+str(component_j+1)+'.png'
        ))
    
        axes[0].clear()
        axes[1].clear()
        plt.close()

            
if __name__ == '__main__':
    import pickle
    from mylib.maze_utils3 import spike_nodes_transform
    with open(r"E:\Data\Dsp_maze\10224\20231012\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    """
    for dim in np.arange(2, 26):
        for i in range(dim-1):
            for j in range(i+1, dim):
                print(dim, i, j)
                pca_dim_reduction(trace, n_components=dim, UMAPi=i, component_j=j)
                
    """
    visualize_neurotraj(
        trace, 
        n_components=15,
        component_i=0,
        component_j=1,
        method="UMAP",
        is_show=True,
        palette='rainbow'
    )