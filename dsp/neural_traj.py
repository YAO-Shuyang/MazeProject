from mazepy.datastruc.neuact import SpikeTrain, NeuralTrajectory
from mazepy.datastruc.variables import Variable1D
from mazepy.datastruc.kernel import GaussianKernel1d
import numpy as np
import copy as cp
# Perform PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from mylib.maze_utils3 import spike_nodes_transform, mkdir
import seaborn as sns
from mylib.maze_graph import NRG, CP_DSP
import umap.umap_ as umap
import os

DSPPalette = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9', '#F2E2C5', '#647D91', '#C06C84']

def get_neural_trajectory(trace: dict, is_normalize: bool = True):
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

def umap_dim_reduction(neural_traj: np.ndarray, n_components:int=20, **parakwargs):
    umap_model = umap.UMAP(n_components=n_components, **parakwargs)  
    reduced_data = umap_model.fit_transform(neural_traj.T)
    
    return reduced_data

def pca_dim_reduction(neural_traj: np.ndarray, n_components=20, **parakwargs):
    pca = PCA(n_components=n_components, **parakwargs)  
    reduced_data = pca.fit_transform(neural_traj.T)
    return reduced_data

def fa_dim_reduction(neural_traj: np.ndarray, n_components=20, **parakwargs):
    fa = FA(n_components=n_components, **parakwargs)  
    reduced_data = fa.fit_transform(neural_traj.T)
    return reduced_data

def lda_dim_reduction(
    neural_traj: np.ndarray, 
    traj_labels: np.ndarray, 
    n_components=20, 
    **parakwargs
):
    lda = LDA(n_components=n_components, **parakwargs)  # Reduce to 2 dimensions for visualization
    reduced_data = lda.fit_transform(neural_traj.T, traj_labels)
    return reduced_data

def visualize_neurotraj(
    trace: dict,
    n_components=20,
    component_i: int = 0,
    component_j: int = 1,
    method: str = "UMAP",
    is_show: bool = False,
    palette: str = 'default',
    **parakwargs
):
    if method not in ["UMAP", "PCA", "FA", "LDA"]:
        raise ValueError("method should be one of ['UMAP', 'PCA', 'FA', 'LDA']")
    
    print(f"Dimensional reduction with {method} - {n_components} components.")
    
    try:
        neural_traj = trace['neural_traj']
        pos_traj = trace['pos_traj']
        lap_ids = trace['traj_lap_ids']
        route_ids = trace['traj_route_ids']
    except:
        trace = get_neural_trajectory(trace)
        neural_traj = trace['neural_traj']
        pos_traj = trace['pos_traj']
        lap_ids = trace['traj_lap_ids']
        route_ids = trace['traj_route_ids']
    
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
    neural_traj = neural_traj[:, idx]
    
    # Extract Place cells for analysis only
    pc_idx = np.unique(
        np.concatenate(
            [np.where(trace[f'node {i}']['is_placecell'] == 1)[0] for i in range(10)]
        )
    )
    neural_traj = neural_traj[pc_idx, :]
    
    # Convert to graph
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]    
        
    if method == "UMAP":
        reduced_data = umap_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "PCA":
        reduced_data = pca_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "FA":
        reduced_data = fa_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "LDA":
        reduced_data = lda_dim_reduction(neural_traj, pos_traj_reord, n_components, **parakwargs)
        
    print(
        f"Visualizing neural trajectory - component {component_i}"
        f" and {component_j}.", 
        end='\n\n'
    )
    

        
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
        mkdir(os.path.join(trace['p'], 'neural_traj', f"{method}_DIM{n_components}"))
        plt.savefig(os.path.join(
            trace['p'], 'neural_traj', f"{method}_DIM{n_components}", f'{method}{component_i}&{component_j}.png'
        ))
    
        axes[0].clear()
        axes[1].clear()
        plt.close()

            
if __name__ == '__main__':
    import pickle
    from mylib.maze_utils3 import spike_nodes_transform
    loc = r"E:\Data\Dsp_maze\10224\20231012\trace.pkl"
    with open(loc, 'rb') as handle:
        trace = pickle.load(handle)

    """
    for dim in np.arange(2, 26):
        for i in range(dim-1):
            for j in range(i+1, dim):
                print(dim, i, j)
                pca_dim_reduction(trace, n_components=dim, UMAPi=i, component_j=j)
                
    """
    """
    for ndim in np.arange(2, 26):
        for i in range(ndim-1):
            for j in range(i+1, ndim):
                print(ndim, i, j)
                visualize_neurotraj(
                    trace, 
                    n_components=ndim, 
                    component_i=i,
                    component_j=j,
                    is_show=False,
                    palette='rainbow',
                    method="UMAP",
                    n_neighbors = 12, # 27: 12; 12: 12
                    min_dist = 0.05
                )
    """                
    visualize_neurotraj(
        trace, 
        n_components=20, 
        component_i=0,
        component_j=1,
        is_show=True,
        palette='rainbow',
        method="UMAP"
    )