from mazepy.datastruc.neuact import SpikeTrain, NeuralTrajectory
from mazepy.datastruc.variables import Variable1D
from mazepy.datastruc.kernel import GaussianKernel1d
import numpy as np
import copy as cp
from tqdm import tqdm
# Perform PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from mylib.maze_utils3 import spike_nodes_transform, mkdir
import seaborn as sns
from mylib.maze_graph import NRG, CP_DSP
from mylib.maze_graph import Father2SonGraph as F2S
import umap.umap_ as umap
import os
from mylib.maze_utils3 import Clear_Axes

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

def segmented_neural_trajectory(trace):
    breakpoints = np.array([6, 23, 94, 66, 109, 99, 144])
    prev_bps = np.array([144, 5, 24, 95, 54, 85, 100])

    neural_traj = cp.deepcopy(trace['neural_traj'])
    pos_traj = cp.deepcopy(trace['pos_traj'])
    print(pos_traj)
    lap_ids = cp.deepcopy(trace['traj_lap_ids'])
    route_ids = cp.deepcopy(trace['traj_route_ids'])
    segment_ids = np.zeros_like(route_ids)-2
    
    lap = -1
    is_move_backward = False
    for i in tqdm(range(len(lap_ids))):
        if lap_ids[i] != lap:
            if i != 0:
                # A non-first lap starts
                segment_ids[beg_idx:i] = seg
                
            # A new lap starts
            lap = lap_ids[i]
            beg_idx = i
            
            if route_ids[i] == 0:
                seg = 0
            elif route_ids[i] == 1:
                seg = 2
            elif route_ids[i] == 2:
                seg = 4
            elif route_ids[i] == 3:
                seg = 6
            elif route_ids[i] == 4:
                seg = 1
            elif route_ids[i] == 5:
                seg = 3
            else:
                seg = 5
        
        if seg >= 6 and pos_traj[i] not in F2S[prev_bps[seg]]:
            continue
        
        if pos_traj[i] in F2S[breakpoints[seg]]:
            if is_move_backward == False:
                segment_ids[beg_idx:i] = seg
                
            beg_idx = i
            seg += 1
            is_move_backward = False
            
        if pos_traj[i] in F2S[prev_bps[seg]] and is_move_backward == False:
            is_move_backward = True
            seg -= 1
    
    segment_ids[beg_idx:] = seg
    
    trace['traj_segment_ids'] = segment_ids
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
    is_return_model=False, 
    model=None,
    **parakwargs
):
    if model is None:
        lda = LDA(n_components=n_components, **parakwargs)  # Reduce to 2 dimensions for visualization
        reduced_data = lda.fit_transform(neural_traj.T, traj_labels)
    else:
        lda = model
        reduced_data = lda.transform(neural_traj.T)
    
    if is_return_model:
        return reduced_data, lda
    else:
        return reduced_data
    
def Isomap_dim_reduction(neural_traj: np.ndarray, n_components=20, n_neighbors=5, **parakwargs):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, **parakwargs)
    reduced_data = isomap.fit_transform(neural_traj.T)
    return reduced_data

def visualize_neurotraj(
    trace: dict,
    n_components=20,
    component_i: int = 0,
    component_j: int = 1,
    method: str = "UMAP",
    segments: None | int | str = 'all',
    is_show: bool = False,
    palette: str = 'default',
    save_dir: str = None,
    **parakwargs
):
    if method not in ["UMAP", "PCA", "FA", "LDA", "Isomap"]:
        raise ValueError("method should be one of ['UMAP', 'PCA', 'FA', 'LDA', 'Isomap']")
    
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
        
    try: 
        segment_traj = trace['traj_segment_ids']
    except:
        trace = segmented_neural_traj(trace)
    
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
    elif method == "Isomap":
        reduced_data = Isomap_dim_reduction(neural_traj, n_components, **parakwargs)
    
    
    
    print(
        f"Visualizing neural trajectory - component {component_i}"
        f" and {component_j}.", 
        end='\n\n'
    )
        
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(4*2, 6))
    reduced_data_temp = reduced_data[:, [component_i, component_j]]
    
    PC2, PC1 = reduced_data_temp[:, 0], reduced_data_temp[:, 1] # df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    idx = np.where(np.ediff1d(lap_ids) != 0)[0]
    beg = np.concatenate(
        [[0], idx+1]
    )
    end = np.concatenate(
        [idx+1, [lap_ids.shape[0]]]
    )
    linewidths = np.full_like(lap_ids, 0.1).astype(np.float64)
    for i in range(beg.shape[0]):
        linewidths[beg[i]:end[i]] = np.linspace(0.2, 2, end[i]-beg[i])

    # Plot the reduced data, color-coding by the related variable
    color_a = sns.color_palette("rainbow", 144)
    color_b = sns.color_palette(palette, 7) if palette != 'default' else DSPPalette
        
    sns.scatterplot( 
        x=PC1, 
        y=PC2, 
        hue=pos_traj_reord,
        palette = "rainbow",
        size=linewidths,
        sizes=(0.4, 3),
        alpha=0.5,
        ax=axes[0],
    )
                     
    for i in range(beg.shape[0]):
        lef, rig = beg[i], end[i]
        
        axes[1].plot(
            PC1[lef:rig], 
            PC2[lef:rig], 
            color=color_b[int(route_ids[lef])], 
            linewidth=0.5
        )
        
    for i in range(PC1.shape[0]-1):
        if lap_ids[i] != lap_ids[i+1]:
            axes[0].plot(PC1[i:i+1], PC2[i:i+1], '^', markersize=5, 
                         markeredgewidth = 0, color='k',
                         alpha=0.5)
            axes[1].plot(PC1[i:i+1], PC2[i:i+1], '^', markersize=5, 
                         markeredgewidth = 0, color='k',
                         alpha=0.5) 
            continue
        
        if lap_ids[i] != lap_ids[i-1] or i == 0:
            axes[0].plot(PC1[i:i+1], PC2[i:i+1], 'o', markersize=5, 
                         markeredgewidth = 0, color=color_a[int(pos_traj_reord[i])-1],
                         alpha=0.5)
            axes[1].plot(PC1[i:i+1], PC2[i:i+1], 'o', markersize=5, 
                         markeredgewidth = 0, color=color_b[int(route_ids[i])],
                         alpha=0.5) 
    
    if is_show:
        plt.show()
    else:
        if save_dir == None:
            save_dir = os.path.join(trace['p'], 'neural_traj', f"{method}_DIM{n_components}")
        mkdir(save_dir)
        
        plt.savefig(os.path.join(
            save_dir, f'{method}{component_i}&{component_j}.png'
        ), dpi=2400)
        plt.savefig(os.path.join(
            save_dir, f'{method}{component_i}&{component_j}.svg'
        ), dpi=2400)
    
        axes[0].clear()
        axes[1].clear()
        plt.close()
        
    return reduced_data




if __name__ == '__main__':
    import pickle
    from mylib.local_path import f2
    from mylib.maze_utils3 import spike_nodes_transform
    loc = r"E:\Data\Dsp_maze\10224\20231012\trace.pkl"
    with open(loc, 'rb') as handle:
        trace = pickle.load(handle)

    for i in range(len(f2)):
        print(f2['Trace File'][i])
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        trace = segmented_neural_traj(trace)
        with open(f2['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
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
                   
    visualize_neurotraj(
        trace, 
        n_components=20, 
        component_i=0,
        component_j=1,
        is_show=True,
        palette='rainbow',
        method="UMAP"
    )
    """ 
    """
    from mylib.maze_utils3 import DrawMazeProfile
        
    trace = segmented_neural_traj(trace)
    
    for i in range(7):
        idx = np.where(trace['traj_segment_ids'] == i)[0]

        pos = trace['pos_traj'][idx]
        route = trace['traj_route_ids'][idx]
        x, y = (pos - 1) // 48+np.random.rand(len(idx))-0.5, (pos - 1) % 48 + np.random.rand(len(idx))-0.5
        
        fig = plt.figure()    
        ax = Clear_Axes(plt.axes())
        DrawMazeProfile(1, axes=ax, nx=48, color='black')
        colors = sns.color_palette('rainbow', 7)
        for r in range(7):
            idx = np.where(route == r)[0]
            print(r, idx)
            ax.plot(y[idx], x[idx], 'o', markersize=4, markeredgewidth=0, alpha=0.5, color = colors[r])
        ax.axis([-0.8, 47.8, 47.8, -0.8])
        ax.set_aspect('equal')
        plt.show()
        
    """