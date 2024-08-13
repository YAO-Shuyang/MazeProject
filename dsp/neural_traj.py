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
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from mylib.maze_utils3 import spike_nodes_transform, mkdir
import seaborn as sns
from mylib.maze_graph import NRG, CP_DSP, DSP_NRG
from mylib.maze_graph import Father2SonGraph as F2S
import umap.umap_ as umap
import os
import scipy
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
        
        kernel = GaussianKernel1d(n = 20, sigma=0.5)
        neuro_traj = neuro_traj.smooth(kernel)
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
    breakpoints = np.array([6, 23, 94, 66, 97, 99, 144])
    prev_bps = np.array([144, 5, 24, 95, 54, 85, 100])
    next_bps = np.array([18, 22, 82, 65, 98, 87, 1])

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
        
        if pos_traj[i] in F2S[breakpoints[seg]] or pos_traj[i] in F2S[next_bps[seg]]:
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

def preprocess_neural_traj(
    trace,
    event_num : int = 10,
    is_ego: bool = False,
    segment: None | int | str = 'all',
    train_label: str = "route"
):
    if is_ego == 'ego_pos_traj':
        is_ego = True
    
    try:
        neural_traj = cp.deepcopy(trace['neural_traj'])
        pos_traj = cp.deepcopy(trace['pos_traj'])
        lap_ids = cp.deepcopy(trace['traj_lap_ids'])
        route_ids = cp.deepcopy(trace['traj_route_ids'])
    except:
        trace = get_neural_trajectory(trace)
        neural_traj = cp.deepcopy(trace['neural_traj'])
        pos_traj = cp.deepcopy(trace['pos_traj'])
        lap_ids = cp.deepcopy(trace['traj_lap_ids'])
        route_ids = cp.deepcopy(trace['traj_route_ids'])
        
    try: 
        segment_traj = cp.deepcopy(trace['traj_segment_ids'])
    except:
        trace = segmented_neural_trajectory(trace)
        segment_traj = trace['traj_segment_ids']
        
    if is_ego:
        ego_pos_traj = trace['ego_pos_traj']
    else:
        ego_pos_traj = None
        
    if segment != 'all':
        if isinstance(segment, int):
            if segment not in [0, 1, 2, 3, 4, 5, 6]:
                raise ValueError(
                    f"Parameter segment should be 'all' or an integer from 0 to 6, "
                    f"but got {segment}."
                )
            idx = np.where(segment_traj == segment)[0]
            neural_traj = neural_traj[:, idx]
            pos_traj = pos_traj[idx]
            lap_ids = lap_ids[idx]
            route_ids = route_ids[idx]
            segment_traj = segment_traj[idx]
            
            if is_ego:
                ego_pos_traj = ego_pos_traj[idx]
            
        elif isinstance(segment, list) or isinstance(segment, np.ndarray):
            if np.asarray(segment).ndim != 1:
                idx = np.where(segment_traj == segment[0])[0]
            else:
                idx = np.where(np.isin(segment_traj, segment))[0]
                
            neural_traj = neural_traj[:, idx]
            pos_traj = pos_traj[idx]
            lap_ids = lap_ids[idx]
            route_ids = route_ids[idx]
            segment_traj = segment_traj[idx]
            
            if is_ego:
                ego_pos_traj = ego_pos_traj[idx]
            
        else:
            raise ValueError(
                f"Parameter segment should be 'all' or an integer from 0 to 6, "
                f"but got {segment}."
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
    neural_traj = neural_traj[:, idx]
    segment_traj = segment_traj[idx]
    
    sum_events_n = np.sum(trace['Spikes'], axis=1)
    # Extract Place cells for analysis only
    if 'pc_idx' not in trace.keys():
        pc_idx = np.unique(
            np.concatenate(
                [np.where(trace[f'node {i}']['is_placecell'] == 1)[0] for i in range(10)]
            )
        )
    else:
        pc_idx = trace['pc_idx']
        
    remain_idx = np.intersect1d(pc_idx, np.where(sum_events_n >= event_num)[0])
    neural_traj = neural_traj[remain_idx, :]
        
    # Convert to graph
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj, np.int64)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]
    
    if decode_type == 'lap':
        idx = np.where(route_ids == 0)[0]
        neural_traj = neural_traj[:, idx]
        pos_traj = pos_traj[idx]
        lap_ids = lap_ids[idx]
        route_ids = route_ids[idx]
        segment_traj = segment_traj[idx]
        
        if is_ego:
            ego_pos_traj = ego_pos_traj[idx]
    
    return {
        'neural_traj': neural_traj,
        'pos_traj': pos_traj,
        'pos_traj_reord': pos_traj_reord,
        'traj_lap_ids': lap_ids,
        'traj_route_ids': route_ids,
        'traj_segment_ids': segment_traj,
        'ego_pos_traj': ego_pos_traj,
        'remain_idx': remain_idx
    }

def calc_trajectory_similarity(
    reduced_data: np.ndarray,
    lap_ids: np.ndarray,
    route_ids: np.ndarray,
    dim: int = 3
):
    assert reduced_data.shape[1] >= dim
    reduced_data = reduced_data[:, :dim].T
    
    idx = np.where(np.ediff1d(lap_ids) != 0)[0]
    beg, end = np.concatenate([[0], idx+1]), np.concatenate([idx+1, [lap_ids.shape[0]]])
    
    dist_mat = np.zeros((beg.shape[0], beg.shape[0])) * np.nan
    
    for i in tqdm(range(beg.shape[0]-1)):
        for j in range(i+1, beg.shape[0]):
            D = np.linalg.norm(reduced_data[:, beg[i]:end[i], np.newaxis] - reduced_data[:, np.newaxis, beg[j]:end[j]], axis=0)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(D)
            dist_mat[i, j] = dist_mat[j, i] = np.nanmean(D[row_ind, col_ind])
            if np.isnan(dist_mat[i, j]):
                print(i, j, " NAN")
    return dist_mat, lap_ids[beg], route_ids[beg]

def umap_dim_reduction(neural_traj: np.ndarray, n_components:int=20, **parakwargs):
    umap_model = umap.UMAP(n_components=n_components, **parakwargs)  
    reduced_data = umap_model.fit_transform(neural_traj.T)
    
    return reduced_data, umap_model

def pca_dim_reduction(neural_traj: np.ndarray, n_components=20, **parakwargs):
    pca = PCA(n_components=n_components, **parakwargs)  
    reduced_data = pca.fit_transform(neural_traj.T)
    return reduced_data, pca

def fa_dim_reduction(neural_traj: np.ndarray, n_components=20, **parakwargs):
    fa = FA(n_components=n_components, **parakwargs)  
    reduced_data = fa.fit_transform(neural_traj.T)
    return reduced_data, fa

def lda_dim_reduction(
    neural_traj: np.ndarray, 
    traj_labels: np.ndarray, 
    n_components=20, 
    model=None,
    **parakwargs
):
    if model is None:
        lda = LDA(n_components=n_components, **parakwargs)  # Reduce to 2 dimensions for visualization
        lda.fit(neural_traj.T, traj_labels)
        reduced_data = lda.transform(neural_traj.T)
    else:
        lda = model
        reduced_data = lda.transform(neural_traj.T)

    return reduced_data, lda
    
def Isomap_dim_reduction(neural_traj: np.ndarray, n_components=20, n_neighbors=5, **parakwargs):
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors, **parakwargs)
    reduced_data = isomap.fit_transform(neural_traj.T)
    return reduced_data, isomap

def visualize_neurotraj(
    trace: dict,
    n_components=20,
    component_i: int = 0,
    component_j: int = 1,
    method: str = "UMAP",
    segment: None | int | str = 'all',
    is_show: bool = False,
    palette: str = 'default',
    save_dir: str = None,
    figsize: tuple = (4*2, 6),
    train_label: str = "route",
    pos_type: str = "pos_traj",
    pca_n: int = 30,
    **parakwargs
):
    if method not in ["UMAP", "PCA", "FA", "LDA", "Isomap"]:
        raise ValueError("method should be one of ['UMAP', 'PCA', 'FA', 'LDA', 'Isomap']")
    
    if train_label not in ['route', 'lap', 'pos']:
        raise ValueError(f"train_label should be one of ['route', 'lap', 'pos'], but {train_label} is given.")
    
    print(f"Dimensional reduction with {method} - {n_components} components.")
    
    res = preprocess_neural_traj(
        trace, 
        segment=segment, 
        train_label=train_label, 
        is_ego=pos_type
    )
    
    neural_traj = res['neural_traj']
    lap_ids = res['traj_lap_ids']
    route_ids = res['traj_route_ids']
    segment_traj = res['traj_segment_ids']
    pos_traj = res['pos_traj']
    pos_traj_reord = res['pos_traj_reord']
    ego_pos_traj = res['ego_pos_traj']
    
    if pos_type == 'ego_pos_traj':
        pos_traj = ego_pos_traj    
        
    if method == "UMAP":
        reduced_data, model = umap_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "PCA":
        reduced_data, model = pca_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "FA":
        reduced_data, model = fa_dim_reduction(neural_traj, n_components, **parakwargs)
    elif method == "LDA":
        reduced_data, pca = pca_dim_reduction(neural_traj, pca_n)
        
        if train_label == 'lap':
            reduced_data, lda = lda_dim_reduction(reduced_data.T, lap_ids, n_components, **parakwargs)
        elif train_label == 'route':
            reduced_data, lda = lda_dim_reduction(reduced_data.T, route_ids, n_components, **parakwargs)
        elif train_label == 'pos':
            reduced_data, lda = lda_dim_reduction(reduced_data.T, pos_traj_reord, n_components, **parakwargs)
            
        model = (pca, lda)
    elif method == "Isomap":
        reduced_data, model = Isomap_dim_reduction(neural_traj, n_components, **parakwargs)
    
    print(
        f"Visualizing neural trajectory - component {component_i}"
        f" and {component_j}.", 
        end='\n\n'
    )
        
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize)
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


def visualize_neurotraj3d(
    trace: dict,
    n_components=20,
    component_i: int = 0,
    component_j: int = 1,
    component_k: int = 2,
    method: str = "UMAP",
    segment: None | int | str = 'all',
    is_show: bool = False,
    palette: str = 'default',
    save_dir: str = None,
    figsize: tuple = (4*3, 6),
    reduced_data: np.ndarray = None,
    train_label: str = 'route',
    elev=30,
    azim=45,
    pos_type: str = 'pos_traj',
    **parakwargs
):
    if n_components < 3:
        raise ValueError(f"n_components for 3d plots should be at least 3.")
    
    if train_label not in ['route', 'lap', 'pos']:
        raise ValueError(f"train_label should be one of ['route', 'lap', 'pos'], but {train_label} is given.")
    
    if component_i == component_j or component_i == component_k or component_j == component_k:
        raise ValueError(
            f"component_i, component_j, component_k should be different, but"
            f" they are {component_i}, {component_j}, {component_k}."
        )
    
    if method not in ["UMAP", "PCA", "FA", "LDA", "Isomap"]:
        raise ValueError("method should be one of ['UMAP', 'PCA', 'FA', 'LDA', 'Isomap']")
    
    print(f"Dimensional reduction with {method} - {n_components} components.")
    
    res = preprocess_neural_traj(
        trace, 
        segment=segment, 
        train_label=train_label, 
        is_ego=pos_type
    )
    
    neural_traj = res['neural_traj']
    lap_ids = res['traj_lap_ids']
    route_ids = res['traj_route_ids']
    segment_traj = res['traj_segment_ids']
    pos_traj = res['pos_traj']
    pos_traj_reord = res['pos_traj_reord']
    ego_pos_traj = res['ego_pos_traj']
    
    if pos_type == 'ego_pos_traj':
        pos_traj = ego_pos_traj
        
    
    if reduced_data is None:
        if method == "UMAP":
            reduced_data, model = umap_dim_reduction(neural_traj, n_components, **parakwargs)
        elif method == "PCA":
            reduced_data, model = pca_dim_reduction(neural_traj, n_components, **parakwargs)
        elif method == "FA":
            reduced_data, model = fa_dim_reduction(neural_traj, n_components, **parakwargs)
        elif method == "LDA":
            reduced_data, pca = pca_dim_reduction(neural_traj, pca_n)
        
            if train_label == 'lap':
                reduced_data, lda = lda_dim_reduction(reduced_data.T, lap_ids, n_components, **parakwargs)
            elif train_label == 'route':
                reduced_data, lda = lda_dim_reduction(reduced_data.T, route_ids, n_components, **parakwargs)
            elif train_label == 'pos':
                reduced_data, lda = lda_dim_reduction(reduced_data.T, pos_traj_reord, n_components, **parakwargs)
            
            model = (pca, lda)
        elif method == "Isomap":
            reduced_data, model = Isomap_dim_reduction(neural_traj, n_components, **parakwargs)
    else:
        model = None
    
    
    print(
        f"Visualizing neural trajectory - component {component_i}, {component_j}, {component_k}.", 
        end='\n\n'
    )
        
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=figsize, subplot_kw={'projection': '3d'})
    reduced_data_temp = reduced_data[:, [component_i, component_j, component_k]]
    
    PC1, PC2, PC3 = reduced_data_temp[:, 0], reduced_data_temp[:, 1], reduced_data_temp[:, 2] # df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

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
    vmin, vmax = np.min(pos_traj_reord), np.max(pos_traj_reord)
    color_a = np.array(sns.color_palette("rainbow", vmax-vmin+1))
    color_b = sns.color_palette(palette, 7) if palette != 'default' else DSPPalette
    color_c = np.array(sns.color_palette("rainbow", beg.shape[0]))
        
    axes[0].scatter( 
        PC1, 
        PC2, 
        PC3,
        color=color_a[pos_traj_reord.astype(np.int64) - vmin],
        alpha=0.5,
        s=3,
        linewidth = 0
    )
                     
    for i in range(beg.shape[0]):
        lef, rig = beg[i], end[i]
        
        axes[1].plot(
            PC1[lef:rig], 
            PC2[lef:rig], 
            PC3[lef:rig],
            color=color_b[int(route_ids[lef])], 
            linewidth=0.5
        )
        axes[2].plot(
            PC1[lef:rig], 
            PC2[lef:rig], 
            PC3[lef:rig],
            color=color_c[i], 
            linewidth=0.5
        )
        
    for i in range(PC1.shape[0]-1):
        if lap_ids[i] != lap_ids[i+1]:
            axes[0].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], '^', markersize=5, 
                         markeredgewidth = 0, color='k',
                         alpha=0.5)
            axes[1].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], '^', markersize=5, 
                         markeredgewidth = 0, color='k',
                         alpha=0.5) 
            axes[2].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], '^', markersize=5, 
                         markeredgewidth = 0, color='k',
                         alpha=0.5) 
            continue
        
        if lap_ids[i] != lap_ids[i-1] or i == 0:
            axes[0].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], 'o', markersize=5, 
                         markeredgewidth = 0.5, markeredgecolor = 'k', color=color_a[int(pos_traj_reord[i])-vmin],
                         alpha=0.5)
            axes[1].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], 'o', markersize=5, 
                         markeredgewidth = 0.5, markeredgecolor='k', color=color_b[int(route_ids[i])],
                         alpha=0.5) 
            axes[1].plot(PC1[i:i+1], PC2[i:i+1], PC3[i:i+1], 'o', markersize=5, 
                         markeredgewidth = 0.5, markeredgecolor='k', color=color_b[int(route_ids[i])],
                         alpha=0.5) 
    
    axes[0].view_init(elev=elev, azim=azim)
    axes[1].view_init(elev=elev, azim=azim)
    axes[2].view_init(elev=elev, azim=azim)
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_zlabel('PC3')
    
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_zlabel('PC3')
    
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].set_zlabel('PC3')
    
    if is_show:
        plt.show()
    else:
        if save_dir == None:
            save_dir = os.path.join(trace['p'], 'neural_traj', f"{method}_DIM{n_components}")
        mkdir(save_dir)
        
        plt.savefig(os.path.join(
            save_dir, f'{method}{component_i}_{component_j}_{component_k}.png'
        ), dpi=2400)
        plt.savefig(os.path.join(
            save_dir, f'{method}{component_i}&{component_j}_{component_k}.svg'
        ), dpi=2400)
    
        axes[0].clear()
        axes[1].clear()
        plt.close()
    
    return {
        'reduced_data': reduced_data,
        'traj_lap_ids': lap_ids,
        pos_type: pos_traj,
        'pos_traj_reord': pos_traj_reord,
        'traj_route_ids': route_ids,
        'traj_segment_ids': segment_traj,
        'remain_idx': res['remain_idx'],
        'model': model
    }

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
        
        trace = get_neural_trajectory(trace)
        trace = segmented_neural_trajectory(trace)
        
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