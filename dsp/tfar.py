# Track Field Across Route.

from mylib.field.tracker import Tracker, TrackerDsp, get_field_ids
from mylib.maze_graph import S2F, CP_DSP, NRG, Father2SonGraph
from mylib.maze_utils3 import mkdir, Clear_Axes, spike_nodes_transform
from mylib.calcium.axes.loc_time_curve import LocTimeCurveAxes
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import copy as cp
from scipy.stats import pearsonr

DSPPalette = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9', '#F2E2C5', '#647D91', '#C06C84']
seg1 = np.array([1,13,14,26,27,15,3,4,5])
seg2 = np.array([6,18,17,29,30,31,19,20,21,9,10,11,12,24])
seg3 = np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95])
seg4 = np.array([94,82,81,80,92,104,103,91,90,78,79,67,55,54])
seg5 = np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97])
seg6 = np.array([109,110,122,123,111,112,100])
seg7 = np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144])

def field_register_dsp_old(trace, overlap_thre: float, maze_type: int = 1, is_shuffle: bool = False, corr_thre: float = 0.3):
    """
    This method was similar to Chen, Yao et al., 2024
    
    Using overlaping of place fields to track and match place fields across routes.
    
    Now it is deprecated. 9/8/2024
    """
    corrs = np.zeros((trace['n_neuron'], 2))

    for i in range(trace['n_neuron']):
        corrs[i, 0] = pearsonr(
            trace['node 0']['smooth_map_all'][i, :],
            trace['node 4']['smooth_map_all'][i, :]
        )[0]
        corrs[i, 1] = pearsonr(
            trace['node 5']['smooth_map_all'][i, :],
            trace['node 9']['smooth_map_all'][i, :]
        )[0]

        
    qualified_cell = np.where(
        ((corrs[:, 0] >= corr_thre)&(trace['node 0']['is_placecell'] == 1)&(trace['node 4']['is_placecell'] == 1)) | 
        ((corrs[:, 1] >= corr_thre)&(trace['node 5']['is_placecell'] == 1)&(trace['node 9']['is_placecell'] == 1))
    )[0]
    
    n_neuron = trace['n_neuron']
    index_map = np.meshgrid(np.arange(1, n_neuron+1), np.arange(10))[0]
    place_field_all = [[trace[f'node {i}']['place_field_all'][k] for i in range(10)] for k in range(n_neuron)]
    is_placecell = np.vstack([trace[f'node {i}']['is_placecell'] for i in range(10)])

    smooth_map_all = np.full((10, n_neuron, 2304), np.nan)
    for i in range(10):
        smooth_map_all[i, :, :] = trace[f'node {i}']['smooth_map_all']
    
    print(f"Registering {n_neuron} neurons")
    field_reg, field_info, place_field_all = Tracker.field_register(
        index_map=index_map,
        place_field_all=place_field_all,
        is_placecell=is_placecell,
        overlap_thre=overlap_thre,
        maze_type=maze_type,
        smooth_map_all=smooth_map_all,
        is_shuffle = is_shuffle
    )
    field_ids = get_field_ids(field_info)
    
    qualified_field = np.isin(field_info[0, :, 0].astype(np.int64), qualified_cell+1)
    
    trace['field_reg'] = field_reg[:, qualified_field]
    trace['field_info'] = field_info[:, qualified_field, :]
    trace['field_ids'] = field_ids[qualified_field]
    trace['place_field_all_multiroute'] = place_field_all
    trace['qualified_cell'] = qualified_cell
    return trace


def field_register_dsp(trace, corr_thre: float = 0.3):
    """
    This method was similar to Chen, Yao et al., 2024
    
    Using overlaping of place fields to track and match place fields across routes.
    
    Now it is deprecated. 9/8/2024
    """
    corrs = np.zeros((trace['n_neuron'], 2))

    for i in range(trace['n_neuron']):
        corrs[i, 0] = pearsonr(
            trace['node 0']['smooth_map_all'][i, :],
            trace['node 4']['smooth_map_all'][i, :]
        )[0]
        corrs[i, 1] = pearsonr(
            trace['node 5']['smooth_map_all'][i, :],
            trace['node 9']['smooth_map_all'][i, :]
        )[0]

        
    qualified_cell = np.where(
        ((corrs[:, 0] >= corr_thre)&(trace['node 0']['is_placecell'] == 1)&(trace['node 4']['is_placecell'] == 1)) | 
        ((corrs[:, 1] >= corr_thre)&(trace['node 5']['is_placecell'] == 1)&(trace['node 9']['is_placecell'] == 1))
    )[0]
    
    n_neuron = trace['n_neuron']
    
    print(f"Registering {n_neuron} neurons")
    field_reg, field_info = TrackerDsp.field_register(trace, qualified_cells=qualified_cell)
    
    trace['field_reg'] = field_reg
    trace['field_info'] = field_info
    trace['qualified_cell'] = qualified_cell
    
    segment_bins = [
        np.concatenate([Father2SonGraph[i] for i in seg1]),
        np.concatenate([Father2SonGraph[i] for i in seg2]),
        np.concatenate([Father2SonGraph[i] for i in seg3]),
        np.concatenate([Father2SonGraph[i] for i in seg4]),
        np.concatenate([Father2SonGraph[i] for i in seg5]), 
        np.concatenate([Father2SonGraph[i] for i in seg6]),
        np.concatenate([Father2SonGraph[i] for i in seg7])
    ]
    
    field_segs = np.zeros(field_reg.shape[1])
    field_centers = field_info[0, :, 2].astype(np.int64)
    
    for i in range(7):
        field_segs[np.isin(field_centers, segment_bins[i])] = i+1
        
    trace['field_segs'] = field_segs
        
    return trace

def _get_range_dsp(
    field_area: np.ndarray
):
    bins = np.unique(S2F[field_area-1])
    transformed_bin = np.zeros_like(bins)
    for i in range(len(bins)):
        transformed_bin[i] = NRG[1][bins[i]]
    return transformed_bin


def proofread(trace, min_reactivate_num: int = 5, min_spike_num: int = 5):
    field_reg = trace['field_reg']
    field_info = trace['field_info']
    
    for i in range(field_reg.shape[1]):
        for j in range(field_reg.shape[0]):
            # Check if the field satisfies the criteria.
            cell = int(field_info[j, i, 0])
            field_center = int(field_info[j, i, 2])
            field_area = trace['place_field_all'][cell-1][field_center]
                
            idx = np.where((np.isin(
                trace[f'node {j}']['spike_nodes'], field_area
            )) & (trace[f'node {j}']['Spikes'][cell-1, :] == 1))[0]
                
            spike_num = idx.shape[0]
            dt = np.ediff1d(trace[f'node {j}']['ms_time_behav'][idx])
            n_react = np.where(dt >= 10000)[0].shape[0]
            
            if field_reg[j, i] == 1:    
                # Too little spikes
                if spike_num < min_spike_num or n_react < min_reactivate_num-1:
                    field_reg[j, i] = 0
                    
            elif field_reg[j, i] == 0:
                # Check if whether it is a weakened field or it is genuinely not active.
                if spike_num >= min_spike_num and n_react >= min_reactivate_num-1:
                    field_reg[j, i] = 2
    
    trace['field_reg_modi'] = field_reg
    
    return trace
                
    
def LocTimeCurve_with_Field(trace):
    save_loc = os.path.join(trace['p'], "LocTimeCurve with Field")
    mkdir(save_loc)
    
    fig = plt.figure(figsize=(3, 4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)    

    for cell in tqdm(trace['qualified_cell']):
        colors = [
            DSPPalette[0], DSPPalette[1], DSPPalette[2], DSPPalette[3], DSPPalette[0],
            DSPPalette[0], DSPPalette[4], DSPPalette[5], DSPPalette[6], DSPPalette[0]]
        
        for i in range(10):
            ax = LocTimeCurveAxes(
                ax,
                behav_time=trace[f'node {i}']['ms_time_behav'],
                behav_nodes=spike_nodes_transform(trace[f'node {i}']['spike_nodes'], 12),
                spikes=trace[f'node {i}']['Spikes'][cell, :],
                spike_time=trace[f'node {i}']['ms_time_behav'],
                maze_type=1,
                line_kwargs={'linewidth': 0.4, 'color': colors[i]},
                bar_kwargs={'markeredgewidth': 0.5, 'markersize': 2, 'color': 'k'},
                is_include_incorrect_paths=True
            )[0]
            
        idx = np.where(trace['field_info'][0, :, 0] == cell+1)[0]
        field_shadow_colors = sns.color_palette("Spectral", idx.shape[0])
        
        for i in range(idx.shape[0]):
            k = int(trace['field_info'][0, idx[i], 2])
            field_area = trace['place_field_all'][cell][k]
            transformed_bins = _get_range_dsp(field_area)
            for j in range(10):
                if trace['field_reg'][j, idx[i]] >= 1:
                    t1 = trace[f'node {j}']['ms_time_behav'][0]/1000
                    t2 = trace[f'node {j}']['ms_time_behav'][-1]/1000
                
                    for d in range(len(transformed_bins)):
                        ax.fill_betweenx([t1, t2], transformed_bins[d]-0.5, transformed_bins[d]+0.5, color=field_shadow_colors[i], alpha=0.5, edgecolor = None)
            
        t1 = trace['node 4']['ms_time_behav'][-1]/1000
        t2 = trace['node 5']['ms_time_behav'][0]/1000
        t3 = trace['node 9']['ms_time_behav'][-1]/1000
        ax.set_yticks([0, t1, t2, t3], [0, t1, 0, t3-t2])
        
        plt.savefig(os.path.join(save_loc, f'{cell+1}.png'), dpi=600)
        plt.savefig(os.path.join(save_loc, f'{cell+1}.svg'), dpi=600)
        ax.clear()
    
    plt.close()

if __name__ == '__main__':
    from mylib.local_path import f2
    import pickle
    from mylib.calcium.field_criteria import place_field_dsp
    from mylib.maze_utils3 import SmoothMatrix
    
    for i in range(22, 28):
        print(i, f2['MiceID'][i], f2['date'][i])   
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        """
        trace['place_field_all'] = place_field_dsp(
            trace=trace,
            thre_type=2,
            parameter=0.5,
            events_num_crit=5,
            split_thre=0.6
        )
        """
        trace = field_register_dsp(trace, corr_thre=0.3)
        trace = proofread(trace, min_reactivate_num=4, min_spike_num=4)
        with open(f2['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
        
        #LocTimeCurve_with_Field(trace)
        