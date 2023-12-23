import copy as cp
import os
import pickle
import time
import warnings
from os.path import exists, join

import h5py
import scipy.stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns
from mylib.behavior.behavevents import BehavEvents
from mylib.maze_graph import correct_paths, NRG
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile, clear_NAN, mkdir, SpikeNodes, SpikeType
from mylib.maze_utils3 import plot_trajactory, spike_nodes_transform, SmoothMatrix, occu_time_transform
from mylib.preprocessing_ms import coverage_curve, calc_speed, uniform_smooth_speed, calc_ratemap
from mylib.preprocessing_ms import plot_spike_monitor, calc_ms_speed
from mylib.preprocessing_ms import Delete_InterLapSpike, calc_SI
from mylib.divide_laps.lap_split import LapSplit
from mylib.calcium.axes.peak_curve import get_y_order
from scipy.io import loadmat
from tqdm import tqdm
from mylib import RateMapAxes, TraceMapAxes, PeakCurveAxes, LocTimeCurveAxes
from mylib.calcium.firing_rate import calc_rate_map_properties

def get_spike_frame_label(ms_time, spike_nodes, trace = None, behavior_paradigm = 'CrossMaze', split_args: dict = {}):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm = behavior_paradigm, **split_args) # Get Split TimeIndex Point
    lap = len(beg_idx) # Number of inter-laps
    # behav spike index
    frame_labels = np.array([], dtype=np.float64)

    beg0 = np.where(ms_time >= trace['correct_time'][beg_idx[0]])[0][0]
    frame_labels = np.concatenate([frame_labels, np.repeat(np.nan, beg0-0)])

    for k in range(lap):
        beg, end = np.where(ms_time >= trace['correct_time'][beg_idx[k]])[0][0], np.where(ms_time <= trace['correct_time'][end_idx[k]])[0][-1]
        labels = BehavEvents.get_frame_labels(spike_nodes[beg:end], trace['maze_type'])
        frame_labels = np.concatenate([frame_labels, labels])

        if k < lap - 1:
            frame_labels = np.concatenate([frame_labels, np.repeat(np.nan, np.where(ms_time >= trace['correct_time'][beg_idx[k+1]])[0][0] - end)]) 
        else:
            frame_labels = np.concatenate([frame_labels, np.repeat(np.nan, spike_nodes.shape[0]-end)])
        

    return frame_labels

def plot_split_trajectory_directional(trace: dict, behavior_paradigm: str = 'CrossMaze', split_args: dict = {}, **kwargs):
    beg_idx, end_idx = LapSplit(trace, behavior_paradigm=behavior_paradigm, **split_args)
    laps = len(beg_idx)
    save_loc = os.path.join(trace['p'], 'behav','laps_trajactory_directional')
    mkdir(save_loc)
    behav_time = trace['correct_time']

    for k in tqdm(range(laps)):
        frame_labels = BehavEvents.get_frame_labels(trace['correct_nodes'][beg_idx[k]:end_idx[k]+1], trace['maze_type'])
        loc_x, loc_y = trace['correct_pos'][beg_idx[k]:end_idx[k]+1, 0] / 20 - 0.5, trace['correct_pos'][beg_idx[k]:end_idx[k]+1, 1] / 20 - 0.5
        fig = plt.figure(figsize = (6,6))
        ax = Clear_Axes(plt.axes())
        ax.set_title('Frame: '+str(beg_idx[k])+' -> '+str(end_idx[k])+'\n'+'Time:  '+str(behav_time[beg_idx[k]]/1000)+' -> '+str(behav_time[end_idx[k]]/1000))
        DrawMazeProfile(maze_type=trace['maze_type'], nx = 48, color='black', axes=ax)
        ax.invert_yaxis()
        idx = np.where(frame_labels == -1)[0]
        ax.plot(loc_x[idx], loc_y[idx], 'o', color = 'limegreen', markeredgewidth = 0, markersize = 3)
        idx = np.where(frame_labels == 1)[0]
        ax.plot(loc_x[idx], loc_y[idx], 'o', color = 'red', markeredgewidth = 0, markersize = 3)

        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.png'), dpi=600)
        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.svg'), dpi=600)
        plt.close()

def RateMap(trace: dict) -> dict:
    n_neuron = trace['n_neuron']
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(6*3, 4.5))
    save_loc = join(trace['p'], 'RateMap')
    mkdir(save_loc)

    axes[0] = Clear_Axes(axes[0])
    axes[1] = Clear_Axes(axes[1])
    axes[2] = Clear_Axes(axes[2])

    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[2].invert_yaxis()

    DrawMazeProfile(axes=axes[0], maze_type=trace['maze_type'], nx = 48)
    DrawMazeProfile(axes=axes[1], maze_type=trace['maze_type'], nx = 48)
    DrawMazeProfile(axes=axes[2], maze_type=trace['maze_type'], nx = 48)
    axes[0].axis([-0.6,47.6,-0.6,47.6])
    axes[1].axis([-0.6,47.6,-0.6,47.6])
    axes[2].axis([-0.6,47.6,-0.6,47.6])

    for i in tqdm(range(n_neuron)):
        axes[0], im1, cbar1 = RateMapAxes(
            ax=axes[0], 
            content=trace['tot']['smooth_map_all'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='Together, SI='+str(round(trace['tot']['SI_all'][i], 3))
        )
        color = 'black' if trace['tot']['is_placecell'][i] == 0 else 'red'
        axes[0].set_title('Together, SI='+str(round(trace['tot']['SI_all'][i], 3)), color=color)

        axes[1], im2, cbar2 = RateMapAxes(
            ax=axes[1], 
            content=trace['cis']['smooth_map_all'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='Cis, SI='+str(round(trace['cis']['SI_all'][i], 3))
        )
        color = 'black' if trace['cis']['is_placecell'][i] == 0 else 'red'
        axes[1].set_title('Cis, SI='+str(round(trace['cis']['SI_all'][i], 3)), color=color)

        axes[2], im3, cbar3 = RateMapAxes(
            ax=axes[2], 
            content=trace['trs']['smooth_map_all'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='Trans, SI='+str(round(trace['trs']['SI_all'][i], 3))
        )
        color = 'black' if trace['trs']['is_placecell'][i] == 0 else 'red'
        axes[2].set_title('Trans, SI='+str(round(trace['trs']['SI_all'][i], 3)), color=color)
        
        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        im1.remove()
        im2.remove()
        im3.remove()
    
    return trace

def TraceMap(trace: dict) -> dict:
    n_neuron = trace['n_neuron']
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))
    save_loc = join(trace['p'], 'TraceMap')
    mkdir(save_loc)

    axes[0] = Clear_Axes(axes[0])
    axes[1] = Clear_Axes(axes[1])
    axes[2] = Clear_Axes(axes[2])

    DrawMazeProfile(axes=axes[0], maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=axes[1], maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=axes[2], maze_type=trace['maze_type'], nx=48,color='black')
    axes[0].axis([-0.6,47.6,-0.6,47.6])
    axes[1].axis([-0.6,47.6,-0.6,47.6])
    axes[2].axis([-0.6,47.6,-0.6,47.6])

    frame_labels = BehavEvents.get_frame_labels(trace['correct_nodes'], trace['maze_type'])
    tot_idx = np.where((frame_labels==1)|(frame_labels==-1))[0]
    trajectory_tot = trace['correct_pos'][tot_idx, :]
    behav_time_tot = trace['correct_time'][tot_idx]

    cis_idx = np.where(frame_labels==1)[0]
    trajectory_cis = trace['correct_pos'][cis_idx, :]
    behav_time_cis = trace['correct_time'][cis_idx]

    trs_idx = np.where(frame_labels==-1)[0]
    trajectory_trs = trace['correct_pos'][trs_idx, :]
    behav_time_trs = trace['correct_time'][trs_idx]

    for i in tqdm(range(n_neuron)):
        axes[0], a1, b1 = TraceMapAxes(
            ax=axes[0], 
            trajectory=cp.deepcopy(trajectory_tot), 
            behav_time=behav_time_tot, 
            spikes=trace['tot']['Spikes'][i], 
            spike_time=trace['tot']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="Entire trajectory[correct track]"
        )

        axes[1], a2, b2 = TraceMapAxes(
            ax=axes[1], 
            trajectory=cp.deepcopy(trajectory_cis), 
            behav_time=behav_time_cis, 
            spikes=trace['cis']['Spikes'][i], 
            spike_time=trace['cis']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="Cis trajectory[correct track]"
        )

        axes[2], a3, b3 = TraceMapAxes(
            ax=axes[2], 
            trajectory=cp.deepcopy(trajectory_trs), 
            behav_time=behav_time_trs, 
            spikes=trace['trs']['Spikes'][i], 
            spike_time=trace['trs']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="Trans trajectory[correct track]"
        )

        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        a = a1 + a2 + a3 + b1 + b2 + b3
        for j in a:
            j.remove()

    return trace

def OldMap(trace: dict, is_draw: bool = True) -> dict:
    maze_type = trace['maze_type']


    # total old map
    Spikes = trace['tot']['Spikes']
    spike_nodes = trace['tot']['spike_nodes']
    occu_time = trace['tot']['occu_time_spf'] if 'occu_time_spf' in trace['tot'].keys() else trace['tot']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['tot']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['tot']['old_map_all'] = old_map_all
    trace['tot']['old_map_clear'] = old_map_clear
    trace['tot']['old_map_smooth'] = old_map_smooth
    trace['tot']['old_map_nanPos'] = old_map_nanPos
    trace['tot']['old_nodes'] = old_nodes
    trace['tot']['occu_time_old'] = occu_time_old
    trace['tot']['old_t_total'] = old_t_total
    trace['tot']['old_t_nodes_frac'] = old_t_nodes_frac
    trace['tot']['old_SI'] = SI

    # cis old map
    Spikes = trace['cis']['Spikes']
    spike_nodes = trace['cis']['spike_nodes']
    occu_time = trace['cis']['occu_time_spf'] if 'occu_time_spf' in trace['cis'].keys() else trace['cis']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['cis']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['cis']['old_map_all'] = old_map_all
    trace['cis']['old_map_clear'] = old_map_clear
    trace['cis']['old_map_smooth'] = old_map_smooth
    trace['cis']['old_map_nanPos'] = old_map_nanPos
    trace['cis']['old_nodes'] = old_nodes
    trace['cis']['occu_time_old'] = occu_time_old
    trace['cis']['old_t_total'] = old_t_total
    trace['cis']['old_t_nodes_frac'] = old_t_nodes_frac
    trace['cis']['old_SI'] = SI

    # trs old map
    Spikes = trace['trs']['Spikes']
    spike_nodes = trace['trs']['spike_nodes']
    occu_time = trace['trs']['occu_time_spf'] if 'occu_time_spf' in trace['trs'].keys() else trace['trs']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['trs']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['trs']['old_map_all'] = old_map_all
    trace['trs']['old_map_clear'] = old_map_clear
    trace['trs']['old_map_smooth'] = old_map_smooth
    trace['trs']['old_map_nanPos'] = old_map_nanPos
    trace['trs']['old_nodes'] = old_nodes
    trace['trs']['occu_time_old'] = occu_time_old
    trace['trs']['old_t_total'] = old_t_total
    trace['trs']['old_t_nodes_frac'] = old_t_nodes_frac
    trace['trs']['old_SI'] = SI

    if not is_draw:
        return trace

    n_neuron = trace['n_neuron']
    save_loc = join(trace['p'], 'OldMap')
    mkdir(save_loc)


    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(6*3, 4.5))
    axes[0] = Clear_Axes(axes[0])
    axes[1] = Clear_Axes(axes[1])
    axes[2] = Clear_Axes(axes[2])

    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[2].invert_yaxis()
    
    DrawMazeProfile(axes=axes[0], maze_type=trace['maze_type'], nx = 12)
    axes[0].axis([-0.6,11.6,-0.6,11.6])
    axes[1].axis([-0.6,11.6,-0.6,11.6])
    axes[2].axis([-0.6,11.6,-0.6,11.6])
    DrawMazeProfile(axes=axes[1], maze_type=trace['maze_type'], nx = 12)
    DrawMazeProfile(axes=axes[2], maze_type=trace['maze_type'], nx = 12)

    for i in tqdm(range(n_neuron)):
        axes[0], im1, cbar1 = RateMapAxes(
            ax=axes[0], 
            content=trace['tot']['old_map_clear'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            nx=12,
            title='Together, SI='+str(round(trace['tot']['old_SI'][i], 3))
        )
        color = 'black' if trace['tot']['is_placecell'][i] == 0 else 'red'
        axes[0].set_title('Together, SI='+str(round(trace['tot']['old_SI'][i], 3)), color=color)

        axes[1], im2, cbar2 = RateMapAxes(
            ax=axes[1], 
            content=trace['cis']['old_map_clear'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            nx=12,
            title='Cis, SI='+str(round(trace['cis']['old_SI'][i], 3))
        )
        color = 'black' if trace['cis']['is_placecell'][i] == 0 else 'red'
        axes[1].set_title('Cis, SI='+str(round(trace['cis']['old_SI'][i], 3)), color=color)

        axes[2], im3, cbar3 = RateMapAxes(
            ax=axes[2], 
            content=trace['trs']['old_map_clear'][i],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            nx=12,
            title='Trans, SI='+str(round(trace['trs']['old_SI'][i], 3))
        )
        color = 'black' if trace['trs']['is_placecell'][i] == 0 else 'red'
        axes[2].set_title('Trans, SI='+str(round(trace['trs']['old_SI'][i], 3)), color=color)
        
        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        im1.remove()
        im2.remove()
        im3.remove()

    return trace

def CisTransPeakCurve(trace: dict, **kwargs) -> dict:  
    maze_type = trace['maze_type']
    save_loc = join(trace['p'], 'PeakCurve')
    mkdir(save_loc)

    pc_idx = np.where((trace['cis']['is_placecell'] == 1)|(trace['trs']['is_placecell'] == 1))[0]

    # Sort by cis map
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,6))
    contents = cp.deepcopy(trace['cis']['old_map_smooth'][pc_idx, :])
    contents = contents[:, correct_paths[int(maze_type)]-1]
    y_order = get_y_order(contents)
    axes[0] = PeakCurveAxes(axes[0], trace['cis']['old_map_smooth'][pc_idx, :], title="cis",
                            maze_type=maze_type, y_order=y_order, **kwargs)
    axes[1] = PeakCurveAxes(axes[1], trace['trs']['old_map_smooth'][pc_idx, :], title='trans',
                            maze_type=maze_type, y_order=y_order, is_invertx=True, **kwargs)
    plt.savefig(join(save_loc, 'sort by cis.png'), dpi = 600)
    plt.savefig(join(save_loc, 'sort by cis.svg'), dpi = 600)
    plt.close()

    # Sort by trs map
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,6))
    contents = cp.deepcopy(trace['trs']['old_map_smooth'][pc_idx, :])
    contents = contents[:, correct_paths[int(maze_type)]-1]
    y_order = get_y_order(contents)
    axes[0] = PeakCurveAxes(axes[0], trace['cis']['old_map_smooth'][pc_idx, :], title="cis",
                            maze_type=maze_type, y_order=y_order, **kwargs)
    axes[1] = PeakCurveAxes(axes[1], trace['trs']['old_map_smooth'][pc_idx, :], title='trans',
                            maze_type=maze_type, y_order=y_order, is_invertx=True, **kwargs)
    plt.savefig(join(save_loc, 'sort by trs.png'), dpi = 600)
    plt.savefig(join(save_loc, 'sort by trs.svg'), dpi = 600)
    plt.close()

    return trace

def CisTransLocTimeCurve(trace: dict, **kwargs) -> dict:
    maze_type = trace['maze_type']
    save_loc = join(trace['p'], 'LocTimeCurve')
    mkdir(save_loc)
    Graph = NRG[int(maze_type)]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,6))
    
    frame_label = BehavEvents.get_frame_labels(trace['correct_nodes'], maze_type=maze_type)
    cis_idx = np.where(frame_label==1)[0]
    trs_idx = np.where(frame_label==-1)[0]
    old_nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)
    linearized_x = np.zeros_like(trace['correct_nodes'], np.float64)

    for i in range(old_nodes.shape[0]):
        linearized_x[i] = Graph[int(old_nodes[i])]
    
    linearized_x = linearized_x + np.random.rand(old_nodes.shape[0]) - 0.5

    cis_x, trs_x = linearized_x[cis_idx], linearized_x[trs_idx]
    cis_t, trs_t = trace['correct_time'][cis_idx], trace['correct_time'][trs_idx]
    n_neuron = trace['n_neuron']

    for i in tqdm(range(n_neuron)):
        cis_color = 'red' if trace['cis']['is_placecell'][i] == 1 else 'black'
        axes[0], a1, b1 = LocTimeCurveAxes(
            axes[0], 
            behav_time=cis_t, 
            spikes=trace['cis']['Spikes'][i], 
            spike_time=trace['cis']['ms_time_behav'], 
            maze_type=maze_type, 
            given_x=cis_x,
            title='cis',
            title_color=cis_color,
        )

        trs_color = 'red' if trace['trs']['is_placecell'][i] == 1 else 'black'
        axes[1], a2, b2 = LocTimeCurveAxes(
            axes[1], 
            behav_time=trs_t, 
            spikes=trace['trs']['Spikes'][i], 
            spike_time=trace['trs']['ms_time_behav'], 
            maze_type=maze_type, 
            given_x=trs_x,
            title='trans',
            title_color=trs_color,
        )

        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)
        a = a1 + a2 + b1 + b2
        for j in a:
            j.remove()
    
    return trace

def FiringRateProcessing(trace: dict) -> dict:
    spike_num_cis = np.nansum(trace['cis']['Spikes'], axis = 1)
    mean_rate_cis = spike_num_cis / trace['cis']['t_total']
    peak_rate_cis = np.nanmax(trace['cis']['smooth_map_all'], axis = 1)
    trace['cis']['mean_rate'] = cp.deepcopy(mean_rate_cis)
    trace['cis']['peak_rate'] = cp.deepcopy(peak_rate_cis)

    spike_num_trs = np.nansum(trace['trs']['Spikes'], axis = 1)
    mean_rate_trs = spike_num_trs / trace['trs']['t_total']
    peak_rate_trs = np.nanmax(trace['trs']['smooth_map_all'], axis = 1)
    trace['trs']['mean_rate'] = cp.deepcopy(mean_rate_trs)
    trace['trs']['peak_rate'] = cp.deepcopy(peak_rate_trs)

    save_loc = join(trace['p'], 'others')
    mkdir(save_loc)

    idx = np.where((trace['cis']['is_placecell']==1)|(trace['trs']['is_placecell']==1))[0]
    mean_rate_cis, mean_rate_trs = mean_rate_cis[idx], mean_rate_trs[idx]
    peak_rate_cis, peak_rate_trs = peak_rate_cis[idx], peak_rate_trs[idx]
    spike_num_cis, spike_num_trs = spike_num_cis[idx], spike_num_trs[idx]

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,4))
    axes[0].set_title("Mean Event Rate / Hz")
    xy_max = int(np.max([np.nanmax(mean_rate_cis), np.nanmax(mean_rate_trs)])*100)/100
    axes[0] = Clear_Axes(axes[0], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([0, xy_max])
    axes[0].set_xlabel("Cis")
    axes[0].set_yticks([0, xy_max])
    axes[0].set_ylabel("Trans")
    colors = sns.color_palette("rocket", 2)
    x = np.linspace(0, xy_max, 2)
    axes[0].plot(x,x,'--', color='black')
    axes[0].plot(mean_rate_cis, mean_rate_trs, 'o', markeredgewidth = 0, markersize = 2, color = colors[0])
    axes[0].plot([np.nanmean(mean_rate_cis)], [np.nanmean(mean_rate_trs)], '^', color = colors[1])
    axes[0].axis([0, xy_max, 0, xy_max])

    axes[1].set_title("Peak Event Rate / Hz")
    xy_max = int(np.max([np.nanmax(peak_rate_cis), np.nanmax(peak_rate_trs)])*100)/100
    axes[1] = Clear_Axes(axes[1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([0, xy_max])
    axes[1].set_xlabel("Cis")
    axes[1].set_yticks([0, xy_max])
    axes[1].set_ylabel("Trans")
    x = np.linspace(0, xy_max, 2)
    axes[1].plot(x,x,'--', color='black')
    axes[1].plot(peak_rate_cis, peak_rate_trs, 'o', markeredgewidth = 0, markersize = 2, color = colors[0])
    axes[1].plot([np.nanmean(peak_rate_cis)], [np.nanmean(peak_rate_trs)], '^', color = colors[1])
    axes[1].axis([0, xy_max, 0, xy_max])

    axes[2].set_title("Event number")
    xy_max = int(np.max([np.nanmax(spike_num_cis), np.nanmax(spike_num_trs)])*100)/100
    axes[2] = Clear_Axes(axes[2], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    axes[2].set_aspect("equal")
    axes[2].set_xticks([0, xy_max])
    axes[2].set_xlabel("Cis")
    axes[2].set_yticks([0, xy_max])
    axes[2].set_ylabel("Trans")
    x = np.linspace(0, xy_max, 2)
    axes[2].plot(x,x,'--', color='black')
    axes[2].plot(spike_num_cis, spike_num_trs, 'o', markeredgewidth = 0, markersize = 2, color = colors[0])
    axes[2].plot([np.nanmean(spike_num_cis)], [np.nanmean(spike_num_trs)], '^', color = colors[1])
    axes[2].axis([0, xy_max, 0, xy_max])

    plt.savefig(join(save_loc, 'firing rate.png'), dpi = 600)
    plt.savefig(join(save_loc, 'firing rate.svg'), dpi = 600)
    plt.close()

    return trace

def HalfHalfStability(trace: dict) -> dict:
    _nbins = 2304
    _coords_range = [0, _nbins +0.0001 ]
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 2, _range = 7, nx = 48)

    half_len =int(trace['cis']['spike_nodes'].shape[0]/2)
    spike_nodes_cis_fir, spike_nodes_cis_sec = trace['cis']['spike_nodes'][0:half_len], trace['cis']['spike_nodes'][half_len::]
    Spikes_cis_fir, Spikes_cis_sec = trace['cis']['Spikes'][:, 0:half_len], trace['cis']['Spikes'][:, half_len::]
    dt_cis_fir, dt_cis_sec = trace['cis']['dt'][0:half_len], trace['cis']['dt'][half_len::]

    occu_time_cis_fir, _, _ = scipy.stats.binned_statistic(
            spike_nodes_cis_fir,
            dt_cis_fir,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)

    occu_time_cis_sec, _, _ = scipy.stats.binned_statistic(
            spike_nodes_cis_sec,
            dt_cis_sec,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)       

    SilentNeuron = trace['cis']['SilentNeuron']
    rate_map_all_cis_fir, rate_map_clear_cis_fir, smooth_map_all_cis_fir, nanPos_cis_fir = calc_ratemap(Spikes = Spikes_cis_fir, spike_nodes = spike_nodes_cis_fir,
                                                                        _nbins = 48*48, occu_time = occu_time_cis_fir, Ms = Ms, is_silent = SilentNeuron)

    rate_map_all_cis_sec, rate_map_clear_cis_sec, smooth_map_all_cis_sec, nanPos_cis_sec = calc_ratemap(Spikes = Spikes_cis_sec, spike_nodes = spike_nodes_cis_sec,
                                                                        _nbins = 48*48, occu_time = occu_time_cis_sec, Ms = Ms, is_silent = SilentNeuron)

    fir_sec_corr_cis = np.zeros(Spikes_cis_fir.shape[0], dtype=np.float64)
    for i in range(Spikes_cis_fir.shape[0]):
        fir_sec_corr_cis[i], _ = scipy.stats.pearsonr(smooth_map_all_cis_fir[i, :], smooth_map_all_cis_sec[i, :])

    t_total_cis_fir = np.nansum(occu_time_cis_fir) / 1000
    t_total_cis_sec = np.nansum(occu_time_cis_sec) / 1000
    t_nodes_frac_cis_fir = occu_time_cis_fir / 1000 / (t_total_cis_fir + 1E-6)
    t_nodes_frac_cis_sec = occu_time_cis_sec / 1000 / (t_total_cis_sec + 1E-6)
    SI_cis_fir = calc_SI(Spikes_cis_fir, rate_map=smooth_map_all_cis_fir, t_total=t_total_cis_fir, t_nodes_frac=t_nodes_frac_cis_fir)
    SI_cis_sec = calc_SI(Spikes_cis_sec, rate_map=smooth_map_all_cis_sec, t_total=t_total_cis_sec, t_nodes_frac=t_nodes_frac_cis_sec)

    appendix_cis = {'rate_map_fir':rate_map_all_cis_fir, 'clear_map_fir':rate_map_clear_cis_fir, 'smooth_map_fir':smooth_map_all_cis_fir, 'nanPos_fir':nanPos_cis_fir, 'occu_time_fir':occu_time_cis_fir,
                'rate_map_sec':rate_map_all_cis_sec, 'clear_map_sec':rate_map_clear_cis_sec, 'smooth_map_sec':smooth_map_all_cis_sec, 'nanPos_sec':nanPos_cis_sec, 'occu_time_sec':occu_time_cis_sec,
                't_total_fir': t_total_cis_fir, 't_total_sec': t_total_cis_sec, 't_nodes_frac_fir': t_nodes_frac_cis_fir, 't_nodes_frac_sec':t_nodes_frac_cis_sec, 'SI_fir': SI_cis_fir,
                'SI_sec': SI_cis_sec, 'fir_sec_corr': fir_sec_corr_cis}
    trace['cis'].update(appendix_cis)


    # calculating the half half firing rate of cis and trs
    half_len =int(trace['trs']['spike_nodes'].shape[0]/2)
    spike_nodes_trs_fir, spike_nodes_trs_sec = trace['trs']['spike_nodes'][0:half_len], trace['trs']['spike_nodes'][half_len::]
    Spikes_trs_fir, Spikes_trs_sec = trace['trs']['Spikes'][:, 0:half_len], trace['trs']['Spikes'][:, half_len::]
    dt_trs_fir, dt_trs_sec = trace['trs']['dt'][0:half_len], trace['trs']['dt'][half_len::]

    occu_time_trs_fir, _, _ = scipy.stats.binned_statistic(
            spike_nodes_trs_fir,
            dt_trs_fir,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)

    occu_time_trs_sec, _, _ = scipy.stats.binned_statistic(
            spike_nodes_trs_sec,
            dt_trs_sec,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)       

    SilentNeuron = trace['trs']['SilentNeuron']
    rate_map_all_trs_fir, rate_map_clear_trs_fir, smooth_map_all_trs_fir, nanPos_trs_fir = calc_ratemap(Spikes = Spikes_trs_fir, spike_nodes = spike_nodes_trs_fir,
                                                                        _nbins = 48*48, occu_time = occu_time_trs_fir, Ms = Ms, is_silent = SilentNeuron)

    rate_map_all_trs_sec, rate_map_clear_trs_sec, smooth_map_all_trs_sec, nanPos_trs_sec = calc_ratemap(Spikes = Spikes_trs_sec, spike_nodes = spike_nodes_trs_sec,
                                                                        _nbins = 48*48, occu_time = occu_time_trs_sec, Ms = Ms, is_silent = SilentNeuron)

    fir_sec_corr_trs = np.zeros(Spikes_trs_fir.shape[0], dtype=np.float64)
    for i in range(Spikes_trs_fir.shape[0]):
        fir_sec_corr_trs[i], _ = scipy.stats.pearsonr(smooth_map_all_trs_fir[i, :], smooth_map_all_trs_sec[i, :])

    t_total_trs_fir = np.nansum(occu_time_trs_fir) / 1000
    t_total_trs_sec = np.nansum(occu_time_trs_sec) / 1000
    t_nodes_frac_trs_fir = occu_time_trs_fir / 1000 / (t_total_trs_fir + 1E-6)
    t_nodes_frac_trs_sec = occu_time_trs_sec / 1000 / (t_total_trs_sec + 1E-6)
    SI_trs_fir = calc_SI(Spikes_trs_fir, rate_map=smooth_map_all_trs_fir, t_total=t_total_trs_fir, t_nodes_frac=t_nodes_frac_trs_fir)
    SI_trs_sec = calc_SI(Spikes_trs_sec, rate_map=smooth_map_all_trs_sec, t_total=t_total_trs_sec, t_nodes_frac=t_nodes_frac_trs_sec)

    appendix_trs = {'rate_map_fir':rate_map_all_trs_fir, 'clear_map_fir':rate_map_clear_trs_fir, 'smooth_map_fir':smooth_map_all_trs_fir, 'nanPos_fir':nanPos_trs_fir, 'occu_time_fir':occu_time_trs_fir,
                'rate_map_sec':rate_map_all_trs_sec, 'clear_map_sec':rate_map_clear_trs_sec, 'smooth_map_sec':smooth_map_all_trs_sec, 'nanPos_sec':nanPos_trs_sec, 'occu_time_sec':occu_time_trs_sec,
                't_total_fir': t_total_trs_fir, 't_total_sec': t_total_trs_sec, 't_nodes_frac_fir': t_nodes_frac_trs_fir, 't_nodes_frac_sec':t_nodes_frac_trs_sec, 'SI_fir': SI_trs_fir,
                'SI_sec': SI_trs_sec, 'fir_sec_corr': fir_sec_corr_trs}
    trace['trs'].update(appendix_trs)


    save_loc = join(trace['p'], 'others')
    mkdir(save_loc)

    cis_idx = np.where(trace['cis']['is_placecell'] == 1)[0]
    trs_idx = np.where(trace['trs']['is_placecell'] == 1)[0]

    fig = plt.figure(figsize=(6, 4.5))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(fir_sec_corr_cis[cis_idx], bins=40, range = (-1,1), rwidth = 0.8)
    ax.set_title("In session stability (Half-Half), Cis")
    ax.set_xticks(np.linspace(-1,1,11))
    plt.savefig(join(save_loc, 'In session stability distribution [cis].png'), dpi=600)
    plt.savefig(join(save_loc, 'In session stability distribution [cis].svg'), dpi=600)
    plt.close()

    fig = plt.figure(figsize=(6, 4.5))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(fir_sec_corr_trs[trs_idx], bins=40, range = (-1,1), rwidth = 0.8, alpha=0.5)
    ax.set_title("In session stability (Half-Half), Trans")
    ax.set_xticks(np.linspace(-1,1,11))
    plt.savefig(join(save_loc, 'In session stability distribution [trans].png'), dpi=600)
    plt.savefig(join(save_loc, 'In session stability distribution [trans].svg'), dpi=600)
    plt.close()
    return trace

def DirectionalityShuffltTest(trace: dict, percent: float = 95) -> dict:
    smooth_map_cis = trace['cis']['smooth_map_all']
    smooth_map_trs = trace['trs']['smooth_map_all']
    n_neuron = trace['n_neuron']

    cis_trs_corr = np.zeros(n_neuron, np.float64)
    directionality = np.ones(n_neuron, np.float64)*np.nan
    save_loc = join(trace['p'], 'Directionality')
    mkdir(save_loc)

    cis_to_trs = []
    print("        Build directionality shuffle distribution")
    for i in tqdm(range(n_neuron)):
        cis_trs_corr[i], _ = pearsonr(smooth_map_cis[i], smooth_map_trs[i])
        for j in range(n_neuron):
            if i == j:
                continue
            rand_corr, _ = pearsonr(smooth_map_cis[i], smooth_map_trs[j])
            if not np.isnan(rand_corr):
                cis_to_trs.append(rand_corr)

    fig = plt.figure(figsize=(6, 4.5))
    ax = Clear_Axes(axes=plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(cis_to_trs, bins = 40, range=(-1,1), rwidth=0.9, alpha = 0.5)

    thre = np.percentile(cis_to_trs, percent)
    print('        The shuffle test boundry is (Correlation):',thre)
    ax.axvline(np.percentile(cis_to_trs, percent), color = 'gray')
    ax.set_xticks(np.linspace(-1,1,11))
    ax.set_xlabel('correlation')

    print("        Shuffle test for directional tuning cells.")
    idx = np.where((trace['cis']['is_placecell']==1)&(trace['trs']['is_placecell']==1))[0]
    for i in tqdm(idx):
        if cis_trs_corr[i] > thre:
            directionality[i] = 0
            color = 'red'
            title = 'Non-directional tuning place cell'
        else:
            directionality[i] = 1
            color = 'black'
            title = 'Directional tuning place cell'

        a = ax.axvline(cis_trs_corr[i], color = color)
        ax.set_title(title)

        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi=600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi=600)

        a.remove()
    
    plt.close()

    directional_ratio = np.nanmean(directionality)
    trace['cis_trs_corr'] = cis_trs_corr
    trace['is_dirtuning_cell'] = directionality
    trace['directional_percent'] = directional_ratio
    print("        The ratio of the directional tuning cell is: "+str(round(directional_ratio, 2)))


    # Shuffle test whether the directional ratio is meaningful:
    ratio_shuffle = np.zeros(1000)
    for i in tqdm(range(1000)):
        values = np.random.choice(cis_to_trs, size=len(idx), replace=False)
        ratio_shuffle[i] = len(np.where(values <= thre)[0])/len(idx) * 100

    fig = plt.figure(figsize=(6, 4.5))
    ax = Clear_Axes(axes=plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.hist(ratio_shuffle, bins = 40, range=(60,100), rwidth=0.9, alpha=0.5)

    thre = np.percentile(ratio_shuffle, 0.05)
    ax.axvline(thre, color = 'gray')
    ax.set_xticks(np.linspace(60,100,5))
    ax.set_xlabel('percentage / %')
    color = 'red' if directional_ratio*100 < thre else 'black'
    ax.axvline(directional_ratio*100, color=color)
    plt.savefig(join(save_loc, 'Whether the percentage non-directional tuning cells is significant.png'), dpi=600)
    plt.savefig(join(save_loc, 'Whether the percentage non-directional tuning cells is significant.svg'), dpi=600)
    plt.close()

    trace['is_nondirectional_rand'] = 1 if directional_ratio*100 >= thre else 0

    return trace

def CountDirectionality(trace: dict) -> dict:
    nonsilent_num_both = 0
    for i in range(trace['n_neuron']):
        if i not in trace['cis']['SilentNeuron'] and i not in trace['trs']['SilentNeuron']:
            nonsilent_num_both += 1
    nonsilent_num_cis = trace['n_neuron'] - len(trace['cis']['SilentNeuron']) - nonsilent_num_both
    nonsilent_num_trs = trace['n_neuron'] - len(trace['trs']['SilentNeuron']) - nonsilent_num_both

    save_loc = join(trace['p'], 'others')
    mkdir(save_loc)

    plt.figure(figsize = (6,4.5))
    ax = Clear_Axes(plt.axes())
    venn2(subsets=(nonsilent_num_cis, nonsilent_num_trs, nonsilent_num_both), set_labels=('Cis', 'Trans'), alpha = 0.5, ax=ax)
    ax.set_title("Silent cell ratio")
    ax.set_aspect("equal")
    plt.savefig(join(save_loc, 'Venn - Silent neurons in 2 directions.png'), dpi=600)
    plt.savefig(join(save_loc, 'Venn - Silent neurons in 2 directions.svg'), dpi=600)
    plt.close()

    place_cells_both = len(np.where((trace['cis']['is_placecell']==1)&(trace['trs']['is_placecell']==1))[0])
    place_cells_cis = np.nansum(trace['cis']['is_placecell']) - place_cells_both
    place_cells_trs = np.nansum(trace['trs']['is_placecell']) - place_cells_both

    plt.figure(figsize = (6,4.5))
    ax = Clear_Axes(plt.axes())
    venn2(subsets=(place_cells_cis, place_cells_trs, place_cells_both), set_labels=('Cis', 'Trans'), alpha = 0.5, ax=ax)
    ax.set_aspect("equal")
    ax.set_title("place cells cell ratio")
    plt.savefig(join(save_loc, 'Venn - Place cells in 2 directions.png'), dpi=600)
    plt.savefig(join(save_loc, 'Venn - Place cells in 2 directions.svg'), dpi=600)
    plt.close()

    trace['silent_overlap'] = (nonsilent_num_cis, nonsilent_num_trs, nonsilent_num_both)
    trace['placecell_overlap'] = (place_cells_cis, place_cells_trs, place_cells_both)

    return trace

def run_all_mice_DLC(i: int, f: pd.DataFrame, work_flow: str, 
                     v_thre: float = 2.5, cam_degree = 0, speed_sm_args = {}):

    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])
    session = int(f['session'][i])

    if behavior_paradigm not in ['ReverseMaze', 'HairpinMaze']:
        raise ValueError(f"This is code for reverse maze and hairpin maze specifically! But {behavior_paradigm} is got.")

    totalpath = work_flow
    p = os.path.join(totalpath, str(MiceID), str(date),"session "+str(session))

    if os.path.exists(os.path.join(p,'trace_behav.pkl')):
        with open(os.path.join(p, 'trace_behav.pkl'), 'rb') as handle:
            trace = pickle.load(handle)
    else:
        warnings.warn(f"{os.path.join(p,'trace_behav.pkl')} is not exist!")
    
    trace['p'] = p
    f.loc[i, 'Path'] = p
    coverage = coverage_curve(trace['processed_pos_new'], maze_type=trace['maze_type'], save_loc=os.path.join(p, 'behav'))
    trace['coverage'] = coverage

    # Read File
    print("    A. Read ms.mat File")
    ms_path = os.path.join(folder, 'ms.mat')
    if os.path.exists(ms_path) == False:
        warnings.warn(f"{ms_path} is not exist!")

    try:
        ms_mat = loadmat(ms_path)
        ms = ms_mat['ms']
        #FiltTraces = np.array(ms['FiltTraces'][0][0]).T
        RawTraces = np.array(ms['RawTraces'][0][0]).T
        DeconvSignal = np.array(ms['DeconvSignals'][0][0]).T
        ms_time = np.array(ms['time'][0])[0,].T[0]
    except:
        with h5py.File(ms_path, 'r') as f:
            ms_mat = f['ms']
            FiltTraces = np.array(ms_mat['FiltTraces'])
            RawTraces = np.array(ms_mat['RawTraces'])
            DeconvSignal = np.array(ms_mat['DeconvSignals'])
            ms_time = np.array(ms_mat['time'],dtype = np.int64)[0,]

    plot_split_trajectory_directional(trace, behavior_paradigm = behavior_paradigm, split_args={})

    print("    B. Calculate putative spikes and correlated location from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 3)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # calc ms speed
    behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
    smooth_speed = uniform_smooth_speed(behav_speed, **speed_sm_args)
    
    ms_speed = calc_ms_speed(behav_speed=smooth_speed, behav_time=trace['correct_time'], ms_time=ms_time)

    # Delete NAN value in spike nodes
    print("      - Delete NAN values in data.")
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    Spikes = cp.deepcopy(Spikes_original[:,idx])
    spike_nodes = cp.deepcopy(spike_nodes_original[idx])
    ms_time_behav = cp.deepcopy(ms_time[idx])
    ms_speed_behav = cp.deepcopy(ms_speed[idx])
    dt = np.append(np.ediff1d(ms_time_behav), 33)
    dt[np.where(dt >= 100)[0]] = 100
    spike_num_mon2 = np.nansum(Spikes, axis = 1)
    
    # Filter the speed (spf: speed filter)
    print(f"      - Filter spikes with speed {v_thre} cm/s.")
    spf_idx = np.where(ms_speed_behav >= v_thre)[0]
    spf_results = [ms_speed_behav.shape[0], spf_idx.shape[0]]
    print(f"        {spf_results[0]} frames -> {spf_results[1]} frames.")
    print(f"        Remain rate: {round(spf_results[1]/spf_results[0]*100, 2)}%")
    print(f"        Delete rate: {round(100 - spf_results[1]/spf_results[0]*100, 2)}%")
    ms_time_behav = ms_time_behav[spf_idx]
    Spikes = Spikes[:, spf_idx]
    spike_nodes = spike_nodes[spf_idx]
    ms_speed_behav = ms_speed_behav[spf_idx]
    dt = dt[spf_idx]
    spike_num_mon3 = np.nansum(Spikes, axis = 1)
    
    # Delete InterLap Spikes
    print("      - Delete the inter-lap spikes.")
    frame_labels = get_spike_frame_label(
        ms_time=ms_time_behav, 
        spike_nodes=spike_nodes,
        trace=trace, 
        behavior_paradigm=behavior_paradigm
    )
    assert frame_labels.shape[0] == spike_nodes.shape[0]

    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 1, _range = 7, nx = 48)

    # cis direction
    idx = np.where(frame_labels == 1)[0]
    trace['cis'] = calc_rate_map_properties(
        trace['maze_type'],
        ms_time_behav[idx],
        Spikes[:, idx],
        spike_nodes[idx],
        ms_speed_behav[idx],
        dt[idx],
        Ms,
        trace['p'],
        kwargs = {'file_name': 'Place cell shuffle [cis]'},
        behavior_paradigm='ReverseMaze'
    )
    
    # trans direction
    idx = np.where(frame_labels == -1)[0]
    trace['trs'] = calc_rate_map_properties(
        trace['maze_type'],
        ms_time_behav[idx],
        Spikes[:, idx],
        spike_nodes[idx],
        ms_speed_behav[idx],
        dt[idx],
        Ms,
        trace['p'],
        kwargs = {'file_name': 'Place cell shuffle [trans]'},
        behavior_paradigm='ReverseMaze'
    )

    # Together the two dimension
    Spikes, spike_nodes, ms_time_behav, ms_speed_behav, dt = Delete_InterLapSpike(behav_time = trace['correct_time'], ms_time = ms_time_behav, 
                                                                              Spikes = Spikes, spike_nodes = spike_nodes, dt = dt, ms_speed=ms_speed_behav,
                                                                              behavior_paradigm = behavior_paradigm, trace = trace)
    trace['tot'] = calc_rate_map_properties(
        trace['maze_type'],
        ms_time_behav,
        Spikes,
        spike_nodes,
        ms_speed_behav,
        dt,
        Ms,
        trace['p'],
        kwargs = {'file_name': 'Place cell shuffle [total]'},
        behavior_paradigm='ReverseMaze'
    )
    spike_num_mon4 = np.nansum(Spikes, axis = 1)
    
    plot_spike_monitor(spike_num_mon1, spike_num_mon2, spike_num_mon3, spike_num_mon4, save_loc = os.path.join(trace['p'], 'behav'))

    print("    C. Calculating firing rate for each neuron and identified their place fields (those areas which firing rate >= 50% peak rate)")
    # Set occu_time <= 50ms spatial bins as nan to avoid too big firing rate

    trace_ms = {'Spikes_original':Spikes_original, 'spike_nodes_original':spike_nodes_original, 'ms_speed_original': ms_speed, 'RawTraces':RawTraces,'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'ms_folder':folder, 'speed_filter_results': spf_results, 'n_neuron': Spikes_original.shape[0], 'Ms':Ms}

    trace.update(trace_ms)
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("    Plotting:")
    print("      1. Ratemap")
    #RateMap(trace)

    print("      2. Tracemap")
    #TraceMap(trace)
  
    print("      3. Quarter_map")
    #trace = QuarterMap(trace, isDraw = False)

    print("      4. Oldmap")
    trace = OldMap(trace, is_draw=False)
       
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("      5. PeakCurve")
    CisTransPeakCurve(trace)
    
    print("      6. Combining tracemap, rate map(48 x 48), old map(12 x 12) and quarter map(24 x 24)")
    #CombineMap(trace)
    
    if trace['maze_type'] != 0:

        # LocTimeCurve
        print("      7. LocTimeCurve:")
        trace = CisTransLocTimeCurve(trace) 

        print("      8. Test the directionality of the place cells.")
        trace = DirectionalityShuffltTest(trace) 
        print("    Analysis:")
        print("      A. Firing rate analysis")
        trace = FiringRateProcessing(trace)
        print("      B. In session stability")
        trace = HalfHalfStability(trace)
        print("      C. Counting the basic firing properties at the 2 directions")
        trace = CountDirectionality(trace)

    """ 
    
    # Firing Rate Processing:
    print("      B. Firing rate Analysis")
    trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = 30)
    
    # Cross-Lap Analysis
    if behavior_paradigm in ['ReverseMaze', 'CrossMaze', 'DSPMaze']:
        print("      C. Cross-Lap Analysis")
        trace = CrossLapsCorrelation(trace, behavior_paradigm = behavior_paradigm)
        print("      D. Old Map Split")
        trace = OldMapSplit(trace)
        print("      E. Calculate Odd-even Correlation")
        trace = odd_even_correlation(trace)
        print("      F. Calculate Half-half Correlation")
        trace = half_half_correlation(trace)
        print("      G. In Field Correlation")
        trace = field_specific_correlation(trace)
    """    
    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")

    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)


if __name__ == '__main__':
    """
    f = pd.read_excel(r"G:\YSY\Hairpin_maze\Hairpin_maze_paradigm.xlsx", sheet_name='calcium')
    work_flow = r'G:\YSY\Hairpin_maze'

    for i in range(len(f)):
        print(i, int(f['MiceID'][i]), int(f['date'][i]), int(f['session'][i]))
        run_all_mice_DLC(i, f, work_flow=work_flow, cam_degree = int(f['cam_degree'][i]))
    
        print("Done.", end='\n\n\n')
    """

    with open(r"G:\YSY\Hairpin_maze\10209\20230613\session 1\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    print(trace['maze_type'])
    a = trace['tot']['rate_map_clear'][0]
    
    Ms = SmoothMatrix(3)
    b = np.dot(a, Ms.T)
    plt.imshow(np.reshape(b, [48,48]))
    plt.show()