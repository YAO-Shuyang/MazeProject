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
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns
from mylib.behavior.behavevents import BehavEvents
from mylib.maze_graph import correct_paths, NRG, Father2SonGraph, DSP_correct_graph1, DSP_incorrect_graph1
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile, clear_NAN, mkdir, SpikeNodes, SpikeType
from mylib.maze_utils3 import plot_trajactory, spike_nodes_transform, SmoothMatrix, occu_time_transform
from mylib.preprocessing_ms import coverage_curve, calc_speed, uniform_smooth_speed, calc_ratemap
from mylib.preprocessing_ms import plot_spike_monitor, calc_ms_speed
from mylib.preprocessing_ms import calc_SI
from mylib.divide_laps.lap_split import LapSplit
from mylib.calcium.axes.peak_curve import get_y_order
from scipy.io import loadmat
from tqdm import tqdm
from mylib import RateMapAxes, TraceMapAxes, PeakCurveAxes, LocTimeCurveAxes
from mylib.calcium.firing_rate import calc_rate_map_properties


def classify_lap(behav_nodes: np.ndarray, beg_idx: np.ndarray, start_from: str = 'correct'):
    classifications_correct = {
        0: [1, 2, 13, 14, 25, 26],
        1: [23, 22, 34, 33],
        2: [66, 65, 64, 63],
        3: [99, 87, 88, 76],
        4: [1, 2, 13, 14, 25, 26],
    }

    classifications_incorrect = {
        0: [1, 2, 13, 14, 25, 26],
        1: [8, 7, 6, 18],
        2: [93, 105, 106, 94],
        3: [135, 134, 133, 121],
        4: [1, 2, 13, 14, 25, 26],
    }

    classes = classifications_correct if start_from == 'correct' else classifications_incorrect
    
    lap_type = np.zeros_like(beg_idx, np.int64) 
    laps = beg_idx.shape[0]

    for i in range(laps):
        is_find = False
        for k in classes.keys():
            if k == 0 and np.nansum(lap_type) != 0:
                continue

            if behav_nodes[beg_idx[i]] in classes[k]:
                lap_type[i] = k
                is_find = True
                break

        if not is_find:
            raise ValueError(f"Fail to identify which node the lap starts! Lap {i+1}, Node {behav_nodes[beg_idx[i]]}, Index {beg_idx[i]}, Classification set {classes}")

    return lap_type
    
def plot_split_trajectory(trace: dict):
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    behav_pos = cp.deepcopy(trace['correct_pos'])
    behav_time = cp.deepcopy(trace['behav_time'])

    save_loc = os.path.join(trace['p'], 'behav','laps_trajactory')
    mkdir(save_loc)

    laps = beg_idx.shape[0]

    for k in tqdm(range(laps)):
        loc_x, loc_y = behav_pos[beg_idx[k]:end_idx[k]+1, 0] / 20 - 0.5, behav_pos[beg_idx[k]:end_idx[k]+1, 1] / 20 - 0.5
        fig = plt.figure(figsize = (6,6))
        ax = Clear_Axes(plt.axes())
        ax.set_title('Frame: '+str(beg_idx[k])+' -> '+str(end_idx[k])+'\n'+'Time:  '+str(behav_time[beg_idx[k]]/1000)+' -> '+str(behav_time[end_idx[k]]/1000))
        DrawMazeProfile(maze_type=trace['maze_type'], nx = 48, color='black', axes=ax)
        ax.invert_yaxis()
        ax.plot(loc_x, loc_y, 'o', color = 'red', markeredgewidth = 0, markersize = 3)

        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.png'), dpi=600)
        plt.savefig(join(save_loc, 'Lap '+str(k+1)+'.svg'), dpi=600)
        plt.close()

def split_calcium_data(
    lap_idx: np.ndarray,
    trace: dict, 
    ms_time: np.ndarray
):
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    beg_idx_ms, end_idx_ms = np.zeros_like(beg_idx), np.zeros_like(end_idx)
    for i in range(beg_idx.shape[0]):
        beg_idx_ms[i] = np.where(ms_time >= trace['correct_time'][beg_idx[i]])[0][0]
        end_idx_ms[i] = np.where(ms_time <= trace['correct_time'][end_idx[i]])[0][-1]
    
    idx = np.concatenate([np.arange(beg_idx_ms[i], end_idx_ms[i]+1) for i in lap_idx])

    return idx 

def RateMap(trace: dict) -> dict:
    n_neuron = trace['n_neuron']
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6*3, 4.5*2))
    save_loc = join(trace['p'], 'RateMap')
    mkdir(save_loc)

    ax1 = Clear_Axes(axes[0, 0])
    ax2 = Clear_Axes(axes[0, 1])
    ax3 = Clear_Axes(axes[0, 2])
    ax4 = Clear_Axes(axes[1, 0])
    ax5 = Clear_Axes(axes[1, 1])
    _ = Clear_Axes(axes[1, 2])

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')

    DrawMazeProfile(axes=ax1, maze_type=trace['maze_type'], nx = 48, color = 'black')
    DrawMazeProfile(axes=ax2, maze_type=trace['maze_type'], nx = 48, color = 'black')
    DrawMazeProfile(axes=ax3, maze_type=trace['maze_type'], nx = 48, color = 'black')
    DrawMazeProfile(axes=ax4, maze_type=trace['maze_type'], nx = 48, color = 'black')
    DrawMazeProfile(axes=ax5, maze_type=trace['maze_type'], nx = 48, color = 'black')
    ax1.axis([-0.6,47.6,-0.6,47.6])
    ax2.axis([-0.6,47.6,-0.6,47.6])
    ax3.axis([-0.6,47.6,-0.6,47.6])
    ax4.axis([-0.6,47.6,-0.6,47.6])
    ax5.axis([-0.6,47.6,-0.6,47.6])

    old_node0 = spike_nodes_transform(trace['node 0']['spike_nodes'], 12)
    old_node1 = spike_nodes_transform(trace['node 1']['spike_nodes'], 12)
    old_node2 = spike_nodes_transform(trace['node 2']['spike_nodes'], 12)
    old_node3 = spike_nodes_transform(trace['node 3']['spike_nodes'], 12)
    old_node4 = spike_nodes_transform(trace['node 4']['spike_nodes'], 12)
    mask0, mask1, mask2, mask3, mask4 = np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan
    areas = DSP_correct_graph1 if trace['start_from'] == 'correct' else DSP_incorrect_graph1
    mask0[get_son_area(areas[0])-1] = 0
    mask1[get_son_area(areas[1])-1] = 0
    mask2[get_son_area(areas[2])-1] = 0
    mask3[get_son_area(areas[3])-1] = 0
    mask4[get_son_area(areas[4])-1] = 0

    for i in tqdm(range(n_neuron)):
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()

        ax1, im1, cbar1 = RateMapAxes(
            ax=ax1, 
            content=trace['node 0']['smooth_map_all'][i]+mask0,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 0']['SI_all'][i], 3))
        )
        color = 'black' if trace['node 0']['is_placecell'][i] == 0 else 'red'
        ax1.set_title('SI='+str(round(trace['node 0']['SI_all'][i], 3)), color=color)

        ax2, im2, cbar2 = RateMapAxes(
            ax=ax2, 
            content=trace['node 1']['smooth_map_all'][i]+mask1,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 1']['SI_all'][i], 3))
        )
        color = 'black' if trace['node 1']['is_placecell'][i] == 0 else 'red'
        ax2.set_title('SI='+str(round(trace['node 1']['SI_all'][i], 3)), color=color)

        ax3, im3, cbar3 = RateMapAxes(
            ax=ax3, 
            content=trace['node 2']['smooth_map_all'][i]+mask2,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 2']['SI_all'][i], 3))
        )
        color = 'black' if trace['node 2']['is_placecell'][i] == 0 else 'red'
        ax3.set_title('SI='+str(round(trace['node 2']['SI_all'][i], 3)), color=color)

        ax4, im4, cbar4 = RateMapAxes(
            ax=ax4, 
            content=trace['node 3']['smooth_map_all'][i]+mask3,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 3']['SI_all'][i], 3))
        )
        color = 'black' if trace['node 3']['is_placecell'][i] == 0 else 'red'
        ax4.set_title('SI='+str(round(trace['node 3']['SI_all'][i], 3)), color=color)

        ax5, im5, cbar5 = RateMapAxes(
            ax=ax5, 
            content=trace['node 4']['smooth_map_all'][i]+mask4,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 4']['SI_all'][i], 3))
        )
        color = 'black' if trace['node 4']['is_placecell'][i] == 0 else 'red'
        ax5.set_title('SI='+str(round(trace['node 4']['SI_all'][i], 3)), color=color)
        
        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        cbar4.remove()
        cbar5.remove()
        im1.remove()
        im2.remove()
        im3.remove()
        im4.remove()
        im5.remove()
    
    return trace

def TraceMap(trace: dict) -> dict:
    n_neuron = trace['n_neuron']
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6*3, 4.5*2))
    save_loc = join(trace['p'], 'TraceMap')
    mkdir(save_loc)

    ax1 = Clear_Axes(axes[0, 0])
    ax2 = Clear_Axes(axes[0, 1])
    ax3 = Clear_Axes(axes[0, 2])
    ax4 = Clear_Axes(axes[1, 0])
    ax5 = Clear_Axes(axes[1, 1])
    _ = Clear_Axes(axes[1,2])

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')

    DrawMazeProfile(axes=ax1, maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=ax2, maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=ax3, maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=ax4, maze_type=trace['maze_type'], nx=48,color='black')
    DrawMazeProfile(axes=ax5, maze_type=trace['maze_type'], nx=48,color='black')
    ax1.axis([-0.6,47.6,-0.6,47.6])
    ax2.axis([-0.6,47.6,-0.6,47.6])
    ax3.axis([-0.6,47.6,-0.6,47.6])
    ax4.axis([-0.6,47.6,-0.6,47.6])
    ax5.axis([-0.6,47.6,-0.6,47.6])

    beg, end = LapSplit(trace, trace['paradigm'])
    lap_type = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg, trace['start_from'])

    idx = np.where(lap_type == 0)[0]
    trajectory0 = trace['correct_pos'][beg[idx[0]]:end[idx[-1]]+1, :]
    behav_time0 = trace['correct_time'][beg[idx[0]]:end[idx[-1]]+1]

    idx = np.where(lap_type == 1)[0]
    trajectory1 = trace['correct_pos'][beg[idx[0]]:end[idx[-1]]+1, :]
    behav_time1 = trace['correct_time'][beg[idx[0]]:end[idx[-1]]+1]

    idx = np.where(lap_type == 2)[0]
    trajectory2 = trace['correct_pos'][beg[idx[0]]:end[idx[-1]]+1, :]
    behav_time2 = trace['correct_time'][beg[idx[0]]:end[idx[-1]]+1]

    idx = np.where(lap_type == 3)[0]
    trajectory3 = trace['correct_pos'][beg[idx[0]]:end[idx[-1]]+1, :]
    behav_time3 = trace['correct_time'][beg[idx[0]]:end[idx[-1]]+1]

    idx = np.where(lap_type == 4)[0]
    trajectory4 = trace['correct_pos'][beg[idx[0]]:end[idx[-1]]+1, :]
    behav_time4 = trace['correct_time'][beg[idx[0]]:end[idx[-1]]+1]

    for i in tqdm(range(n_neuron)):
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()

        ax1, a1, b1 = TraceMapAxes(
            ax=ax1, 
            trajectory=cp.deepcopy(trajectory0), 
            behav_time=behav_time0, 
            spikes=trace['node 0']['Spikes'][i], 
            spike_time=trace['node 0']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="Entire trajectory"
        )

        ax2, a2, b2 = TraceMapAxes(
            ax=ax2, 
            trajectory=cp.deepcopy(trajectory1), 
            behav_time=behav_time1, 
            spikes=trace['node 1']['Spikes'][i], 
            spike_time=trace['node 1']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="node 1"
        )

        ax3, a3, b3 = TraceMapAxes(
            ax=ax3, 
            trajectory=cp.deepcopy(trajectory2), 
            behav_time=behav_time2, 
            spikes=trace['node 2']['Spikes'][i], 
            spike_time=trace['node 2']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="node 2"
        )

        ax4, a4, b4 = TraceMapAxes(
            ax=ax4, 
            trajectory=cp.deepcopy(trajectory3), 
            behav_time=behav_time3, 
            spikes=trace['node 3']['Spikes'][i], 
            spike_time=trace['node 3']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="node 3"
        )

        ax5, a5, b5 = TraceMapAxes(
            ax=ax5, 
            trajectory=cp.deepcopy(trajectory4), 
            behav_time=behav_time4, 
            spikes=trace['node 4']['Spikes'][i], 
            spike_time=trace['node 4']['ms_time_behav'],
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            title="node 4"
        )

        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()

        a = a1 + a2 + a3 + a4 + a5 + b1 + b2 + b3 + b4 + b5
        for j in a:
            j.remove()

    return trace

def LocTimeCurve(trace: dict) -> dict:
    maze_type = trace['maze_type']
    save_loc = join(trace['p'], 'LocTimeCurve')
    mkdir(save_loc)
    Graph = NRG[int(maze_type)]

    fig = plt.figure(figsize=(4,6))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    
    old_nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)
    linearized_x = np.zeros_like(trace['correct_nodes'], np.float64)

    for i in range(old_nodes.shape[0]):
        linearized_x[i] = Graph[int(old_nodes[i])]
    
    linearized_x = linearized_x + np.random.rand(old_nodes.shape[0]) - 0.5

    n_neuron = trace['n_neuron']

    idx = np.where((trace['node 0']['is_placecell'] == 1)|(trace['node 4']['is_placecell'] == 1))[0]

    for i in tqdm(range(n_neuron)):
        color = 'red' if i in idx else 'black'
        ax, a1, b1 = LocTimeCurveAxes(
            ax, 
            behav_time=trace['correct_time'], 
            spikes=np.concatenate([trace['node '+str(j)]['Spikes'][i, :] for j in range(5)]), 
            spike_time=np.concatenate([trace['node '+str(j)]['ms_time_behav'] for j in range(5)]), 
            maze_type=maze_type, 
            given_x=linearized_x,
            title='cis',
            title_color=color,
        )

        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)
        a = a1 + b1
        for j in a:
            j.remove()
    
    return trace

def get_son_area(area: np.ndarray):
    return np.concatenate([Father2SonGraph[i] for i in area])

def PartialPeakCurve(trace: dict, **kwargs) -> dict:
    

    if trace['start_from'] == 'correct':
        paths = {
            0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            1: np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            2: np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            3: np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
        }
        areas = DSP_correct_graph1
    else:
        paths = {
            0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            1: np.array([8,7,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            2: np.array([93,105,106,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            3: np.array([135,134,133,121,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
        }
        areas = DSP_incorrect_graph1

    grid_spec = [len(paths[k]) for k in paths.keys()]
    fig, axes = plt.subplots(ncols=5, nrows=1, figsize=(4*5, 6), gridspec_kw={'width_ratios':grid_spec})
    n_neuron = trace['n_neuron']

    maze_type = trace['maze_type']
    save_loc = join(trace['p'], 'PeakCurve')
    mkdir(save_loc)

    pc_idx = np.where((trace['node 0']['is_placecell'] == 1)|(trace['node 4']['is_placecell'] == 1))[0]
    contents = trace['node 0']['old_map_smooth'][pc_idx, :]
    contents = contents[:, paths[0]-1]
    y_order = get_y_order(contents)

    x_ticks = np.array([len(paths[0]) - len(paths[i]) for i in range(4)] + [len(paths[0])-1], dtype=np.int64)
    x_labels = ['start', 'node 1', 'node 2', 'node 3', 'end']
        
    for i in range(5):
        axes[i].set_yticks([0, n_neuron-1],lables=[1, n_neuron])
        contents = trace['node '+str(i)]['old_map_smooth'][pc_idx, :]
        contents = contents[:, paths[i]-1]
        title = "start from node "+str(i) if i not in [0, 4] else "start from the entry"
        axes[i] = PeakCurveAxes(axes[i], contents, title=title, is_sortx=False,
                                maze_type=maze_type, y_order=y_order, **kwargs)
        axes[i].set_xticks(x_ticks[i::]-x_ticks[i], labels=x_labels[i::]) if i < 4 else axes[i].set_xticks(x_ticks, labels=x_labels)
        axes[i].set_xlim([-0.5, len(paths[i])-0.5])
    
    plt.savefig(os.path.join(save_loc, 'Partial Peak Curve.png'), dpi=600)
    plt.savefig(os.path.join(save_loc, 'Partial Peak Curve.svg'), dpi=600)
    plt.close()

    corr_matrix = np.zeros((n_neuron, 6), np.float64)
    for i in tqdm(range(n_neuron)):
        corr_matrix[i, 0], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[1])-1], trace['node 1']['smooth_map_all'][i][get_son_area(areas[1])-1])
        corr_matrix[i, 1], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[1])-1], trace['node 4']['smooth_map_all'][i][get_son_area(areas[1])-1])
        corr_matrix[i, 2], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[2])-1], trace['node 2']['smooth_map_all'][i][get_son_area(areas[2])-1])
        corr_matrix[i, 3], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[2])-1], trace['node 4']['smooth_map_all'][i][get_son_area(areas[2])-1])
        corr_matrix[i, 4], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[3])-1], trace['node 3']['smooth_map_all'][i][get_son_area(areas[3])-1])
        corr_matrix[i, 5], _ = pearsonr(trace['node 0']['smooth_map_all'][i][get_son_area(areas[3])-1], trace['node 4']['smooth_map_all'][i][get_son_area(areas[3])-1])

    trace['partial_map_corr'] = corr_matrix

    idx = np.where((trace['node 0']['is_placecell'] == 1)|(trace['node 4']['is_placecell'] == 1))[0]
    is_placecell = np.zeros(n_neuron)
    is_placecell[idx] = 1

    Data = {
        'Start Node': np.concatenate([np.repeat('Node 1', n_neuron*2), np.repeat('Node 2', n_neuron*2), np.repeat('Node 3', n_neuron*2)]),
        'Compare': np.concatenate([np.repeat('exper.', n_neuron),
                                   np.repeat('control', n_neuron),
                                   np.repeat('exper.', n_neuron),
                                   np.repeat('control', n_neuron),
                                   np.repeat('exper.', n_neuron),
                                   np.repeat('control', n_neuron)]),
        'Correlation': np.concatenate([corr_matrix[:, 0], corr_matrix[:, 1], corr_matrix[:, 2], corr_matrix[:, 3], corr_matrix[:, 4], corr_matrix[:, 5]]),
        'is_placecell': np.concatenate([is_placecell, is_placecell, is_placecell, is_placecell, is_placecell, is_placecell])
    }

    idx = np.where(Data['is_placecell'] == 1)[0]
    SubData = {
        'Start Node': Data['Start Node'][idx],
        'Compare': Data['Compare'][idx],
        'Correlation': Data['Correlation'][idx],
        'is_placecell': Data['is_placecell'][idx]
    }

    plt.figure(figsize = (8,6))
    colors = sns.color_palette("rocket", 4)
    ax = Clear_Axes(plt.axes(), close_spines=['top','right'], ifxticks=True, ifyticks=True)
    sns.barplot(
        x='Start Node', 
        y='Correlation', 
        data=SubData, 
        hue='Compare', 
        palette='rocket',
        capsize=0.1,
        width=0.6,
        errcolor='black',
    )
    mkdir(os.path.join(trace['p'], 'others'))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0,1,6))
    ax.legend(facecolor = None, edgecolor = None)
    plt.savefig(os.path.join(trace['p'], 'others', 'Correlation comparison.png'), dpi=600)
    plt.savefig(os.path.join(trace['p'], 'others', 'Correlation comparison.svg'), dpi=600)
    plt.close()

    return trace

def OldMap(trace: dict, is_draw: bool = True) -> dict:
    maze_type = trace['maze_type']
    # total old map
    Spikes = trace['node 0']['Spikes']
    spike_nodes = trace['node 0']['spike_nodes']
    occu_time = trace['node 0']['occu_time_spf'] if 'occu_time_spf' in trace['node 0'].keys() else trace['node 0']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    mask0 = np.zeros(144, np.float64)*np.nan
    for n in tqdm(old_nodes):
        mask0[n-1] = 0

    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['node 0']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['node 0']['old_map_all'] = cp.deepcopy(old_map_all)
    trace['node 0']['old_map_clear'] = cp.deepcopy(old_map_clear)
    trace['node 0']['old_map_smooth'] = cp.deepcopy(old_map_smooth)
    trace['node 0']['old_map_nanPos'] = cp.deepcopy(old_map_nanPos)
    trace['node 0']['old_nodes'] = cp.deepcopy(old_nodes)
    trace['node 0']['occu_time_old'] = cp.deepcopy(occu_time_old)
    trace['node 0']['old_t_total'] = cp.deepcopy(old_t_total)
    trace['node 0']['old_t_nodes_frac'] = cp.deepcopy(old_t_nodes_frac)
    trace['node 0']['old_SI'] = cp.deepcopy(SI)

    # node 1 old map
    Spikes = trace['node 1']['Spikes']
    spike_nodes = trace['node 1']['spike_nodes']
    occu_time = trace['node 1']['occu_time_spf'] if 'occu_time_spf' in trace['node 1'].keys() else trace['node 1']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    mask1 = np.zeros(144, np.float64)*np.nan
    for n in tqdm(old_nodes):
        mask1[n-1] = 0
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['node 1']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['node 1']['old_map_all'] = cp.deepcopy(old_map_all)
    trace['node 1']['old_map_clear'] = cp.deepcopy(old_map_clear)
    trace['node 1']['old_map_smooth'] = cp.deepcopy(old_map_smooth)
    trace['node 1']['old_map_nanPos'] = cp.deepcopy(old_map_nanPos)
    trace['node 1']['old_nodes'] = cp.deepcopy(old_nodes)
    trace['node 1']['occu_time_old'] = cp.deepcopy(occu_time_old)
    trace['node 1']['old_t_total'] = cp.deepcopy(old_t_total)
    trace['node 1']['old_t_nodes_frac'] = cp.deepcopy(old_t_nodes_frac)
    trace['node 1']['old_SI'] = cp.deepcopy(SI)

    # node 2 old map
    Spikes = trace['node 2']['Spikes']
    spike_nodes = trace['node 2']['spike_nodes']
    occu_time = trace['node 2']['occu_time_spf'] if 'occu_time_spf' in trace['node 2'].keys() else trace['node 2']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    mask2 = np.zeros(144, np.float64)*np.nan
    for n in tqdm(old_nodes):
        mask2[n-1] = 0
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['node 2']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['node 2']['old_map_all'] = cp.deepcopy(old_map_all)
    trace['node 2']['old_map_clear'] = cp.deepcopy(old_map_clear)
    trace['node 2']['old_map_smooth'] = cp.deepcopy(old_map_smooth)
    trace['node 2']['old_map_nanPos'] = cp.deepcopy(old_map_nanPos)
    trace['node 2']['old_nodes'] = cp.deepcopy(old_nodes)
    trace['node 2']['occu_time_old'] = cp.deepcopy(occu_time_old)
    trace['node 2']['old_t_total'] = cp.deepcopy(old_t_total)
    trace['node 2']['old_t_nodes_frac'] = cp.deepcopy(old_t_nodes_frac)
    trace['node 2']['old_SI'] = cp.deepcopy(SI)

    # node 3 old map
    Spikes = trace['node 3']['Spikes']
    spike_nodes = trace['node 3']['spike_nodes']
    occu_time = trace['node 3']['occu_time_spf'] if 'occu_time_spf' in trace['node 3'].keys() else trace['node 3']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    mask3 = np.zeros(144, np.float64)*np.nan
    for n in tqdm(old_nodes):
        mask3[n-1] = 0
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['node 3']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['node 3']['old_map_all'] = cp.deepcopy(old_map_all)
    trace['node 3']['old_map_clear'] = cp.deepcopy(old_map_clear)
    trace['node 3']['old_map_smooth'] = cp.deepcopy(old_map_smooth)
    trace['node 3']['old_map_nanPos'] = cp.deepcopy(old_map_nanPos)
    trace['node 3']['old_nodes'] = cp.deepcopy(old_nodes)
    trace['node 3']['occu_time_old'] = cp.deepcopy(occu_time_old)
    trace['node 3']['old_t_total'] = cp.deepcopy(old_t_total)
    trace['node 3']['old_t_nodes_frac'] = cp.deepcopy(old_t_nodes_frac)
    trace['node 3']['old_SI'] = cp.deepcopy(SI)


    # node 4 old map
    Spikes = trace['node 4']['Spikes']
    spike_nodes = trace['node 4']['spike_nodes']
    occu_time = trace['node 4']['occu_time_spf'] if 'occu_time_spf' in trace['node 4'].keys() else trace['node 4']['occu_time']
    
    old_nodes = spike_nodes_transform(spike_nodes = spike_nodes, nx = 12)
    mask4 = np.zeros(144, np.float64)*np.nan
    for n in tqdm(old_nodes):
        mask4[n-1] = 0
    occu_time_old = occu_time_transform(occu_time = occu_time, nx = 12)
    Ms = SmoothMatrix(maze_type = maze_type, nx = 12, _range = 1, sigma = 2)
    old_map_all, old_map_clear, old_map_smooth, old_map_nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = old_nodes, _nbins = 12*12, 
        occu_time = occu_time_old, Ms = Ms, is_silent = trace['node 4']['SilentNeuron'])

    old_t_total = np.nansum(occu_time_old) / 1000
    old_t_nodes_frac = occu_time_old / 1000 / (old_t_total + 1E-6)
    SI = calc_SI(Spikes, old_map_clear, old_t_total, old_t_nodes_frac)

    trace['node 4']['old_map_all'] = cp.deepcopy(old_map_all)
    trace['node 4']['old_map_clear'] = cp.deepcopy(old_map_clear)
    trace['node 4']['old_map_smooth'] = cp.deepcopy(old_map_smooth)
    trace['node 4']['old_map_nanPos'] = cp.deepcopy(old_map_nanPos)
    trace['node 4']['old_nodes'] = cp.deepcopy(old_nodes)
    trace['node 4']['occu_time_old'] = cp.deepcopy(occu_time_old)
    trace['node 4']['old_t_total'] = cp.deepcopy(old_t_total)
    trace['node 4']['old_t_nodes_frac'] = cp.deepcopy(old_t_nodes_frac)
    trace['node 4']['old_SI'] = cp.deepcopy(SI)

    if not is_draw:
        return trace

    n_neuron = trace['n_neuron']
    save_loc = join(trace['p'], 'OldMap')
    mkdir(save_loc)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=2, nrows=2, figsize=(6*2, 4.5*2))

    ax1 = Clear_Axes(ax1)
    ax2 = Clear_Axes(ax2)
    ax3 = Clear_Axes(ax3)
    ax4 = Clear_Axes(ax4)
    ax5 = Clear_Axes(ax5)
    _ = Clear_Axes(ax6)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    ax5.set_aspect('equal')

    DrawMazeProfile(axes=ax1, maze_type=trace['maze_type'], nx = 12, color = 'black')
    DrawMazeProfile(axes=ax2, maze_type=trace['maze_type'], nx = 12, color = 'black')
    DrawMazeProfile(axes=ax3, maze_type=trace['maze_type'], nx = 12, color = 'black')
    DrawMazeProfile(axes=ax4, maze_type=trace['maze_type'], nx = 12, color = 'black')
    DrawMazeProfile(axes=ax5, maze_type=trace['maze_type'], nx = 12, color = 'black')
    ax1.axis([-0.6,11.6,-0.6,11.6])
    ax2.axis([-0.6,11.6,-0.6,11.6])
    ax3.axis([-0.6,11.6,-0.6,11.6])
    ax4.axis([-0.6,11.6,-0.6,11.6])
    ax5.axis([-0.6,11.6,-0.6,11.6])

    for i in tqdm(range(n_neuron)):
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()

        ax1, im1, cbar1 = RateMapAxes(
            ax=ax1, 
            nx=12,
            content=trace['node 0']['old_map_clear'][i, :]+mask0,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 0']['old_SI'][i], 3))
        )
        color = 'black' if trace['node 0']['is_placecell'][i] == 0 else 'red'
        ax1.set_title('SI='+str(round(trace['node 0']['old_SI'][i], 3)), color=color)

        ax2, im2, cbar2 = RateMapAxes(
            ax=ax2, 
            nx=12,
            content=trace['node 1']['old_map_clear'][i, :]+mask1,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 1']['old_SI'][i], 3))
        )
        color = 'black' if trace['node 1']['is_placecell'][i] == 0 else 'red'
        ax2.set_title('SI='+str(round(trace['node 1']['old_SI'][i], 3)), color=color)

        ax3, im3, cbar3 = RateMapAxes(
            ax=ax3, 
            nx=12,
            content=trace['node 2']['old_map_clear'][i, :]+mask2,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 2']['old_SI'][i], 3))
        )
        color = 'black' if trace['node 2']['is_placecell'][i] == 0 else 'red'
        ax3.set_title('SI='+str(round(trace['node 2']['old_SI'][i], 3)), color=color)

        ax4, im4, cbar4 = RateMapAxes(
            ax=ax4, 
            nx=12,
            content=trace['node 3']['old_map_clear'][i, :]+mask3,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 3']['old_SI'][i], 3))
        )
        color = 'black' if trace['node 3']['is_placecell'][i] == 0 else 'red'
        ax4.set_title('SI='+str(round(trace['node 3']['old_SI'][i], 3)), color=color)

        ax5, im5, cbar5 = RateMapAxes(
            ax=ax5, 
            nx=12,
            content=trace['node 4']['old_map_clear'][i, :]+mask3,
            maze_type=trace['maze_type'],
            is_plot_maze_walls=False,
            is_colorbar=True,
            title='SI='+str(round(trace['node 4']['old_SI'][i], 3))
        )
        color = 'black' if trace['node 4']['is_placecell'][i] == 0 else 'red'
        ax5.set_title('SI='+str(round(trace['node 4']['old_SI'][i], 3)), color=color)
        
        plt.savefig(join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(join(save_loc, str(i+1)+'.svg'), dpi = 600)

        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax4.invert_yaxis()
        ax5.invert_yaxis()


        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
        cbar4.remove()
        cbar5.remove()
        im1.remove()
        im2.remove()
        im3.remove()
        im4.remove()
        im5.remove()

    return trace

def PVCorrelation(trace: dict) -> dict:
    if trace['start_from'] == 'correct':
        paths = {
            0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            1: np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            2: np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            3: np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
        }
        areas = DSP_correct_graph1
    else:
        paths = {
            0: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            1: np.array([8,7,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            2: np.array([93,105,106,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            3: np.array([135,134,133,121,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
            4: np.array([1,13,14,26,27,15,3,4,5,6,18,17,29,30,31,19,20,21,9,10,11,12,24,23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95,94,82,81,80,92,104,103,91,90,78,79,67,55,54,66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97,109,110,122,123,111,112,100,99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144], dtype=np.int64),
        }
        areas = DSP_incorrect_graph1

    PVCorr = np.zeros((4, 48**2), dtype = np.float64)

    idx = np.where((trace['node 0']['is_placecell'] == 1)|(trace['node 1']['is_placecell'] == 1))[0]

    mask1, mask2, mask3, mask4 = np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan, np.zeros(48**2, np.float64)*np.nan
    mask1[get_son_area(paths[1])-1] = 0
    mask2[get_son_area(paths[2])-1] = 0
    mask3[get_son_area(paths[3])-1] = 0
    mask4[get_son_area(paths[4])-1] = 0
    mask_list = [mask1, mask2, mask3, mask4]

    for i in range(48**2):
        PVCorr[0, i], _ = pearsonr(trace['node 0']['smooth_map_all'][idx, i], trace['node 1']['smooth_map_all'][idx, i])
        PVCorr[1, i], _ = pearsonr(trace['node 0']['smooth_map_all'][idx, i], trace['node 2']['smooth_map_all'][idx, i])
        PVCorr[2, i], _ = pearsonr(trace['node 0']['smooth_map_all'][idx, i], trace['node 3']['smooth_map_all'][idx, i])
        PVCorr[3, i], _ = pearsonr(trace['node 0']['smooth_map_all'][idx, i], trace['node 4']['smooth_map_all'][idx, i])

    save_loc = os.path.join(trace['p'], 'PV Correlation')
    mkdir(save_loc)

    print(PVCorr[0, :])
    fig = plt.figure(figsize=(6, 4.5))
    ax = Clear_Axes(plt.axes())
    ax.set_aspect('equal')
    ax = DrawMazeProfile(axes=ax, maze_type=trace['maze_type'], color = 'black')
    ax.invert_yaxis()

    for i in tqdm(range(4)):
        im = ax.imshow(np.reshape(PVCorr[i, :]+mask_list[i], [48,48]), cmap='jet', vmin = 0, vmax = 1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("PV Correlation")

        plt.savefig(os.path.join(save_loc, f'Node {i+1}.png'), dpi = 600)
        plt.savefig(os.path.join(save_loc, f'Node {i+1}.svg'), dpi = 600)

        cbar.remove()
        im.remove()
    

    trace['PVCorrelation'] = PVCorr
    return trace


def run_all_mice_DLC(i: int, f: pd.DataFrame, work_flow: str, 
                     v_thre: float = 2.5, cam_degree = 0, speed_sm_args = {}):#p = None, folder = None, behavior_paradigm = 'CrossMaze'):
    t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])
    session = int(f['session'][i])
    start_from = str(f['Start From'][i])

    if behavior_paradigm not in ['DSPMaze']:
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

    beg, end = LapSplit(trace, trace['paradigm'])
    lap_type = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg, start_from=start_from)
    print(lap_type)
    unique_type = np.unique(lap_type)

    plot_split_trajectory(trace)


    print("    B. Calculate putative spikes and correlated location from deconvolved signal traces. Delete spikes that evoked at interlaps gap and those spikes that cannot find it's clear locations.")
    # Calculating Spikes, than delete the interlaps frames
    Spikes_original = SpikeType(Transients = DeconvSignal, threshold = 3)
    spike_num_mon1 = np.nansum(Spikes_original, axis = 1) # record temporary spike number
    # Calculating correlated spike nodes
    spike_nodes_original = SpikeNodes(Spikes = Spikes_original, ms_time = ms_time, 
                behav_time = trace['correct_time'], behav_nodes = trace['correct_nodes'])

    # Filter the speed
    if 'smooth_speed' not in trace.keys():
        behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
        trace['smooth_speed'] = uniform_smooth_speed(behav_speed, **speed_sm_args)

    ms_speed = calc_ms_speed(behav_speed=trace['smooth_speed'], behav_time=trace['correct_time'], 
                             ms_time=ms_time)

    # Delete NAN value in spike nodes
    print("      - Delete NAN values in data.")
    idx = np.where(np.isnan(spike_nodes_original) == False)[0]
    Spikes = cp.deepcopy(Spikes_original[:,idx])
    spike_nodes = cp.deepcopy(spike_nodes_original[idx])
    ms_time_behav = cp.deepcopy(ms_time[idx])
    ms_speed_behav = cp.deepcopy(ms_speed[idx])
    dt = np.append(np.ediff1d(ms_time_behav), 33)
    dt[np.where(dt >= 100)[0]] = 100
    
    # Filter the speed
    print(f"     - Filter spikes with speed {v_thre} cm/s.")
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
    spike_num_mon2 = np.nansum(Spikes, axis = 1)

    # Delete InterLap Spikes
    n_neuron = Spikes.shape[0]
    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 2, _range = 7, nx = 48)
    print("      - Calculate and shuffle sub-ratemap")
    spike_num_mon3 = np.zeros(n_neuron, dtype=np.int64)
    for n in unique_type:
        print(f"        Node {n} -----------------------------------------------------------")
        lap_idx = np.where(lap_type == n)[0]
        idx = split_calcium_data(lap_idx=lap_idx, trace=trace, ms_time=ms_time_behav)
        spike_num_mon3 += np.nansum(Spikes[:, idx], axis = 1)

        trace['node '+str(n)] = calc_rate_map_properties(
            trace['maze_type'],
            ms_time_behav[idx],
            Spikes[:, idx],
            spike_nodes[idx],
            ms_speed_behav[idx],
            dt[idx],
            Ms,
            trace['p'],
            kwargs = {'file_name': 'Place cell shuffle [trans]'}
        )

    plot_spike_monitor(spike_num_mon1, spike_num_mon2, spike_num_mon3, save_loc = os.path.join(trace['p'], 'behav'))

    print("    C. Calculating firing rate for each neuron and identified their place fields (those areas which firing rate >= 50% peak rate)")
    # Set occu_time <= 50ms spatial bins as nan to avoid too big firing rate

    trace_ms = {'Spikes_original':Spikes_original, 'spike_nodes_original':spike_nodes_original, 'ms_speed_original': ms_speed, 'RawTraces':RawTraces,'DeconvSignal':DeconvSignal,
                'ms_time':ms_time, 'ms_folder':folder, 'speed_filter_results': spf_results, 'n_neuron': Spikes_original.shape[0], 'Ms':Ms, 'start_from': start_from}

    trace.update(trace_ms)
    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)

    print("    Plotting:")
    print("      1. Ratemap")
    RateMap(trace)

    print("      2. Tracemap")
    TraceMap(trace)

    print("      4. Oldmap")
    trace = OldMap(trace, is_draw=False)

    path = os.path.join(p,"trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    
    print("      5. PeakCurve")
    trace = PartialPeakCurve(trace)
    
    print("      6. Loc-time curve")
    trace = LocTimeCurve(trace)

    print("      7. Population Vector Correlation")
    trace = PVCorrelation(trace)

    trace['processing_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(trace['p'],"trace.pkl")
    print("    ",path)
    with open(path, 'wb') as f:
        pickle.dump(trace, f)
    print("    Every file has been saved successfully!")

    t2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print(t1,'\n',t2)

if __name__ == '__main__':
    with open(r"G:\YSY\Dsp_maze\10209\20230601\session 1\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    OldMap(trace, is_draw=False)
    PVCorrelation(trace)
    PartialPeakCurve(trace)
    LocTimeCurve(trace)