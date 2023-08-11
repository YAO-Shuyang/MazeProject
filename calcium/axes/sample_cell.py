import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from os.path import join

from mylib import RateMapAxes, TraceMapAxes, LinearizedRateMapAxes, LocTimeCurveAxes
from mylib.maze_utils3 import spike_nodes_transform, SF2FF, get_spike_frame_label
from mylib.behavior.behavevents import BehavEvents
from mylib.calcium.field_criteria import GetPlaceField
from mylib.field.in_field import set_range
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from mylib.maze_graph import correct_paths


# trace, i, save_loc: str, file_name: str
def SampleCellAxes(
    smooth_map: np.ndarray,
    old_map: np.ndarray,
    behav_pos: np.ndarray,
    behav_time: np.ndarray,
    behav_nodes: np.ndarray,
    ms_time: np.ndarray,
    spikes: np.ndarray,
    maze_type: int,
    place_fields: dict,
    trace: dict,
    i: int,
    save_loc: str,
    file_name: str
):
    fig = plt.figure(figsize=(5.5, 4))
    grid = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[7, 4])

    ax1 = fig.add_subplot(grid[0:2, 0])   # First two rows, first column
    ax2 = fig.add_subplot(grid[2:4, 0])   # Last two rows, first column
    ax3 = fig.add_subplot(grid[0, 1])     # First row, second column
    ax4 = fig.add_subplot(grid[1:, 1])    # Second row onwards, second column
    
    _, im, cbar = RateMapAxes(
        ax=ax1,
        content=smooth_map,
        maze_type=maze_type,
        title=str(round(trace['SI_all'][i], 4))
    )
    cbar.outline.set_visible(False)
    ax1.set_aspect("equal")
    ax1.axis([-1, 48, 48, -1])   
     
    TraceMapAxes(
        ax=ax2,
        trajectory=cp.deepcopy(behav_pos),
        behav_time=behav_time,
        spikes=spikes,
        spike_time=ms_time,
        maze_type=maze_type,
        maze_kwargs={'color':'brown'},
        traj_kwargs={'linewidth': 0.5},
        markersize=2
    )
    ax2.set_aspect("equal")
    ax2.axis([-1, 48, 48, -1])

    MTOP = gaussian_smooth_matrix1d(1000, window = 20, sigma=3, folder=0.1)
    CP = cp.deepcopy(correct_paths[int(maze_type)])
    LinearizedRateMapAxes(
        ax=ax3,
        content=old_map,
        maze_type=maze_type,
        M=MTOP,
        linewidth=0.8
    )
    ax3.set_xlim([0, len(CP)+1])
    y_max = np.nanmax(old_map)

    lef, rig = np.zeros(len(place_fields.keys()), dtype=np.float64), np.zeros(len(place_fields.keys()), dtype=np.float64)
    colors = sns.color_palette("rainbow", 10) if lef.shape[0] < 8 else sns.color_palette("rainbow", lef.shape[0]+2)
    for j, k in enumerate(place_fields.keys()):
        father_field = SF2FF(place_fields[k])
        lef[j], rig[j] = set_range(maze_type=maze_type, field=father_field)
    lef = np.sort(lef + 0.5)
    rig = np.sort(rig + 1.5)
        
    for k in range(lef.shape[0]):
        ax3.plot([lef[k], rig[k]], [-y_max*0.09, -y_max*0.09], color = colors[k+2])
        ax4.fill_betweenx(y=[0, np.nanmax(behav_time)/1000], x1=lef[k], x2=rig[k], color=colors[k+2], alpha=0.5, edgecolor=None)
        field = CP[int(lef[k]-0.5):int(rig[k]-0.5)]
        for fd in field:
            y, x = (fd-1)//12*4-0.5, (fd-1)%12*4-0.5
            ax2.fill_betweenx([y, y+4], x1=x, x2=x+4, color = colors[k+2], alpha=0.5, edgecolor=None)
    
    frame_labels = get_spike_frame_label(
        ms_time=cp.deepcopy(behav_time), 
        spike_nodes=cp.deepcopy(behav_nodes),
        trace=trace, 
        behavior_paradigm='CrossMaze',
        window_length = 1
    )
    
    indices = np.where(frame_labels==1)[0]
    LocTimeCurveAxes(
        ax=ax4,
        behav_time=behav_time[indices],
        spikes=spikes,
        spike_time=ms_time,
        maze_type=maze_type,
        behav_nodes=cp.deepcopy(behav_nodes[indices]),
        line_kwargs={'markeredgewidth': 0, 'markersize': 0.6, 'color': 'black'},
        bar_kwargs={'markeredgewidth': 0.6, 'markersize': 3}
    )
    ax4.set_xlim([0, len(CP)+1])
    plt.tight_layout()
    plt.savefig(join(save_loc, file_name+'.png'), dpi=600)
    plt.savefig(join(save_loc, file_name+'.svg'), dpi=600)
    plt.close()