import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from mylib.maze_utils3 import Clear_Axes
from mylib.maze_graph import NRGs, correct_paths

def LocTimeCurveAxes(
    ax: Axes,
    behav_time: np.ndarray,
    spikes: np.ndarray,
    spike_time: np.ndarray,
    maze_type: int,
    behav_nodes: np.ndarray | None = None,
    given_x: np.ndarray | None = None,
    title: str = "",
    title_color: str = "black",
    is_invertx: bool = False,
    line_kwargs: dict = {'markeredgewidth': 0, 'markersize': 1, 'color': 'black'},
    bar_kwargs: dict = {'markeredgewidth': 1, 'markersize': 4, 'color': 'red'},
    is_dotted_line: bool = False,
    is_include_incorrect_paths: bool = False,
    is_ego: bool = False,
    NRG: dict = NRGs
) -> tuple[Axes, list, list]:

    ax = Clear_Axes(axes=ax, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.set_aspect("auto")

    if given_x is None:
        assert behav_nodes is not None
        linearized_x = np.zeros_like(behav_nodes, np.float64)
        graph = NRG[int(maze_type)] if is_ego == False else NRG

        try:
            linearized_x = graph[behav_nodes.astype(np.int64) - 1]
            #for i in range(behav_nodes.shape[0]):
            #    linearized_x[i] = graph[int(behav_nodes[i])]
        except:
            print(behav_nodes.shape)
            
        linearized_x = linearized_x + np.random.rand(behav_nodes.shape[0]) - 0.5
    else:
        linearized_x = given_x
        
    if maze_type == 1:
        thre = 111
    elif maze_type == 2:
        thre = 101
    elif maze_type == 3:
        thre = 144
    else:
        raise ValueError(f"Invalid maze type: {maze_type}")
    
    spike_burst_time = spike_time[np.where(spikes == 1)[0]]
    spike_loc_id = np.zeros_like(spike_burst_time, dtype=np.int64)

    for i, t in enumerate(spike_burst_time):
        try:
            spike_loc_id[i] = np.where(behav_time>=t)[0][0]
        except:
            spike_loc_id[i] = np.where(behav_time<t)[0][-1]

    x_spikes = linearized_x[spike_loc_id]
    t_spikes = behav_time[spike_loc_id]

    t_max = int(np.nanmax(behav_time)/1000)
    
    if is_include_incorrect_paths == False:
        idx = np.where(linearized_x < thre+0.5)[0]
        linearized_x = linearized_x[idx]
        behav_time = behav_time[idx]

    if is_dotted_line:
        a = ax.plot(linearized_x, behav_time/1000, 'o', **line_kwargs)
    else:
        dx = np.ediff1d(linearized_x)
        idx = np.where((dx < -10) | ((dx > 0) & (linearized_x[:-1] > thre-0.5)) | (dx > 10))[0]
        linearized_x = np.insert(linearized_x, idx+1, np.nan)
        behav_time = np.insert(behav_time.astype(float), idx+1, np.nan)
        a = ax.plot(linearized_x, behav_time/1000, **line_kwargs)
        
    b = ax.plot(x_spikes, t_spikes/1000, '|', **bar_kwargs)

    ax.set_title(title, color=title_color)
    ax.set_xticks([1, len(correct_paths[int(maze_type)])/2, len(correct_paths[int(maze_type)])], labels = ['start', 'correct track', 'end'])
    ax.set_xlim([0, 145])
    ax.set_yticks([0, t_max])
    ax.set_ylim([0, t_max])
    ax.set_ylabel("Time / s")

    if is_invertx:
        ax.invert_xaxis()

    return ax, a, b


def LocTimeCurveAxes_LineBackground(
    ax: Axes,
    behav_time: np.ndarray,
    spikes: np.ndarray,
    spike_time: np.ndarray,
    maze_type: int,
    behav_nodes: np.ndarray | None = None,
    given_x: np.ndarray | None = None,
    title: str = "",
    title_color: str = "black",
    is_invertx: bool = False,
    line_kwargs: dict = {'markeredgewidth': 0, 'markersize': 1, 'color': 'black'},
    bar_kwargs: dict = {'markeredgewidth': 1, 'markersize': 4}
) -> tuple[Axes, list, list]:

    ax = Clear_Axes(axes=ax, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.set_aspect("auto")

    if given_x is None:
        assert behav_nodes is not None
        linearized_x = np.zeros_like(behav_nodes, np.float64)
        graph = NRG[int(maze_type)]

        for i in range(behav_nodes.shape[0]):
            linearized_x[i] = graph[int(behav_nodes[i])]
    
        linearized_x = linearized_x + np.random.rand(behav_nodes.shape[0]) - 0.5
    else:
        linearized_x = given_x
    
    spike_burst_time = spike_time[np.where(spikes == 1)[0]]
    spike_loc_id = np.zeros_like(spike_burst_time, dtype=np.int64)

    for i, t in enumerate(spike_burst_time):
        try:
            spike_loc_id[i] = np.where(behav_time>=t)[0][0]
        except:
            spike_loc_id[i] = np.where(behav_time<t)[0][-1]

    x_spikes = linearized_x[spike_loc_id]
    t_spikes = behav_time[spike_loc_id]

    t_max = int(np.nanmax(behav_time)/1000)

    a = ax.plot(linearized_x, behav_time/1000, 'o', **line_kwargs)
    b = ax.plot(x_spikes, t_spikes/1000, '|', color='red', **bar_kwargs)

    ax.set_title(title, color=title_color)
    ax.set_xticks([1, len(correct_paths[int(maze_type)])/2, len(correct_paths[int(maze_type)])], labels = ['start', 'correct track', 'end'])
    ax.set_xlim([0, 145])
    ax.set_yticks([0, t_max])
    ax.set_ylim([0, t_max])
    ax.set_ylabel("Time / s")

    if is_invertx:
        ax.invert_xaxis()

    return ax, a, b