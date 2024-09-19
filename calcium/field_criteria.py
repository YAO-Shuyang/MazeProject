import numpy as np
import copy as cp
from tqdm import tqdm
from mylib.maze_utils3 import maze_graphs, spike_nodes_transform, GetDMatrices
from collections import deque
from numba import jit
import time

def convert_graph_to_matrix(graph: dict):
    num_nodes = len(graph.keys())
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for k in graph.keys():
        for i in graph[k]:
            matrix[i-1, k-1] = matrix[k-1, i-1] = 1  # Assuming node numbering starts from 1
    return matrix

def get_field_thre(thre_type: int, peak_rate: float, parameter: float = 0.5) -> float:
    """
    get_field_thre: to get the threshold of the map

    Parameters
    ----------
    thre_type : int
        The kind of threshold that you wanted to select. Type 1 is set 
          an relative threshold in terms of how many fold of (a fraction 
          of) the peak rate of the map. Type 2 is to set an absolute 
          threshold instead of a relative one.
    peak_rate: float
        The peak event rate of the given map. The type 1 threshold will
          be calculated according to the peak_rate that you provided.
    parameter : float, optional
        The fraction or absolute threshold you provided, by default 0.5

    Returns
    -------
    float
        The threshold of this map according to your choice.
    """
    if thre_type == 1:
        assert parameter >= 0 and parameter <= 1
        return peak_rate*parameter
    
    elif thre_type == 2:
        assert parameter >= 0
        return parameter

# generate all subfield. ============================================================================
# place field analysis, return a dict contatins all field. If you want to know the field number of a certain cell, you only need to get it by use 
# len(trace['place_field_all'][n].keys())
def GetPlaceField(
    trace: dict, 
    n: int, 
    thre_type: int, 
    nx: int = 48, 
    parameter: float = 0.5, 
    events_num_crit: int = 10, 
    need_events_num: bool = True, 
    smooth_map: np.ndarray | None = None,
    split_thre: float = 0.2,
    reactivate_num: int = 5
) -> dict:
    """
    GeneratePlaceField: to generate all fields of the given cell.

    Parameters
    ----------
    maze_type : int
        The maze type. Valid values belong to {0, 1, 2, 3}
    nx : int, optional
        The bin size`, by default 48
    smooth_map : np.ndarray, optional
        The rate map you provided, typically the map that has been smoothed, by default None

    Returns
    -------
    dict
        The field dictionary.
    """
    if smooth_map is None:
        smooth_map = trace['smooth_map_all'][n, :]
    maze_type = trace['maze_type']

    # rate_map should be one without NAN value. Use function clear_NAN(rate_map_all) to process first.
    r_max = np.nanmax(smooth_map)
    thre = get_field_thre(peak_rate=r_max, thre_type = thre_type, parameter = parameter)
    all_fields = np.where(smooth_map >= thre)[0]+1
    search_set = np.array([], dtype=np.int64)
    All_field = {}
    
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
        
    D = GetDMatrices(trace['maze_type'], nx=nx)
    #print("Initial: ", t2-t1)
    while len(np.setdiff1d(all_fields, search_set))!=0:
        #t3 = time.time()
        #print("  One iteration start:")
        diff = np.setdiff1d(all_fields, search_set)
        diff_idx = np.argmax(smooth_map[diff-1])
        
        
        subfield = _field(diff = diff, diff_idx = diff_idx, graph = graph,split_thre=split_thre, smooth_map = smooth_map)
        #t4 = time.time()
        #print("     Get raw field: ", t4-t3)
        IS_QUALIFIED_FIELD, retain_fields = field_quality_control(spike_nodes=trace['spike_nodes'], 
                                                                  spikes=trace['Spikes'][n, :],
                                                                  ms_time=trace['ms_time_behav'], field_bins=subfield, 
                                                                  events_num_crit = events_num_crit,
                                                                  reactivate_num = reactivate_num)
        #t5 = time.time()
        #print("     Field quality control: ", t5-t4)
        if IS_QUALIFIED_FIELD:
            submap = smooth_map[retain_fields-1]
            peak_idx = np.argmax(submap)
            peak_loc = retain_fields[peak_idx]
            peak = np.max(submap)
        
            All_field[peak_loc] = retain_fields
        #t6 = time.time()
        #print("     Check quality: ", t6-t5, end = '\n\n')

        search_set = np.concatenate([search_set, subfield])
    
    # Sort the fields by their distance to the start point
    res = {}
    for key in sorted(All_field.keys(), key = lambda kv:D[kv-1, 0]):
        res[key] = All_field[key]

    return res
         
def _field(
    diff: list | np.ndarray, 
    diff_idx: int, 
    graph: dict,
    split_thre: float = 0.2,
    smooth_map: np.ndarray | None = None
) -> np.ndarray:
    # Identify single field from all fields.
    
    point = diff[diff_idx]
    peak_rate = smooth_map[point-1]
    Area = [point]
    StepExpand = deque([point])
    while len(StepExpand) > 0:
        k = StepExpand[0]
        surr = graph[k]
        for j in surr:
            if j in diff and j not in Area and (smooth_map[j-1] < smooth_map[k-1] or smooth_map[j-1] >= split_thre*peak_rate):
                StepExpand.append(j)
                Area.append(j)
                
        StepExpand.popleft()
        
    return np.array(Area, dtype=np.int64)

from scipy.stats import pearsonr
def field_quality_control(
    spike_nodes: np.ndarray, 
    spikes: np.ndarray, 
    ms_time: np.ndarray,
    field_bins: np.ndarray, 
    events_num_crit: int = 10,
    reactivate_num: int = 5
) -> bool:
    field_bins_expanded = field_bins[:, np.newaxis]
    matches = (spike_nodes == field_bins_expanded) & (spikes == 1)
    total_indices = np.where(matches.any(axis=0))[0]
    events_num = len(total_indices)

    # Test Events Number
    if events_num <= events_num_crit:
        return False, None
    
    if len(field_bins) == 1:
        return True, field_bins
    
    # Test the number of laps it distributed.
    dt = np.ediff1d(ms_time[total_indices])
    
    if len(np.where(dt >= 10000)[0]) < reactivate_num:
        return False, None
    else:
        return True, field_bins
    
# get all cell's place field
def place_field(
    trace: dict, 
    thre_type: int = 2, 
    parameter: float = 0.4, 
    events_num_crit: int = 10, 
    need_events_num: bool = True, 
    split_thre: float = 0.2,
    reactivate_num: int = 5
):
    place_field_all = []
    smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
    maze_type = trace['maze_type']
    for k in tqdm(range(trace['n_neuron'])):
        a_field = GetPlaceField(
            trace=trace, 
            n=k, 
            thre_type=thre_type, 
            parameter=parameter, 
            events_num_crit=events_num_crit, 
            need_events_num=need_events_num,
            split_thre=split_thre,
            reactivate_num=reactivate_num
        )
        place_field_all.append(a_field)
    print("    Place field has been generated successfully.")
    return place_field_all

def get_field_number(trace: dict, key: str = 'place_field_all'):
    num = np.zeros(trace['n_neuron'], dtype=np.int64)
    for i in range(trace['n_neuron']):
        num[i] = len(trace[key][i].keys())
    return num

def place_field_dsp(
    trace: dict, 
    thre_type: int = 2, 
    parameter: float = 0.4, 
    events_num_crit: int = 10, 
    need_events_num: bool = True, 
    split_thre: float = 0.2,
    reactivate_num: int = 5
):
    G = maze_graphs[(1, 48)]
    place_field_all = []
    smooth_map_all = np.zeros((trace['n_neuron'], 10, 2304))
    trace['Spikes'] = np.concatenate([trace[f'node {i}']['Spikes'] for i in range(10)], axis=1)
    trace['spike_nodes'] = np.concatenate([trace[f'node {i}']['spike_nodes'] for i in range(10)])
    trace['ms_time_behav'] = np.concatenate([trace[f'node {i}']['ms_time_behav'] for i in range(10)])
    for i in range(10):
        smooth_map_all[:, i, :] = trace[f'node {i}']['smooth_map_all']
        for n in range(trace['n_neuron']):
            idx = np.where(trace[f'node {i}']['Spikes'][n, :] == 1)[0]
            spike_bins = np.unique(trace[f'node {i}']['spike_nodes'][idx])
            extended_bins = []
            for j in spike_bins:
                extended_bins += G[j]
        
            extended_bins = np.unique(extended_bins)
            remove_bins = np.setdiff1d(np.arange(1, 2305), extended_bins)
            smooth_map_all[n, i, remove_bins-1] = 0
            
    trace['smooth_map_all'] = np.sum(smooth_map_all, axis=1)
    
    maze_type = trace['maze_type']
    for k in tqdm(range(trace['n_neuron'])):
        a_field = GetPlaceField(
            trace=trace, 
            n=k, 
            thre_type=thre_type, 
            parameter=parameter, 
            events_num_crit=events_num_crit, 
            need_events_num=need_events_num,
            split_thre=split_thre,
            reactivate_num=reactivate_num
        )
        place_field_all.append(a_field)
    print("    Place field has been generated successfully.")

    del trace['Spikes']
    del trace['spike_nodes']
    del trace['ms_time_behav']

    return place_field_all