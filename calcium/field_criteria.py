import numpy as np
import copy as cp
from tqdm import tqdm
from mylib.maze_utils3 import maze_graphs, spike_nodes_transform, GetDMatrices

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
def get_all_fields(
    trace: dict, 
    n: int, 
    thre_type: int, 
    nx: int = 48, 
    parameter: float = 0.5, 
    events_num_crit: int = 10, 
    need_events_num: bool = True, 
    smooth_map: np.ndarray | None = None,
    split_thre: float = 0.2
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
    
    D = GetDMatrices(trace['maze_type'], nx=nx)
    search_area = cp.deepcopy(all_fields)
    
    while search_area.shape[0] != 0:
        diff_idx = np.argmax(smooth_map[search_area-1])
        subfields = extract_single_field(diff = search_area, diff_idx = diff_idx, maze_type = maze_type, 
                          nx = nx, thre=thre, split_thre=split_thre, smooth_map = smooth_map)
        for k in subfields.keys():
            IS_QUALIFIED_FIELD, retain_fields = field_quality_control(trace=trace, n=n, field_bins=subfields[k], 
                                                                      events_num_crit = events_num_crit)

            if IS_QUALIFIED_FIELD:
                submap = smooth_map[retain_fields-1]
                peak_idx = np.argmax(submap)
                peak_loc = retain_fields[peak_idx]
                peak = np.max(submap)
        
                All_field[peak_loc] = retain_fields
        

            search_set = np.concatenate([search_set, subfields[k]])
            
        search_area = np.setdiff1d(all_fields, search_set)
    
    # Sort the fields by their distance to the start point
    res = {}
    for key in sorted(All_field.keys(), key = lambda kv:D[kv-1, 0]):
        res[key] = All_field[key]

    return res

def broad_first_search_from_center(
    peak_bin: int,
    graph: dict,
    search_area: np.ndarray,
    smooth_map: np.ndarray,
    thre: float
):  
    is_saddle_exist = False
    saddle_points = []
    
    MaxStep = 300
    step = 0
    Area = [peak_bin]
    StepExpand = {0: [peak_bin]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:   
                # If j is not included yet and we do not meet a saddle point during broad first search
                if j in search_area and j not in Area and (smooth_map[j-1] <= smooth_map[k-1] or smooth_map[j-1] >= thre):
                    StepExpand[step+1].append(j)
                    Area.append(j)
                
                if smooth_map[j-1] < thre and smooth_map[j-1] > smooth_map[k-1]:
                    # Meet a saddle point
                    is_saddle_exist = True
                    saddle_points.append(j)
                    
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
        
        step += 1
        
    return np.array(Area, dtype=np.int64), is_saddle_exist, [saddle_points, [peak_bin] * len(saddle_points)]


def broad_first_search_from_saddle(
    saddle_bin: int,
    graph: dict,
    search_area: np.ndarray,
    smooth_map: np.ndarray,
    split_thre: float
): 
    peak_bin, peak_rate = saddle_bin, smooth_map[saddle_bin-1]
    
    # First Step: Find the peak:
    MaxStep = 300
    step = 0
    while step <= MaxStep:
        surr = graph[peak_bin]
        if max(peak_rate, np.max(smooth_map[np.array(surr)-1])) > peak_rate:
            peak_bin = surr[np.argmax(smooth_map[np.array(surr)-1])]
            peak_rate = smooth_map[peak_bin-1]
        else:
            break
        
        step += 1
    
    is_saddle_exist = False
    saddle_points = []
    
    step = 0
    Area = [peak_bin]
    StepExpand = {0: [peak_bin]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:   
                # If j is not included yet and we do not meet a saddle point during broad first search
                if j in search_area and j not in Area and (smooth_map[j-1] <= smooth_map[k-1] or smooth_map[j-1] >= split_thre*peak_rate):
                    StepExpand[step+1].append(j)
                    Area.append(j)
                
                if smooth_map[j-1] < split_thre*peak_rate and smooth_map[j-1] > smooth_map[k-1]:
                    # Meet a saddle point
                    is_saddle_exist = True
                    saddle_points.append(j)
                    
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
        
        step += 1
        
    return np.array(Area, dtype=np.int64), is_saddle_exist, [saddle_points, [peak_bin] * len(saddle_points)]

def extract_single_field(
    diff: list | np.ndarray, 
    diff_idx: int, 
    maze_type: int, 
    nx: int = 48, 
    thre: float = 0.5, 
    split_thre: float = 0.2,
    smooth_map: np.ndarray | None = None
) -> dict:
    # Identify single field from all fields.
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
    
    peak_bin = diff[diff_idx]
    peak_rate = smooth_map[peak_bin-1]

    subfields = {}
    raw_field, is_saddle_exist, saddle_points = broad_first_search_from_center(
        peak_bin = peak_bin, 
        graph = graph, 
        search_area = diff, 
        smooth_map = smooth_map, 
        thre = split_thre*peak_rate
    )
    subfields[peak_bin] = raw_field
    
    search_area = np.setdiff1d(diff, raw_field)
    
    if is_saddle_exist:
        while len(saddle_points[0]) > 0:
            # Get a field from saddle point
            raw_field, is_saddle_exist, sub_saddle_points = broad_first_search_from_saddle(
                saddle_bin = saddle_points[0][0], 
                graph = graph, 
                search_area = search_area, 
                smooth_map = smooth_map, 
                split_thre = split_thre
            )
            
            # Identify if the saddle is really lower than split_threshold * both peaks
            if smooth_map[saddle_points[0][0]-1] > split_thre*smooth_map[raw_field[0]-1]: # This saddle is not a real saddle
                # Merge two fields
                if smooth_map[raw_field[0]-1] > smooth_map[saddle_points[1][0]-1]:
                    # Newly generated field becomes major
                    merged_field = np.concatenate([raw_field, subfields[saddle_points[1][0]]])
                    del subfields[saddle_points[1][0]]
                    subfields[merged_field[0]] = merged_field
                    
                    # Update all the prior peaks index
                    for i, p in enumerate(saddle_points[1]):
                        if p == saddle_points[1][0]:
                            saddle_points[1][i] = merged_field[0]
                else:
                    subfields[saddle_points[1][0]] = np.concatenate([subfields[saddle_points[1][0]], raw_field])
                    if is_saddle_exist:
                        # Update all the prior peaks index
                        sub_saddle_points[1] = [saddle_points[1][0]] * len(sub_saddle_points[1])
                        saddle_points = [saddle_points[0] + sub_saddle_points[0], saddle_points[1] + sub_saddle_points[1]]
            
            # Add new saddle        
            search_area = np.setdiff1d(search_area, raw_field)
            
            if len(saddle_points[0]) == 1:
                break
            else:
                saddle_points = [saddle_points[0][1:], saddle_points[1][1:]]

    return subfields

from scipy.stats import pearsonr
def field_quality_control(trace: dict, n: int, field_bins: np.ndarray | list, events_num_crit: int = 10, stability_thre: float = -1) -> bool:
    spike_nodes = trace['spike_nodes']
    total_indices = np.concatenate([np.where((spike_nodes == i)&(trace['Spikes'][n, :] == 1))[0] for i in field_bins])
    events_num = len(total_indices)

    # Test Events Number
    if events_num <= events_num_crit:
        return False, None
    
    if len(field_bins) == 1:
        return True, field_bins
    
    # Test the number of laps it distributed.
    dt = np.ediff1d(trace['ms_time_behav'][total_indices])
    
    if len(np.where(dt >= 10000)[0]) < 5:
        return False, None
    else:
        return True, field_bins
    
# get all cell's place field
def place_field(trace: dict, thre_type: int = 2, parameter: float = 0.4, events_num_crit: int = 10, need_events_num: bool = True, split_thre: float = 0.2):
    place_field_all = []
    smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
    maze_type = trace['maze_type']
    for k in tqdm(range(trace['n_neuron'])):
        a_field = get_all_fields(
                trace=trace, 
                n=k, 
                thre_type=thre_type, 
                parameter=parameter, 
                events_num_crit=events_num_crit, 
                need_events_num=need_events_num,
                split_thre=split_thre
        )
        place_field_all.append(a_field)
    print("    Place field has been generated successfully.")
    return place_field_all

def get_field_number(trace: dict, key: str = 'place_field_all'):
    num = np.zeros(trace['n_neuron'], dtype=np.int64)
    for i in range(trace['n_neuron']):
        num[i] = len(trace[key][i].keys())
    return num