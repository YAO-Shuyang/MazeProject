import numpy as np
import copy as cp
from tqdm import tqdm
from mylib.maze_utils3 import maze_graphs, spike_nodes_transform, EndPoints, StartPoints, correct_paths

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
def GetPlaceField(trace: dict, n: int, thre_type: int, nx: int = 48, parameter: float = 0.5) -> dict:
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
    smooth_map = trace['smooth_map_all'][n, :]
    maze_type = trace['maze_type']

    # rate_map should be one without NAN value. Use function clear_NAN(rate_map_all) to process first.
    r_max = np.nanmax(smooth_map)
    thre = get_field_thre(peak_rate=r_max, thre_type = thre_type, parameter = parameter)
    all_fields = np.where(smooth_map >= thre)[0]+1
    search_set = np.array([], dtype=np.int64)
    All_field = {}

    while len(np.setdiff1d(all_fields, search_set))!=0:
        diff = np.setdiff1d(all_fields, search_set)
        point = diff[0]
        subfield = _field(all_fields = all_fields, point = point, maze_type = maze_type, nx = nx, thre=thre)
        IS_QUALIFIED_FIELD, retain_fields = field_quality_control(trace=trace, n=n, field_bins=subfield)

        if IS_QUALIFIED_FIELD:
            peak_loc = subfield[0]
            peak = smooth_map[peak_loc-1]
            # find peak idx as keys of place_field_all dict objects.
            for k in subfield:
                if smooth_map[k-1] > peak:
                    peak = smooth_map[k-1]
                    peak_loc = k
        
            All_field[peak_loc] = retain_fields

        search_set = np.concatenate([search_set, subfield])
    
    return All_field
               
def _field(all_fields: list | np.ndarray, point: int, maze_type: int, nx: int = 48, thre: float = 0.5):
    # Identify single field from all fields.
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
            
    MaxStep = 300
    step = 0
    Area = [point]
    StepExpand = {0: [point]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:
                if j in all_fields and j not in Area:
                    StepExpand[step+1].append(j)
                    Area.append(j)
        
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
            
        step += 1
    return np.array(Area, dtype=np.int64)

def field_quality_control(trace: dict, n: int, field_bins: np.ndarray | list) -> bool:
    father_field = spike_nodes_transform(field_bins, nx = 12)
    father_field_uniq = np.unique(father_field)
    spike_nodes = spike_nodes_transform(trace['spike_nodes'], nx=12)

    total_indices = np.concatenate([np.where(spike_nodes == i)[0] for i in father_field_uniq])
    events_num = np.nansum(trace['Spikes'][n, total_indices])

    if events_num <= 10:
        return False, None

    retain_fields = np.array([], dtype=np.int64)
    for pf in father_field_uniq:
        events_indices = np.where(spike_nodes==pf)[0]
        events_num = np.nansum(trace['Spikes'][n, events_indices])
        if events_num != 0:
            retain_fields = np.concatenate([retain_fields, field_bins[np.where(father_field==pf)[0]]])

    if len(retain_fields) == 0:
        return False, None
    else:
        return True, retain_fields

# get all cell's place field
def place_field(trace: dict, thre_type: int, parameter: float = 0.4):
    place_field_all = []
    smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
    maze_type = trace['maze_type']
    for k in tqdm(range(trace['n_neuron'])):
        if k in trace['SilentNeuron']:
            place_field_all.append({})
        else:
            a_field = GetPlaceField(trace=trace, n=k, thre_type=thre_type, parameter=parameter)
            place_field_all.append(a_field)
    print("    Place field has been generated successfully.")
    return place_field_all

def count_field_number(trace: dict) -> dict:
    n_neuron = np.zeross

if __name__ == '__main__':
    import pickle
    
    with open(r"E:\Data\Cross_maze\11095\20220830\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        
    place_field(trace, thre_type=2, parameter=0.4)