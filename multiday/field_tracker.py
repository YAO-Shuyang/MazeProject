import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mylib.maze_utils3 import GetDMatrices

# Build a tree structure for Field objects
class FieldNode(object):
    def __init__(
        self,
        level: int,
        center: int,
        value: np.ndarray,
        rate: float
    ) -> None:
        """__init__: Initialize FieldNode

        Parameters
        ----------
        level : int
            The level of the node
        center : int
            The center of the place field
        value : np.ndarray
            The field area of the place field
        rate : float
            The center rate of the place field
        """
        self._level = level
        self._center = center
        self._value = value
        self._rate = rate
        self._prev = []
        self._next = []
        self._prev_num = 0
        self._next_num = 0
        self.interval = np.nan
        self.is_checked = False
    
    @property
    def level(self) -> int:
        return self._level
    
    @property
    def center(self) -> int:
        return self._center
    
    @property
    def value(self) -> np.ndarray:
        return self._value
    
    @property
    def rate(self) -> float:
        return self._rate
    
    @property
    def size(self):
        return len(self._value)

    @property
    def get_next(self):
        return self._next
    
    @property
    def get_prev(self):
        return self._prev
    
    def add_next(self, node) -> list:
        self._next.append(node)
        self._next_num += 1
        return self._next
    
    def add_prev(self, node) -> list:
        self._prev.append(node)
        self._prev_num += 1
        return self._prev
        
    def pop_next(self):
        self._next_num -= 1
        self._next.pop()
        return self._next
    
    def pop_prev(self):
        self._prev_num -= 1
        return self._prev.pop()
    

def is_overlap(node1: FieldNode, node2: FieldNode, thre: float = 0.6) -> bool:
    overlap = np.intersect1d(node1.value, node2.value)
    return len(overlap)/node1.size >= thre or len(overlap)/node2.size >= thre

import copy as cp
def BuildTree(place_field_all: list[list[dict]], i: int, smooth_map_all: np.ndarray):
    fields = [place_field_all[k][i] for k in range(len(place_field_all))]
    
    field_queue = []
    all_fields = []
    for k in fields[0].keys():
        node = FieldNode(0, k, fields[0][k], smooth_map_all[0, i, k-1])
        field_queue.append(node)
        all_fields.append(node)
    
    # Initialize the root
    for d in range(1, len(fields)):
        if fields[d] is not None:
            for k in fields[d].keys():
                node = FieldNode(d, k, fields[d][k], smooth_map_all[d, i, k-1])

                is_overlaped = False
                for f in range(len(field_queue)):
                    if is_overlap(field_queue[f], node):
                        field_queue[f].add_next(node)
                        node.add_prev(field_queue[f])
                        
                        field_queue[f].interval = d-field_queue[f].level
                        field_queue[f] = node
                        is_overlaped = True
                        break
                    
                if not is_overlaped:
                    field_queue.append(node)
                
                all_fields.append(node)
    
    
    # Find Gap
    freq = np.zeros(len(fields))
    prior_prob = np.zeros((len(fields), 2))
    n_field = 0
    for n, f in enumerate(all_fields):
        if f.level >= len(fields)-1:
            break
        
        if np.isnan(f.interval) == False:
            freq[int(f.interval)-1] += 1
        
        n_field += 1

        if f.is_checked:
            continue
        
        num = 0
        sonf = [(f, 0)]
        while len(sonf) > 0:
            if np.isnan(sonf[0][0].interval):
                if sonf[0][0].level < len(fields)-1:
                    prior_prob[sonf[0][1], 0] += 1

            else:
                prior_prob[sonf[0][1], 1] += 1
                num += 1
                for k in sonf[0][0].get_next:
                    sonf.append((k, num))
            
            sonf[0][0].is_checked = True
            sonf = sonf[1:]
            
    return freq, n_field, prior_prob

def init_field_reg(curr_level: int, tot_level: int):
    res = np.zeros((tot_level, 1))
    res[curr_level] = 1
    return res

def init_field_info(curr_level: int, tot_level: int, info: np.ndarray):
    res = np.zeros((tot_level, 1, 4)) * np.nan
    res[curr_level, 0, :] = info
    return res
    
def field_register_singlecell(place_field_all: list[dict], smooth_map_all: np.ndarray, cell_id: int):
    roots = []
    root_level = 0
    
    # Initialize roots
    for i in range(len(place_field_all)):
        if place_field_all[i] is not None:
            if len(place_field_all[i].keys()) != 0:
                root_level = i
                for j, k in enumerate(place_field_all[i].keys()):
                    roots.append(FieldNode(i, k, place_field_all[i][k], smooth_map_all[i, k-1]))
                break
    
    field_reg = np.zeros((len(place_field_all), len(roots)), dtype=np.float64)
    field_info = np.zeros((len(place_field_all), len(roots), 4)) * np.nan
    for i, rnodes in enumerate(roots):
        field_info[root_level, i, :] = np.array([cell_id, rnodes.center, len(rnodes.value), rnodes.rate])
        
    field_reg[root_level, :] = 1
    for i in range(root_level+1, len(place_field_all)):
        if place_field_all[i] is not None:
            for k in place_field_all[i].keys():
                field = FieldNode(i, k, place_field_all[i][k], smooth_map_all[i, k-1])
                
                is_preexisted = False
                for j, rnodes in enumerate(roots):
                    if is_overlap(rnodes, field):
                        rnodes.add_next(field)
                        field.add_prev(rnodes)
                        roots[j] = field
                        is_preexisted = True
                        field_reg[i, j] = 1
                        field_info[i, j, :] = np.array([cell_id, field.center, field.size, field.rate])
                        break
                
                if is_preexisted == False:
                    roots.append(field)
                    reg_col = init_field_reg(i, len(place_field_all))
                    info_col = init_field_info(i, len(place_field_all), np.array([cell_id, field.center, field.size, field.rate]))
                    field_reg = np.concatenate([field_reg, reg_col], axis = 1)
                    field_info = np.concatenate([field_info, info_col], axis=1)
                    
    for i in range(len(place_field_all)):
        if place_field_all[i] is None:
            field_reg[i, :] = np.nan
                    
    return field_reg, field_info

def field_register(trace: dict):
    
    index_map = trace['index_map']
    place_field_all = trace['place_field_all']
    smooth_map_all = trace['smooth_map_all']
    
    field_info = np.zeros((index_map.shape[0], 1, 4))
    field_reg = np.zeros((index_map.shape[0], 1))
    
    for i in tqdm(range(index_map.shape[1])):
        reg, info = field_register_singlecell(
            place_field_all=[place_field_all[d][i] for d in range(index_map.shape[0])],
            smooth_map_all=smooth_map_all[:, i, :],
            cell_id=i
        )
        field_reg = np.concatenate([field_reg, reg], axis=1)
        field_info = np.concatenate([field_info, info], axis=1)
        
    trace['field_reg'] = field_reg[:, 1:]
    trace['field_info'] = field_info[:, 1:, :]
    return trace


def conditional_prob(trace: dict):
    """
    if trace['Stage'] in ['Stage 1', 'Stage 1+2'] or trace['maze_type'] == 2:
        field_reg = cp.deepcopy(trace['field_reg'])[2:, :]
    else:
    """
    field_reg = cp.deepcopy(trace['field_reg'])

    prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64)
    recover_prob = np.zeros(field_reg.shape[0], dtype=np.int64)
    disapear_num = 0
    retain_duration = np.zeros(field_reg.shape[0], dtype=np.int64)
    no_detect_duration = np.zeros((field_reg.shape[0], 2), dtype=np.int64)
    
    for j in tqdm(range(field_reg.shape[1])):
        count_one, count_nil = 0, 0 # turn on and turn off
        count_nod = 0 # no detect
        is_disappear = False
        for i in range(field_reg.shape[0]):
            if np.isnan(field_reg[i, j]):
                if count_one >= 1:
                    retain_duration[count_one-1] += 1
                    count_nod += 1
                elif count_one == 0 and count_nod > 1:
                    count_nod += 1
                    
                count_one = 0
                count_nil = 0
                is_disappear = False
            elif field_reg[i, j] == 1:
                if count_nod > 1:
                    
                if count_one >= 1:
                    prob[count_one, 1] += 1
                elif count_one == 0:
                    if count_nil >= 1 and is_disappear:
                        recover_prob[count_nil-1] += 1
                        count_nil = 0
                
                is_disappear = True
                count_one += 1
            elif field_reg[i, j] == 0:
                if count_nil == 0:
                    if count_one >= 1:
                        prob[count_one, 0] += 1
                        count_one = 0
                        disapear_num += 1
                    
                count_nil += 1
            else:
                raise ValueError(f"field reg contains invalid value {field_reg[i, j]} at row {i} column {j}. ")
            
        if count_one >= 1:
            retain_duration[count_one-1] += 1
    
    print(np.sum(prob, axis=1))
    idx = np.where(prob[:, 1] < 4)[0]
    res = prob[:, 1] / np.sum(prob, axis = 1)
    res[idx] = np.nan
    return res, recover_prob / disapear_num, retain_duration


def field_overlapping(trace):
    field_reg = trace['field_reg']
    
    maps = np.where(np.isnan(field_reg), 0, 1)
    idx = np.where(np.sum(maps, axis=0) == field_reg.shape[0])[0]
    field_reg = field_reg[:, idx]
    
    overlapping = np.ones((field_reg.shape[0], field_reg.shape[0]), np.float64)
    for i in range(field_reg.shape[0]-1):
        for j in range(i+1, field_reg.shape[0]):
            
            idx = np.where((np.isnan(field_reg[i, :]) == False) & (np.isnan(field_reg[j, :]) == False))[0]
            if len(idx) == 0:
                overlapping[i, j] = np.nan
                overlapping[j, i] = np.nan
                continue

            overlapping[i, j] = len(np.where((field_reg[i, idx] == 1) & (field_reg[j, idx] == 1))[0]) / len(np.where(field_reg[i, idx] == 1)[0])
            overlapping[j, i] = overlapping[i, j]
    return overlapping

if __name__ == '__main__':
    
    import pickle
    with open(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)    
    
    field_overlapping(trace)
    """
    x = np.arange(0, 26)
    y = np.zeros((26, 2))
    for i in range(26):
        y[i, 1] = X2(i)
        y[i, 0] = X(i)
    plt.plot(x, y)
    plt.show()

    with open(r"E:\Data\Cross_maze\10224\Maze1-2-footprint\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        
    is_placecell = trace['is_placecell']
    freqency = np.zeros(13)
    prior = np.zeros((13, 2))
    n_field = 0
    
    for i in np.where(np.nansum(is_placecell, axis=0) == is_placecell.shape[0])[0]:
        freq, num, pr = BuildTree(trace['place_field_all'], i, trace['smooth_map_all'])
        freqency += freq
        n_field += num
        prior += pr
    
    for i in np.where((np.nansum(is_placecell[1:, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[0, :] == 0))[0]:
        freq, num, pr = BuildTree([trace['place_field_all'][k] for k in range(1, len(trace['place_field_all']))] , i, trace['smooth_map_all'][1:, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
        
    for i in np.where((np.nansum(is_placecell[:-1, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[-1, :] == 0))[0]:
        freq, num = BuildTree([trace['place_field_all'][k] for k in range(len(trace['place_field_all'])-1)] , i, trace['smooth_map_all'][:-1, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
    
    with open(r"E:\Data\Cross_maze\10227\Maze1-2-footprint\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    is_placecell = trace['is_placecell']
    for i in np.where(np.nansum(is_placecell, axis=0) == is_placecell.shape[0])[0]:
        freq, num, pr = BuildTree(trace['place_field_all'], i, trace['smooth_map_all'])
        freqency += freq
        n_field += num
        prior += pr
    
    for i in np.where((np.nansum(is_placecell[1:, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[0, :] == 0))[0]:
        freq, num, pr = BuildTree([trace['place_field_all'][k] for k in range(1, len(trace['place_field_all']))] , i, trace['smooth_map_all'][1:, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
        
    for i in np.where((np.nansum(is_placecell[:-1, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[-1, :] == 0))[0]:
        freq, num = BuildTree([trace['place_field_all'][k] for k in range(len(trace['place_field_all'])-1)] , i, trace['smooth_map_all'][:-1, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
    

    with open(r"E:\Data\Cross_maze\10212\Maze1-2-footprint\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        
    is_placecell = trace['is_placecell']
    """
    """
    for i in np.where(np.nansum(is_placecell, axis=0) == is_placecell.shape[0])[0]:
        freq, num, pr = BuildTree(trace['place_field_all'], i, trace['smooth_map_all'])
        freqency += freq
        n_field += num
        prior += pr
    
    for i in np.where((np.nansum(is_placecell[1:, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[0, :] == 0))[0]:
        freq, num, pr = BuildTree([trace['place_field_all'][k] for k in range(1, len(trace['place_field_all']))] , i, trace['smooth_map_all'][1:, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
        
    for i in np.where((np.nansum(is_placecell[:-1, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[-1, :] == 0))[0]:
        freq, num = BuildTree([trace['place_field_all'][k] for k in range(len(trace['place_field_all'])-1)] , i, trace['smooth_map_all'][:-1, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr

    with open(r"E:\Data\Cross_maze\10209\Maze1-2-footprint\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    is_placecell = trace['is_placecell']
    for i in np.where(np.nansum(is_placecell, axis=0) == is_placecell.shape[0])[0]:
        freq, num, pr = BuildTree(trace['place_field_all'], i, trace['smooth_map_all'])
        freqency += freq
        n_field += num
        prior += pr
    
    for i in np.where((np.nansum(is_placecell[1:, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[0, :] == 0))[0]:
        freq, num, pr = BuildTree([trace['place_field_all'][k] for k in range(1, len(trace['place_field_all']))] , i, trace['smooth_map_all'][1:, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
        
    for i in np.where((np.nansum(is_placecell[:-1, :], axis=0) == is_placecell.shape[0]-1)&(is_placecell[-1, :] == 0))[0]:
        freq, num = BuildTree([trace['place_field_all'][k] for k in range(len(trace['place_field_all'])-1)] , i, trace['smooth_map_all'][:-1, :, :])
        freqency[:-1] += freq
        n_field += num
        prior[:-1, :] += pr
    
    prior = (prior.T / np.nansum(prior, axis=1)).T
    print(prior)
    
    plt.plot(np.arange(1, 14), prior[:, 1])
    plt.show()
    
    from mylib.maze_utils3 import ExponentialFit, Exponential
    
    r, p = ExponentialFit(np.arange(1, 14), freqency/n_field, 1, 1)
    print(r, p)
    y = Exponential(np.arange(1, 14), r, p)
    plt.plot(np.arange(1, 14), freqency/n_field)
    plt.plot(np.arange(1, 14), y)
    plt.show()
    print(freqency/n_field)
    """


        
            
