import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mylib.maze_utils3 import GetDMatrices
from mylib.stats.indeptest import indept_field_evolution_chi2, indept_field_evolution_mutual_info

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


def conditional_prob(trace: dict = None, field_reg: np.ndarray = None, thre: int = 5):
    """
    if trace['Stage'] in ['Stage 1', 'Stage 1+2'] or trace['maze_type'] == 2:
        field_reg = cp.deepcopy(trace['field_reg'])[2:, :]
    else:
    """
    if field_reg is None:
        field_reg = cp.deepcopy(trace['field_reg'])

    duration = np.arange(field_reg.shape[0])  # Retained duration, silent duration, not detected duration
    on_next_prob = np.zeros((field_reg.shape[0], 4), dtype=np.int64) # State ON with duration t -> State ON/OFF/NOT DETECTED on the next sessions
    off_next_prob = np.zeros(field_reg.shape[0], dtype=np.int64) # State OFF on Session t -> State ON DETECTED on Session t+1
    nod_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State NOT DETECTED on Session t -> State ON DETECTED on Session t+1
    silent_num = 0
    no_detect_num = 0
    
    for j in tqdm(range(field_reg.shape[1])):
        count_one, count_nil = 0, 0 # turn on and turn off
        count_nod = 0 # no detect
        is_disappear = False
        for i in range(field_reg.shape[0]):
            if np.isnan(field_reg[i, j]):
                if count_one >= 1:
                    on_next_prob[count_one, 2] += 1
                    count_nod += 1
                    no_detect_num += 1
                elif count_one == 0 and count_nil == 0 and count_nod >= 1:
                    count_nod += 1
                elif count_nil >= 1:
                    no_detect_num += 1
                
                count_one = 0
                count_nil = 0
                is_disappear = False
            elif field_reg[i, j] == 1:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 1] += 1
                    count_nod = 0
                if count_one >= 1:
                    on_next_prob[count_one, 1] += 1
                elif count_one == 0:
                    if count_nil >= 1 and is_disappear:
                        off_next_prob[count_nil] += 1
                        count_nil = 0
                
                is_disappear = True
                count_one += 1
            elif field_reg[i, j] == 0:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 0] += 1
                    count_nod = 0
                if count_nil == 0:
                    if count_one >= 1:
                        on_next_prob[count_one, 0] += 1
                        count_one = 0
                        silent_num += 1

                count_nil += 1
            else:
                raise ValueError(f"field reg contains invalid value {field_reg[i, j]} at row {i} column {j}. ")

        if count_one >= 1:
            on_next_prob[count_one-1, 3] += 1
        
    retained_prob = on_next_prob[:, 1] / np.sum(on_next_prob[:, :2], axis=1)
    nodetect_prob = on_next_prob[:, 2] / (np.sum(on_next_prob[:, :3], axis=1) + np.concatenate([[0], on_next_prob[:-1, 3]]))

    recover_prob = off_next_prob / silent_num
    redetect_prob = nod_next_prob[:, 1] / (nod_next_prob[:, 0] + nod_next_prob[:, 1])
    redetect_frac = np.sum(nod_next_prob, axis=1) / no_detect_num
    
    retained_prob[np.where(on_next_prob[:, 1] <= thre)[0]] = np.nan
    nodetect_prob[np.where(on_next_prob[:, 2] <= thre)[0]] = np.nan
    recover_prob[np.where(off_next_prob <= thre)[0]] = np.nan
    redetect_prob[np.where(nod_next_prob[:, 1] <= thre)[0]] = np.nan

    return duration, retained_prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob


def conditional_prob_jumpnan(trace: dict = None, field_reg: np.ndarray = None, thre: int = 5):
    """
    if trace['Stage'] in ['Stage 1', 'Stage 1+2'] or trace['maze_type'] == 2:
        field_reg = cp.deepcopy(trace['field_reg'])[2:, :]
    else:
    """
    if field_reg is None:
        field_reg = cp.deepcopy(trace['field_reg'])

    duration = np.arange(field_reg.shape[0])  # Retained duration, silent duration, not detected duration
    on_next_prob = np.zeros((field_reg.shape[0], 4), dtype=np.int64) # State ON with duration t -> State ON/OFF/NOT DETECTED on the next sessions
    off_next_prob = np.zeros(field_reg.shape[0], dtype=np.int64) # State OFF on Session t -> State ON DETECTED on Session t+1
    nod_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State NOT DETECTED on Session t -> State ON DETECTED on Session t+1
    silent_num = 0
    no_detect_num = 0
    
    for j in tqdm(range(field_reg.shape[1])):
        count_one, count_nil = 0, 0 # turn on and turn off
        count_nod = 0 # no detect
        is_disappear = False
        for i in range(field_reg.shape[0]):
            if np.isnan(field_reg[i, j]):
                if i < field_reg.shape[0]-1 and field_reg[i-1, j] == 1:
                    if field_reg[i+1, j] == 1:
                        on_next_prob[count_one, 1] += 1
                        count_one += 1
                        continue
                
                if count_one >= 1:
                    on_next_prob[count_one, 2] += 1
                    count_nod += 1
                    no_detect_num += 1
                elif count_one == 0 and count_nil == 0 and count_nod >= 1:
                    count_nod += 1
                elif count_nil >= 1:
                    no_detect_num += 1
                
                count_one = 0
                count_nil = 0
                is_disappear = False
            elif field_reg[i, j] == 1:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 1] += 1
                    count_nod = 0
                if count_one >= 1:
                    on_next_prob[count_one, 1] += 1
                elif count_one == 0:
                    if count_nil >= 1 and is_disappear:
                        off_next_prob[count_nil] += 1
                        count_nil = 0
                
                is_disappear = True
                count_one += 1
            elif field_reg[i, j] == 0:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 0] += 1
                    count_nod = 0
                if count_nil == 0:
                    if count_one >= 1:
                        on_next_prob[count_one, 0] += 1
                        count_one = 0
                        silent_num += 1

                count_nil += 1
            else:
                raise ValueError(f"field reg contains invalid value {field_reg[i, j]} at row {i} column {j}. ")

        if count_one >= 1:
            on_next_prob[count_one-1, 3] += 1
        
    retained_prob = on_next_prob[:, 1] / np.sum(on_next_prob[:, :2], axis=1)
    nodetect_prob = on_next_prob[:, 2] / (np.sum(on_next_prob[:, :3], axis=1) + np.concatenate([[0], on_next_prob[:-1, 3]]))

    recover_prob = off_next_prob / silent_num
    redetect_prob = nod_next_prob[:, 1] / (nod_next_prob[:, 0] + nod_next_prob[:, 1])
    redetect_frac = np.sum(nod_next_prob, axis=1) / no_detect_num
    
    retained_prob[np.where(on_next_prob[:, 1] <= thre)[0]] = np.nan
    nodetect_prob[np.where(on_next_prob[:, 2] <= thre)[0]] = np.nan
    recover_prob[np.where(off_next_prob <= thre)[0]] = np.nan
    redetect_prob[np.where(nod_next_prob[:, 1] <= thre)[0]] = np.nan

    return duration, retained_prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob

def field_overlapping(trace):
    field_reg = trace['field_reg']
    
    maps = np.where(np.isnan(field_reg), 0, 1)
    
    overlapping = np.ones((field_reg.shape[0], field_reg.shape[0]), np.float64)
    for i in range(field_reg.shape[0]-1):
        for j in range(i+1, field_reg.shape[0]):
            
            idx = np.where(np.sum(maps[i:j+1], axis=0) == j-i+1)[0]
            sub_reg = field_reg[:, idx]
            if len(np.where(sub_reg[i, :] == 1)[0]) < 10:
                overlapping[i, j] = np.nan
                overlapping[j, i] = np.nan
                continue
            
            overlapping[i, j] = len(np.where((sub_reg[i, :] == 1) & (sub_reg[j, :] == 1))[0]) / len(np.where(sub_reg[i, :] == 1)[0])
            overlapping[j, i] = overlapping[i, j]
    
    start_session = np.zeros_like(overlapping)
    for i in range(overlapping.shape[0]):
        start_session[i, :] = i+1
    
    intervals = np.zeros_like(overlapping)
    for i in range(overlapping.shape[0]):
        for j in range(overlapping.shape[0]):
            intervals[i, j] = j-i
        
    return overlapping[np.where(np.triu(overlapping, k=1) != 0)], start_session[np.where(np.triu(start_session, k=1) != 0)], intervals[np.where(np.triu(intervals, k=1) != 0)]

def get_evolve_event_label(evolve_event: np.ndarray):
    n = len(evolve_event)
    # Create an array of powers of 2, reversed, to match the binary representation
    powers_of_two = 2 ** np.arange(n)[::-1]
    # Perform dot product
    return np.dot(evolve_event, powers_of_two)

def indept_test_for_evolution_events(field_reg: np.ndarray, field_info: np.ndarray):
    sessions, chi2_stat, MI, pair_type, pair_num = np.arange(field_reg.shape[0]-1), [], [],[], []
    
    for i in range(2, field_reg.shape[0]-5):
        print(f"    Session {i+1} -> {i+2}")
        idx = np.where(np.isnan(np.sum(field_reg[i:i+5, :], axis=0)) == False)[0]
        real_distribution = np.zeros(idx.shape[0])
        for j in range(real_distribution.shape[0]):
            real_distribution[j] = get_evolve_event_label(field_reg[i:i+5, idx[j]])
        
        evol_event_sib, evol_event_non = [], []
        for j in tqdm(range(len(idx))):
            for k in range(len(idx)):
                if j == k:
                    continue
                
                if np.sum(field_reg[i:i+5, idx[j]]) == 0 or np.sum(field_reg[i:i+5, idx[k]]) == 0:
                    continue
                
                if int(np.nanmax(field_info[i:i+5, idx[j], 0])) == int(np.nanmax(field_info[i:i+5, idx[k], 0])):
                    # if np.nansum(field_reg[2:, idx[j]]) >= 4 and np.nansum(field_reg[2:, idx[k]]) >= 4:
                    evol_event_sib.append([get_evolve_event_label(field_reg[i:i+5, idx[j]]), 
                                               get_evolve_event_label(field_reg[i:i+5, idx[k]])])
                else:
                    evol_event_non.append([get_evolve_event_label(field_reg[i:i+5, idx[j]]), 
                                           get_evolve_event_label(field_reg[i:i+5, idx[k]])])
                    
        evol_event_sib = np.array(evol_event_sib, np.int64)
        evol_event_non = np.array(evol_event_non, np.int64)[np.random.randint(0, len(evol_event_non), size=evol_event_sib.shape[0]), :]
        print(evol_event_sib.shape, evol_event_non.shape)
        chi_stat_sib, chi_stat_non, n_pair_stat, _ = indept_field_evolution_chi2(
            real_distribution=real_distribution,
            X_pairs=evol_event_sib,
            Y_pairs=evol_event_non
        )
        mi_sib, mi_non, _, _ = indept_field_evolution_mutual_info(
            X_pairs=evol_event_sib,
            Y_pairs=evol_event_non
        )
        
        chi2_stat = chi2_stat + [chi_stat_sib, chi_stat_non]
        MI = MI + [mi_sib, mi_non]
        pair_type = pair_type + ['Sibling', 'Non-sibling']
        pair_num = pair_num + [evol_event_sib.shape[0], evol_event_non.shape[0]]
        print("      Chi2 test: ", chi_stat_sib, chi_stat_non)
        print("      Mutual info: ", mi_sib, mi_non)
        
    return sessions, np.array(chi2_stat), np.array(MI), np.array(pair_type), np.ndarray(pair_num)

        


if __name__ == '__main__':
    
    import pickle
    import pandas as pd
    with open(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)    
    
    indept_test_for_evolution_events(trace['field_reg'], trace['field_info'])
    """
    REG = {}
    INFO = {}
    for i, day in enumerate([ 20230806, 20230808, 20230810, 20230812, 
          20230814, 20230816, 20230818, 20230820, 
          20230822, 20230824, 20230827, 20230829,
          20230901,
          20230906, 20230908, 20230910, 20230912, 
          20230914, 20230916, 20230918, 20230920, 
          20230922, 20230924, 20230926, 20230928, 
          20230930]):
        REG[day] = trace['field_reg'][i, :]
        INFO[day] = trace['field_info'][i, :, 0]
    
    REG = pd.DataFrame(REG)
    INFO = pd.DataFrame(INFO)
    
    REG.to_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227superlong-field_reg_output.xlsx", index=False)
    INFO.to_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227superlong-field_info_output.xlsx", index=False)
    """
        
            
