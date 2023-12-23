import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from mylib.divide_laps.lap_split import LapSplit
from tqdm import tqdm
    
def is_active(
    field: np.ndarray | list,
    signal: np.ndarray,
    spike_nodes: np.ndarray
) -> bool:
    idx = np.concatenate([np.where(spike_nodes == i)[0] for i in field])
    if np.nansum(signal[idx]) > 0:
        return 1
    else:
        return 0

def active_field_prop(
    trace: dict
):
    signals = cp.deepcopy(trace['DeconvSignal'])
    ms_time = cp.deepcopy(trace['ms_time'])
    spike_nodes = cp.deepcopy(trace['spike_nodes_original'])
    is_pc = cp.deepcopy(trace['is_placecell'])
    is_perfect = cp.deepcopy(trace['is_perfect'])
    
    beg, end = LapSplit(trace, trace['paradigm'])
    
    assert is_perfect.shape[0] == beg.shape[0]
    
    # Initial maps
    field_reg = []
    for i in range(is_pc.shape[0]):
        if is_pc[i] == 1:
            for j, k in enumerate(trace['place_field_all'][i].keys()):
                field_reg.append([i, j, k, len(trace['place_field_all'][i][k])])
    
    field_reg = np.array(field_reg, np.int64)
    
    activation_map = np.zeros((is_perfect.shape[0], field_reg.shape[0]))
    
    for i in tqdm(range(is_perfect.shape[0])):
        #print(f"    Analyze Lap {i+1} (is_perfect? {is_perfect[i]}):")
        t1, t2 = np.where(ms_time >= trace['correct_time'][beg[i]])[0][0], np.where(ms_time <= trace['correct_time'][end[i]])[0][-1]
        
        for j in range(field_reg.shape[0]):
            cell, fkey = field_reg[j, 0], field_reg[j, 2]
            if is_active(
                field = trace['place_field_all'][cell][fkey],
                signal= signals[cell, t1:t2],
                spike_nodes=spike_nodes[t1:t2]
            ) == 1:
                activation_map[i, j] = 1
        #print(f"  Activated Fields: {np.sum(activation_map[i, :])}/{field_reg.shape[0]}, proportion: {np.round(np.sum(activation_map[i, :])/field_reg.shape[0]*100,2)}%")

    trace['field_reg'], trace['activation_map'] = field_reg, activation_map
    return trace
    
def field_register(trace: dict):
    is_pc = cp.deepcopy(trace['cis']['is_placecell'])
    
    # Initial maps
    field_reg = []
    for i in range(is_pc.shape[0]):
        if is_pc[i] == 1:
            for j, k in enumerate(trace['cis']['place_field_all'][i].keys()):
                field_reg.append([i, j, k, len(trace['cis']['place_field_all'][i][k])])
    
    field_reg_cis = np.array(field_reg, np.int64)

    is_pc = cp.deepcopy(trace['trs']['is_placecell'])
    
    # Initial maps
    field_reg = []
    for i in range(is_pc.shape[0]):
        if is_pc[i] == 1:
            for j, k in enumerate(trace['trs']['place_field_all'][i].keys()):
                field_reg.append([i, j, k, len(trace['trs']['place_field_all'][i][k])])
    
    field_reg_trs = np.array(field_reg, np.int64)

    trace['cis']['field_reg'], trace['trs']['field_reg'] = field_reg_cis, field_reg_trs
    return trace

if __name__ == '__main__':
    import pickle
    
    from mylib.local_path import f3, f4
        
    for i in range(len(f4)):
        print(i, f4['MiceID'][i], f4['date'][i], ' session '+str(f4['session'][i]))
        with open(f4['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        trace = field_register(trace)
        
        with open(f4['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
        print()