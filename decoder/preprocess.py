import numpy as np
import copy as cp
from mylib.divide_laps.lap_split import LapSplit

def preprocess_signal(
    trace: dict,
    signal_type: str = "RawTraces",
) -> tuple[np.ndarray, np.ndarray]:
    if signal_type not in ['RawTraces', 'RawTracesGenerateSpikes', 'Spikes', 'DeconvSignal']:
        raise ValueError("signal_type must be RawTraces, Spikes, RawTracesGenerateSpikes or DeconvSignal, not {}".format(signal_type))
    
    T = trace['ms_time_behav'].shape[0]
        
    if signal_type in ['RawTraces', 'DeconvSignal']:
        cal_signal = cp.deepcopy(trace[signal_type])
        cal_spikes_processed = np.zeros_like(trace['Spikes'], dtype=np.float64)
        for i in range(T):
            t = np.where(trace['ms_time'] == trace['ms_time_behav'][i])[0][0]
            cal_spikes_processed[:, i] = cal_signal[:, t]
            
    elif signal_type == 'RawTracesGenerateSpikes':
        cal_signal = cp.deepcopy(trace['RawTraces'])
        sig_std = np.std(cal_signal, axis=1)
        cal_spikes = np.zeros_like(cal_signal, dtype=np.float64)
        
        for i in range(cal_spikes.shape[0]):
            cal_spikes[i, :] = np.where(cal_signal[i, :] >= 3*sig_std[i], 1, 0)
            
        cal_spikes_processed = np.zeros_like(trace['Spikes'], dtype=np.float64)
        for i in range(T):
            t = np.where(trace['ms_time'] == trace['ms_time_behav'][i])[0][0]            
            cal_spikes_processed[:, i] = cal_spikes[:, t]
    else:
        cal_spikes_processed = cp.deepcopy(trace[signal_type])
    
    assert cal_spikes_processed.shape[1] == T and cp.deepcopy(trace['spike_nodes']).shape[0] == T
    print(cal_spikes_processed.shape, cp.deepcopy(trace['spike_nodes']).shape)
    return cal_spikes_processed, cp.deepcopy(trace['spike_nodes'])

def shuffle_data(
    Spikes: np.ndarray
):
    idx = np.arange(Spikes.shape[1])
    shifts = np.random.choice(idx, Spikes.shape[0])
    shuffle_Spikes = cp.deepcopy(Spikes)
    
    for i in range(Spikes.shape[0]):
        shuffle_Spikes[i, :] = np.roll(Spikes[i, :], shifts[i])
        
    return shuffle_Spikes
    
 
def split_train_test_sets(
    trace,
    Spikes: np.ndarray,
    spike_nodes: np.ndarray,
    decode_lap: int,
    maximum_train_percentage: float = 0.6,
):
    T = Spikes.shape[1]
    
    if trace['maze_type'] == 0:
        TBehav = trace['correct_time'].shape[0]
        beg, end = np.array([int(TBehav*i) for i in np.linspace(0, 0.8, 5)], np.int64), np.array([int(TBehav*i)-1 for i in np.linspace(0.2, 1, 5)])
    else:
        beg, end = LapSplit(trace, trace['paradigm'])
        
    lap_num = beg.shape[0]
    
    assert decode_lap < lap_num
    
    train_lap_num = int(lap_num*maximum_train_percentage)
    
    if train_lap_num == 0:
        return Spikes[:, np.arange(int(maximum_train_percentage*T))], spike_nodes[np.arange(int(maximum_train_percentage*T))], Spikes[:, np.arange(int(maximum_train_percentage*T), T)], spike_nodes[np.arange(int(maximum_train_percentage*T), T)]
            
    if decode_lap in np.arange(lap_num-train_lap_num, lap_num):
        train_laps = [i for i in range(lap_num-train_lap_num-1, decode_lap)] + [i for i in range(decode_lap+1, lap_num)]
    else:
        train_laps = [i for i in range(lap_num-train_lap_num, lap_num)]
    
    if len(train_laps) == 1:
        train_idx = np.where((trace['ms_time_behav'] >= trace['correct_time'][beg[train_laps[0]]])&(trace['ms_time_behav'] <= trace['correct_time'][end[train_laps[0]]]))[0]
    else:
        train_idx = np.concatenate([np.where((trace['ms_time_behav'] >= trace['correct_time'][beg[i]])&(trace['ms_time_behav'] <= trace['correct_time'][end[i]]))[0] for i in train_laps])
    
    test_idx = np.where((trace['ms_time_behav'] >= trace['correct_time'][beg[decode_lap]])&(trace['ms_time_behav'] <= trace['correct_time'][end[decode_lap]]))[0]
    
    return Spikes[:, train_idx], spike_nodes[train_idx], Spikes[:, test_idx], spike_nodes[test_idx]
