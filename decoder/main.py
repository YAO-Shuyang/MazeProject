from mylib.local_path import f1
from mylib.divide_laps.lap_split import LapSplit
from mylib.decoder.NaiveBayesianDecoder import NaiveBayesDecoder
from mylib.maze_utils3 import DateTime, mkdir

import pandas as pd
import numpy as np
import pickle
import os

from mylib.decoder.preprocess import preprocess_signal, split_train_test_sets, shuffle_data

def main(
    f1: pd.DataFrame,
    i:int,
    res:pd.DataFrame,
    j:int,
    save_loc:str = r"E:\Data\Simulation_pc\cross_maze_decode_res",
    save_sheet: str = r"E:\Data\Simulation_pc\cross_maze_decode_output.xlsx",
    signal_type: str = "RawTracesGenerateSpikes",
    max_percentage: float = 0.6,
    nx: int = 48,
    is_shuffle: bool = False
):
    if f1['include'][i] == 0 or os.path.exists(f1['Trace File'][i]) == False:
        return res, j
    
    if os.path.exists(save_loc) == False:
        mkdir(save_loc)
    
    print(f"{i} {f1['MiceID'][i]} {f1['date'][i]} Session {f1['session'][i]} {f1['Stage'][i]} {f1['training_day'][i]} -----------------------------------------------")
    print("  A. Initial parameters and split training and testing set.")
    with open(f1['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    T = trace['Spikes'].shape[1]
    
    if trace['maze_type'] == 0:
        TBehav = trace['correct_time'].shape[0]
        beg, end = np.array([int(TBehav*i) for i in np.linspace(0, 0.8, 5)], np.int64), np.array([int(TBehav*i)-1 for i in np.linspace(0.2, 1, 5)])
    else:
        beg, end = LapSplit(trace, trace['paradigm'])
        
    Spikes, spike_nodes = preprocess_signal(
        trace,
        signal_type = signal_type,
    )
    
    print(is_shuffle)
    if is_shuffle:
        print("   Shuffle...")
        Spikes = shuffle_data(Spikes)
    
    for l in range(beg.shape[0]):
        if is_shuffle:
            res.loc[j, 'is_shuffle'] = 'Shuffle'
        else:
            res.loc[j, 'is_shuffle'] = 'No'
        print(f"    Lap {l+1}/{beg.shape[0]}")
        res.loc[j, 'Test ID'] = j
        res.loc[j, 'MiceID'] = int(f1['MiceID'][i])
        res.loc[j, 'Date'] = int(f1['date'][i])
        res.loc[j, 'Session'] = int(f1['session'][i])
        res.loc[j, 'Stage'] = f1['Stage'][i]
        res.loc[j, 'Training Day'] = f1['training_day'][i]
        res.loc[j, 'Maze Type'] = int(f1['maze_type'][i])
        res.loc[j, 'Lap ID'] = l+1
        res.loc[j, 'Time Begin'] = DateTime(is_print=True)
        res.loc[j, 'Signal Type'] = signal_type
        
        print("    B. Training and testing set split.")
        Spikes_train, spike_nodes_train, Spikes_test, spike_nodes_test = split_train_test_sets(
            trace,
            Spikes=Spikes,
            spike_nodes=spike_nodes,
            decode_lap=l,
            maximum_train_percentage=max_percentage,
        )
        
        save_dirname = os.path.join(save_loc, f"{i}_{int(f1['MiceID'][i])}_{int(f1['date'][i])}_session{int(f1['session'][i])}_{f1['Stage'][i]}_lap{l+1}_{signal_type}.pkl")
        
        res.loc[j, 'Neuron Number'] = Spikes.shape[0]
        res.loc[j, 'nx'] = nx
        res.loc[j, 'Total Frame Size'] = Spikes.shape[1]
        res.loc[j, 'Fitting Frame Size'] = Spikes_train.shape[1]
        res.loc[j, 'Testing Frame Size'] = Spikes_test.shape[1]
        res.loc[j, 'Testing Size'] = spike_nodes_test.shape[0]/(spike_nodes_test.shape[0] + spike_nodes_train.shape[0])
        res.loc[j, 'Trace File'] = f1['Trace File'][i]
        res.loc[j, 'Results File'] = save_dirname
        
        print("    C. Fit the Naive Bayesian model.")
        Model = NaiveBayesDecoder(
            maze_type=f1['maze_type'][i],
            res=nx
        )
        Model.fit(Spikes_train, spike_nodes_train)
        spike_nodes_pred = Model.predict(Spikes_test, spike_nodes_test)
        MSE, std_abd, RMSE, MAE, std_mae, abd = Model.metrics_mae(spike_nodes_test, spike_nodes_pred)
        abHit, geHit = Model.metrics_accuracy(spike_nodes_test, spike_nodes_pred)
        res.loc[j, 'MSE'] = MSE
        res.loc[j, 'RMSE'] = RMSE
        res.loc[j, 'MAE'] = MAE
        res.loc[j, 'Std. MAE'] = std_mae
        res.loc[j, 'Absolute Accuracy'] = abHit
        res.loc[j, 'General Accuracy'] = geHit
        res.loc[j, 'Decoder Version'] = Model.version
        
        with open(save_dirname, 'wb') as handle:
            pickle.dump({
                    "y_pred": spike_nodes_pred,
                    "y_test": spike_nodes_test
                }, 
                handle
            )
        res.loc[j, 'Time End'] = DateTime(is_print=True)
        print("  Done. Next Lap ↓", end='\n\n')
        j = j+1
        
        res.to_excel(save_sheet, index=False)
    
    return res, j


if __name__ == '__main__':
    res = pd.read_excel(r"E:\Data\Simulation_pc\cross_maze_decode.xlsx", sheet_name='Lap-wise')
    j = 3734
    for i in range(382, len(f1)):
        res, j = main(f1, i, res, j, save_loc = r"E:\Data\Simulation_pc\cross_maze_decode_res")