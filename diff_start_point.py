import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import copy as cp
from mylib.maze_utils3 import spike_nodes_transform
from mylib.maze_graph import DSPLapSplitCheckPoint1_1, DSPLapSplitCheckPoint2_1, DSPLapSplitCheckPoint1_2, DSPLapSplitCheckPoint2_2

def DSPMazeLapSplit(trace, check_length = 5):
    raise ValueError("We strongly recommend you to use manually labeled data to get lap information.")

    if trace['maze_type'] in [1,2]:
        beg_idx = []
        end_idx = []
        if len(np.where(np.isnan(trace['correct_nodes']))[0]) != 0:
            print('Error! correct_nodes contains NAN value!')
            return [], []

        #check area 1, check area 2
        if trace['maze_type'] == 1:
            check1 = DSPLapSplitCheckPoint1_1
            check2 = DSPLapSplitCheckPoint1_2
        elif trace['maze_type'] == 2:
            check1 = DSPLapSplitCheckPoint2_1
            check2 = DSPLapSplitCheckPoint2_2
        else:
            assert False

        behav_nodes = cp.deepcopy(trace['correct_nodes'])
        behav_nodes = spike_nodes_transform(spike_nodes = behav_nodes, nx = 12)
        pre_state = 2

        # Check if lap-start or lap-end point frame by frame
        for k in range(behav_nodes.shape[0]):
            # start area --(state = 0)--> check area 1 --(state = 1)--> check area 2 --(state = 2)--> end area
            n = behav_nodes[k]
            
            if check1[n-1] == -1 and check2[n-1] == -1: # state = 0
                if pre_state == 1:
                    pre_state = 0
                elif pre_state == 2:
                    beg_idx.append(k)
                    if k != 0:
                        end_idx.append(k-1)
                    pre_state = 0
            
            elif check1[n-1] >= 0 and check2[n-1] == -1:  # state = 1
                pre_state == 1
            
            elif check1[n-1] >= 0 and check2[n-1] >= 0: #state = 2
                pre_state = 2

        if pre_state == 2:
            end_idx.append(behav_nodes.shape[0]-1)
            
        if len(beg_idx) != len(end_idx):
            # Abort when swith = 1
            beg_idx.pop()
        return np.array(beg_idx, dtype = np.int64), np.array(end_idx, dtype = np.int64)

    elif trace['maze_type'] == 0:
        behav_nodes = trace['correct_nodes']
        unit = int(behav_nodes.shape[0] / 2)
        beg_idx = [0, unit]
        end_idx = [unit-1, behav_nodes.shape[0]-1]
        return np.array(beg_idx, dtype = np.int64), np.array(end_idx, dtype = np.int64)
    else:
        print("    Error in maze_type! Report by mylib.maze_utils3.CrossMazeLapSplit")
        return np.array([], dtype = np.int64), np.array([], dtype = np.int64)

if __name__ == '__main__':
    import pickle

    with open(r"G:\YSY\Dsp_maze\10212\20230602\session 1\trace_behav.pkl", 'rb') as handle:
        trace = pickle.load(handle)
