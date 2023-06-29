from mylib.maze_graph import ReverseLapSplitCheckPoint
import numpy as np
import copy as cp
from mylib.maze_utils3 import spike_nodes_transform, plot_trajactory

# This function is specifically written for reverse maze paradigm.
def ReverseMazeLapSplit(trace, check_length = 2):
    if trace['maze_type'] in [1,2,3]:
        beg_idx = []
        end_idx = []
        if len(np.where(np.isnan(trace['correct_nodes']))[0]) != 0:
            print('Error! correct_nodes contains NAN value!')
            return [], []

        #check area 1, check area 2
        check = ReverseLapSplitCheckPoint[trace['maze_type']]

        behav_nodes = cp.deepcopy(trace['correct_nodes'])
        behav_nodes = spike_nodes_transform(spike_nodes = behav_nodes, nx = 12)
        
        # initiate value for start state
        if check[0] == 1:
            start_state = 1
        elif check[0] == -1:
            start_state = 0
        else:
            raise ValueError(f"start_state has invalid value {start_state}")

        beg_idx.append(0)

        for k in range(behav_nodes.shape[0]):
            # start (state = 0) --(check area)--> end (state = 1)
            n = behav_nodes[k]

            if n not in [1, 144]:
                continue

            if n == 1 and start_state == 1:
                end_idx.append(k)
                if k != behav_nodes.shape[0]-1:
                    beg_idx.append(k+1)
                start_state = 0
            elif n == 144 and start_state == 0:
                end_idx.append(k)
                if k != behav_nodes.shape[0]-1:
                    beg_idx.append(k+1)
                start_state = 1
        
        if len(beg_idx) != len(end_idx):
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
    print()