import numpy as np
import copy as cp
from mylib.maze_utils3 import spike_nodes_transform
from mylib.maze_graph import ReverseLapSplitCheckPoint, StartPoints, EndPoints

# This function is specifically written for reverse maze paradigm.
def ReverseMazeLapSplit(trace):
    if trace['maze_type'] in [1,2,3]:
        if 'lap beg time' in trace.keys() and 'lap end time' in trace.keys():
            beg_time_ori, end_time_ori = trace['lap beg time'], trace['lap end time']
            behav_time = cp.deepcopy(trace['correct_time'])
            beg, end = np.zeros_like(beg_time_ori, dtype=np.int64), np.zeros_like(end_time_ori, dtype=np.int64)
            laps = beg_time_ori.shape[0]

            for i in range(laps):
                beg[i] = np.where(behav_time >= beg_time_ori[i])[0][0]
                end[i] = np.where(behav_time <= end_time_ori[i])[0][-1]
            
            return beg, end
        
        beg_idx = []
        end_idx = []
        if len(np.where(np.isnan(trace['correct_nodes']))[0]) != 0:
            print('Error! correct_nodes contains NAN value!')
            return [], []

        #check area 1, check area 2
        check = ReverseLapSplitCheckPoint[int(trace['maze_type'])]
        beg, end = StartPoints[int(trace['maze_type'])], EndPoints[int(trace['maze_type'])]

        behav_nodes = cp.deepcopy(trace['correct_nodes'])
        behav_nodes = spike_nodes_transform(spike_nodes = behav_nodes, nx = 12)
        
        # initiate value for start state
        if check[behav_nodes[0]-1] == 1:
            start_state = 1
        elif check[behav_nodes[0]-1] == -1:
            start_state = 0
        else:
            raise ValueError(f"start_state has invalid value {start_state}")

        beg_idx.append(0)

        for k in range(behav_nodes.shape[0]):
            # start (state = 0) --(check area)--> end (state = 1)
            n = behav_nodes[k]

            if n not in [beg, end]:
                continue

            if n == beg and start_state == 1:
                end_idx.append(k)
                if k != behav_nodes.shape[0]-1:
                    beg_idx.append(k+1)
                start_state = 0
            elif n == end and start_state == 0:
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