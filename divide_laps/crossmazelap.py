import numpy as np
import copy as cp
from mylib.maze_utils3 import spike_nodes_transform
from mylib.maze_graph import correct_paths, maze_graphs

def get_check_area(maze_type: int, start_point: int, check_length: int = 5):
    if start_point > 144 or start_point < 1:
        raise ValueError("Only values belong to set {1, 2, ..., 144} are valid! But "+f"{start_point} was inputt.")

    area = [start_point]
    graph = maze_graphs[(int(maze_type),12)]
    surr = graph[start_point]
    prev = 1

    StepExpand = {1: [start_point]}
    while prev <= check_length:
        StepExpand[prev+1] = []
        for j in StepExpand[prev]:
            for k in graph[j]:
                if k not in area:
                    area.append(k)
                    StepExpand[prev+1].append(k)
        prev += 1

    return area


# This function has been proved to be suited for all session for cross maze paradigm.
def CrossMazeLapSplit(trace, check_length = 6, mid_length = 5):
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
            return [],[]

        # Define start area, end area and check area with certain sizes that are defaultly set as 5(check_length).
        co_path = correct_paths[trace['maze_type']]
        start_area = get_check_area(maze_type=trace['maze_type'], start_point=1, check_length=check_length)
        end_area = get_check_area(maze_type=trace['maze_type'], start_point=144, check_length=check_length)
        mid = co_path[int(len(co_path) * 0.5)]
        check_area = get_check_area(maze_type=trace['maze_type'], start_point=mid, check_length=mid_length)

        behav_nodes = cp.deepcopy(trace['correct_nodes'])
        behav_nodes = spike_nodes_transform(spike_nodes = behav_nodes, nx = 12)

        switch = 0
        check = 0

        # Check if lap-start or lap-end point frame by frame
        for k in range(behav_nodes.shape[0]):
            # Define checking properties for check area. If recorded point change from end area to start area without passing by the check area, we identify it 
            # as the start of a new lap
            
            if behav_nodes[k] in check_area:
                # Enter check area from end area.
                if switch == 2:
                    check = 0

                # Enter check area from start area.
                if switch == 1: 
                    check = 1

                # Abnormal case: that mice does not occur at start area before they enter the check area.
                if switch == 0:
                    check = 1
                    switch = 1 # Assume that switch state must be belong to {1,2}
                    beg_idx.append(0)

            if behav_nodes[k] in start_area:
                # if switch = 0
                if switch == 0:
                    beg_idx.append(k)
                    switch = 1
                if switch == 2 and check == 1:
                    end_idx.append(k-1)
                    beg_idx.append(k)
                    switch = 1

            if behav_nodes[k] in end_area:
                switch = 2   # state 2: at end area
                check = 1    # check = 1 represents mice have passed the check area

        if switch == 2:
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