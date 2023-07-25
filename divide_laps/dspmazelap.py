import numpy as np
import copy as cp


def DSPMazeLapSplit(trace: dict, **kwargs):
    assert 'lap beg time' in trace.keys() and 'lap end time' in trace.keys()

    beg_time_ori, end_time_ori = trace['lap beg time'], trace['lap end time']
    behav_time = cp.deepcopy(trace['correct_time'])
    beg, end = np.zeros_like(beg_time_ori, dtype=np.int64), np.zeros_like(end_time_ori, dtype=np.int64)
    laps = beg_time_ori.shape[0]

    for i in range(laps):
        beg[i] = np.where(behav_time >= beg_time_ori[i])[0][0]
        end[i] = np.where(behav_time <= end_time_ori[i])[0][-1]

    return beg, end


