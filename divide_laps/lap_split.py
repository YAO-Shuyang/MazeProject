import numpy as np
from mylib.divide_laps.reverselap import ReverseMazeLapSplit
from mylib.divide_laps.crossmazelap import CrossMazeLapSplit
from mylib.divide_laps.dspmazelap import DSPMazeLapSplit

def LapSplit(trace, behavior_paradigm = 'CrossMaze', **kwargs) -> tuple[np.ndarray, np.ndarray]:
    if behavior_paradigm == 'CrossMaze':
        return CrossMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm in ['ReverseMaze', 'HairpinMaze']:
        return ReverseMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm == 'DSPMaze':
        return DSPMazeLapSplit(trace, **kwargs)
    elif behavior_paradigm == 'SimpleMaze':
        return np.array([0], dtype = np.int64), np.array([trace['correct_time'].shape[0]-1], dtype = np.int64)