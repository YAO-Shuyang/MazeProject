import numpy as np
import pandas as pd
import copy as cp

def read_time_stamp(file_dir: str, key_word: str = 'Time Stamp (ms)') -> np.ndarray:
    f = pd.read_csv(file_dir)
    return np.array(f[key_word])

def dlc_position_generation(dlc: dict, dtype: str = 'prefer', prefer_body_part: str = 'bodypart1'):
    """dlc_position_generation _summary_

    Generation of position from the dlc data

    Parameters
    ----------
    dlc : dict
        dlc coordinates dictionary, e.g.,  {'bodypart1': numpy.array([...]), 'bodypart2': numpy.array([...]), 
                                            'bodypart3': numpy.array([...]), 'objectA': numpy.array([...])}
    dtype : str, optional
        The way to generate the position of mice, by default 'prefer'
        Only {'prefer', 'mass'}
        'prefer' method simply adopts the positions of one point (prefered point) among 
          those bodyparts labeled as the position of a freely moving mouses.
        'mass' method adopts the mass point of all bodyparts labeled as the position of
          a freely moving mouse.
    
    """
    
    if dtype == 'prefer':
        return cp.deepcopy(dlc[prefer_body_part])

    elif dtype == 'mass':
        pos = None
        nkey = 0
        for b in dlc.keys():
            pos = pos + dlc[b] if pos is not None else dlc[b]
            nkey += 1
        return pos/nkey