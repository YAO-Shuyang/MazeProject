import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mylib.maze_graph import S2F, Father2SonGraph
from scipy.stats import pearsonr
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile

def extend_sonfield(
    sonfield: np.ndarray
):
    return np.concatenate([Father2SonGraph[k] for k in np.unique(S2F[sonfield-1])])

# Within Field Stability
def within_field_half_half_correlation(
    smooth_map_fir: np.ndarray,
    smooth_map_sec: np.ndarray,
    place_field_all: list[dict]
):
    HHC = []
    
    for n, placecell in enumerate(place_field_all):
        corr = {}

        for center in placecell.keys():
            if len(placecell[center]) <= 16:
                extended_field = extend_sonfield(placecell[center])
            else:
                extended_field = placecell[center]

            corr[center] = pearsonr(smooth_map_fir[n, extended_field-1], smooth_map_sec[n, extended_field-1])[0]

        HHC.append(corr)
        
    return HHC


# Within Field Stability (Odd-even correlation)
def within_field_odd_even_correlation(
    smooth_map_odd: np.ndarray,
    smooth_map_evn: np.ndarray,
    place_field_all: list[dict]
):
    OEC = []
    
    for n, placecell in enumerate(place_field_all):
        corr = {}

        for center in placecell.keys():
            if len(placecell[center]) <= 16:
                extended_field = extend_sonfield(placecell[center])
            else:
                extended_field = placecell[center]
                
            corr[center] = pearsonr(smooth_map_odd[n, extended_field-1], smooth_map_evn[n, extended_field-1])[0]
              
        OEC.append(corr)
    
    return OEC

if __name__ == '__main__':
    import pickle
    
    with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", "rb") as handle:
        trace = pickle.load(handle)
    
    within_field_half_half_correlation(smooth_map_fir = trace['smooth_map_fir'], smooth_map_sec = trace['smooth_map_sec'], place_field_all = trace['place_field_all'])
    