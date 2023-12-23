import numpy as np
import copy as cp
from tqdm import tqdm
from mylib.calcium.field_criteria import GetPlaceField
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile, maze_graphs, mkdir
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import os

from mazepy.behav.graph import Graph
from numba import jit

def _get_dis(p1: np.ndarray, p2: np.ndarray, G: Graph) -> float:
    _p1, _p2 = p1/4+0.125, p2/4+0.125
    x1, y1, x2, y2 = _p1[0], _p1[1], _p2[0], _p2[1]
    try:
        return G.shortest_distance((x1, y1), (x2, y2))*8
    except:
        raise ValueError(f"{p1} -> {_p1} and {p2} -> {_p2} do not work.")

class PlaceFieldNode:
    def __init__(
        self,
        center: int,
        G: Graph,
        field: np.ndarray,
        shift: float | None = None
    ):
        self.center = center
        self.field = field
        self.shift = shift
        self.root = None
        self.center_pos = np.array([(center - 1) % 48, (center - 1) // 48], np.float64)
        self.dis_to_start = _get_dis(self.center_pos, np.array([0, 0]), G=G)

    def _is_overlap_field(self, field: np.ndarray, ref_field: np.ndarray) -> bool:
        """_is_overlap_field: to check if the field is overlapped with the reference field

        Parameters
        ----------
        field : np.ndarray
            The field to be checked.
        ref_field : np.ndarray
            The reference field.

        Returns
        -------
        bool
            Whether the field is overlapped with the reference field.
        """
        return np.any(field == ref_field)
        
    @staticmethod
    def is_overlap(field: np.ndarray, ref_field: np.ndarray) -> bool:
        """is_overlap_field: to check if the field is overlapped with the reference field"""
        #return np.any(field == ref_field)
        return len(np.setdiff1d(field, ref_field)) < len(field)
         
    def _get_shift(self, root, G: Graph) -> float:
        """_get_shift: to get the shift of the field compared to the reference field

        Parameters
        ----------
        root : _type_
            The reference field
        G : Graph
            The maze graph

        Returns
        -------
        float
            The shift.
        """
        return self.dis_to_start - root.dis_to_start
        
    def set_shift(self, root, G: Graph):
        if self._is_overlap_field(self.field, root.field):
            self.root = root
            self.shift = self._get_shift(root, G)
            return self.shift
        else:
            print(f"Field {self.field} is not overlapped with the reference field {root.field}.")
    
import scipy.stats
class SingleTrialFieldMatcher:
    def __init__(self) -> None:
        pass
    
    def _get_trend(self, slope: float, pvalue: float):
        SIGNIFICANT = pvalue < 0.05
        FORWARD = slope > 0
        BACKWARD = slope < 0
        if FORWARD and SIGNIFICANT:
            return "Forward"
        elif BACKWARD and SIGNIFICANT:
            return "Backward"
        else:
            return "Retain"
    
    def fit_shift_trend(self, shifts: np.ndarray):
        slopes = np.zeros(shifts.shape[1], dtype=np.float64)
        pvalues = np.zeros(shifts.shape[1], dtype=np.float64)
        rvalues = np.zeros(shifts.shape[1], dtype=np.float64)
        trends = []
        
        x = np.arange(len(shifts))
        for y in range(shifts.shape[1]):
            idx = np.where(np.isnan(shifts[:, y])==False)[0]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[idx], shifts[idx, y])
            slopes[y] = slope
            pvalues[y] = p_value
            rvalues[y] = r_value
            trends.append(self._get_trend(slope, p_value))
        
        return slopes, pvalues, rvalues, trends
            
    
    @staticmethod
    def match_fields(
        trace: dict,
        n: int,
        G: Graph,
        thre_type: int=2, 
        parameter: float = 0.4, 
        events_num_crit: int = 10
    ):  
        """match_fields: to match single trial fields with the reference field.

        Parameters
        ----------
        trace : dict
            Data which contains basic information of this recording session.
        n : int
            The index of this cell.
        global_map : np.ndarray
            The entire map of this cell within this session.
            shape: (2304, )
        single_trial_maps : np.ndarray
            The single trial maps of this cell within this session.
            shape: (trial_num, 2304)
        """
        global_map = cp.deepcopy(trace['smooth_map_all'][n, :])
        single_trial_maps = cp.deepcopy(trace['smooth_map_split'][:, n, :])
        
        ref_fields = GetPlaceField(trace, n, thre_type=thre_type, parameter=parameter, events_num_crit=events_num_crit, smooth_map=global_map)
        roots = []
        for k in ref_fields.keys():
            root = PlaceFieldNode(center = k, G=G, field = np.array(ref_fields[k], dtype=np.int64))
            roots.append(root)
        
        shifts = np.zeros([single_trial_maps.shape[0], len(ref_fields.keys())], dtype=np.float64) * np.nan
        sizes = np.zeros([single_trial_maps.shape[0], len(ref_fields.keys())], dtype=np.float64) * np.nan
        centers = np.zeros([single_trial_maps.shape[0], len(ref_fields.keys())], dtype=np.float64) * np.nan
        
        for trial in range(single_trial_maps.shape[0]):
            fields = GetPlaceField(trace, n, thre_type=thre_type, parameter=parameter, events_num_crit=0, smooth_map=single_trial_maps[trial, :], need_events_num=False)
            for k in fields.keys():
                node = PlaceFieldNode(center=k, G=G, field=np.array(fields[k], dtype=np.int64))
                for i, ref in enumerate(roots):
                    if PlaceFieldNode.is_overlap(node.field, ref.field):
                        if np.isnan(shifts[trial, i]):
                            shifts[trial, i] = node.dis_to_start - ref.dis_to_start
                            sizes[trial, i] = node.field.shape[0]
                            centers[trial, i] = node.center
                        else:
                            if shifts[trial, i] > node.dis_to_start - ref.dis_to_start:
                                shifts[trial, i] = node.dis_to_start - ref.dis_to_start
                                sizes[trial, i] = node.field.shape[0]
                                centers[trial, i] = node.center
        
        matcher = SingleTrialFieldMatcher()
        slopes, pvalues, rvalues, trends = matcher.fit_shift_trend(shifts)
                      
        return {
            'shifts': shifts,
            'sizes': sizes,
            'centers': centers,
            'slopes': slopes,
            'pvalues': pvalues,
            'rvalues': rvalues,
            'trends': trends
        }
                               

def visulize_field_centers(info: dict, maze_type: int, save_loc: str):

    field_num = info['centers'].shape[1]
    trial_num = info['centers'].shape[0]
    fig = plt.figure(figsize=(4, 4))
    ax = Clear_Axes(plt.axes())
    mkdir(save_loc)
    
    for n in tqdm(range(field_num)):
        
        colors = sns.color_palette("rainbow", trial_num)
        x, y = (info['centers'][:, n]-1) % 48 + np.random.rand(trial_num)*0.5-0.25, (info['centers'][:, n]-1) // 48 + np.random.rand(trial_num)*0.5-0.25
        t = np.arange(trial_num)
        DrawMazeProfile(maze_type=maze_type, axes=ax, color='black', linewidth=1, nx=48)
        sns.scatterplot(
            x = x,
            y = y,
            hue=t,
            palette=colors,
            legend=False,
            size=t,
            sizes=(3, 3.001),
            linewidths = 0,
            edgecolor = None,
            alpha=0.6
        )
        plt.savefig(os.path.join(save_loc, f"field {n+1}.png"), dpi=2400)
        plt.savefig(os.path.join(save_loc, f"field {n+1}.svg"), dpi=2400)
        ax.clear()
    
    plt.close()
    

if __name__ == '__main__':
    import pickle
    
    with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    info = SingleTrialFieldMatcher.match_fields(
        trace,
        n = 0,
        G = Graph(12, 12, cp.deepcopy(maze_graphs[(trace['maze_type'],12)])),
        parameter=0.1
    )
    visulize_field_centers(info, maze_type=trace['maze_type'], save_loc=r"E:\Data\Cross_maze\10209\20230728\session 2\single trial centers")
    """       
    #trace = single_trial_field_center(trace)
    with open(r"E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl", 'wb') as f:
        pickle.dump(trace, f) 
    """

    