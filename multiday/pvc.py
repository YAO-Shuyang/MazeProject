from mylib.multiday.core import MultiDayCore
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mylib.maze_utils3 import DrawMazeProfile, Clear_Axes
from mylib.maze_graph import correct_paths
import pickle
import copy as cp
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from numba import jit

@jit(nopython=True)
def calc_pearsonr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) != len(y):
        raise ValueError("Input vectors must have the same length")
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator

@jit(nopython=True)
def get_pvc(maps1: np.ndarray, maps2: np.ndarray) -> np.ndarray:
    PVC = np.zeros((maps1.shape[1], maps2.shape[1]), dtype=np.float64)
    for i in range(maps1.shape[1]):
        for j in range(maps2.shape[1]):
            PVC[i, j] = calc_pearsonr(maps1[:, i], maps2[:, j])
    return PVC

class MultiDayPopulationVectorCorrelation:
    def __init__(
        self, 
        f: pd.DataFrame, 
        file_indices: np.ndarray, 
        keys: list[str] | None = ['old_map_clear', 'maze_type', 'is_placecell'],
        core: MultiDayCore | None = None,
    ) -> None:
        try:
            assert len(file_indices) > 1
        except:
            raise ValueError(f"File count should be more than 1, but {file_indices} was input.")
        
        if core is None:
            self.core = MultiDayCore.get_core(
                f=f,
                file_indices=file_indices,
                keys=keys
            )
        else:
            self.core = core
            
        self.PV = self.PVC = None

            
    def get_placecell_identity(self, index_map: np.ndarray) -> np.ndarray:
        res = np.zeros_like(index_map, dtype=np.int64)

        for i in range(index_map.shape[0]):
            for j in range(index_map.shape[1]):
                if index_map[i, j] != 0 and self.core.res['is_placecell'][i][int(index_map[i, j])-1] == 1:
                    res[i, j] = 1   
        return res

    def get_cell_pair(self, index_map: np.ndarray, pc_id: np.ndarray, i: int, j: int) -> np.ndarray:
        try:
            assert i < index_map.shape[0] and j < index_map.shape[0]
        except:
            raise ValueError(f"Index {i} and {j} must be less than {index_map.shape[0]}!")
        
        return np.where((index_map[i, :] !=0)&(index_map[j, :] !=0)&(pc_id[i, :] == 1)&(pc_id[j, :] == 1))[0]

    def get_pvc(
        self,
        f: pd.DataFrame, 
        file_indices: np.ndarray,         
        index_map: np.ndarray,
        occu_num: int = 13,
        align_idx: np.ndarray = np.arange(13),        
        keys: list[str] | None = ['old_map_clear', 'maze_type', 'is_placecell']
    ) -> np.ndarray:
        
        cellnum = np.where(index_map == 0, 0, 1)[align_idx]
        cellnum = np.nansum(cellnum, axis=0)    
            
        maze_type = self.core.res['maze_type'][0]
        CP = cp.deepcopy(correct_paths[int(maze_type)])
        
        pc_id = self.get_placecell_identity(index_map)
        pc_num = np.nansum(pc_id, axis=0)
        
        pc_ratio = pc_num / occu_num
        indices = np.where((cellnum == occu_num)&(pc_ratio >= 0.2))[0]

        PVC = np.zeros((occu_num, occu_num, CP.shape[0], CP.shape[0]), dtype=np.float64)
        
        print("    Calulating Population Vector Pearson Correlation...")
        for i in range(occu_num):
            for j in range(occu_num):
                vec_idx = self.get_cell_pair(index_map=index_map, pc_id=pc_id, i=i, j=j)
                pv1 = self.core.res['old_map_clear'][i][index_map[i, vec_idx].astype(np.int64)-1, :]# (self.core.res['old_map_clear'][n][index_map[n, indices], :].T * is_cells).T
                pv2 = self.core.res['old_map_clear'][j][index_map[j, vec_idx].astype(np.int64)-1, :]
                
                PVC[i, j, :, :] = get_pvc(pv1[:, CP-1], pv2[:, CP-1])
        print("  Done.")
        self.PVC = PVC      
        return PVC
    
    def visualize(
        self,
        save_loc: str | None = None,
        file_name: str | None = "Population Vector Correlation",
        is_show: bool = False,
        **kwargs
    ):
        if self.PVC is None:
            raise ValueError("self.PVC is None. Run self.get_pvc() first.")
        
        sessions = self.PVC.shape[0]
        bin_num = self.PVC.shape[2]
        print("    Visualizing...")
        fig, axes = plt.subplots(nrows=sessions, ncols=sessions, figsize=(sessions*3, sessions*3))
        
        for i in range(sessions):
            for j in range(sessions):
                axes[i, j] = Clear_Axes(axes[i, j])
                axes[i, j].imshow(self.PVC[i, j], **kwargs)
                
                
        if is_show or save_loc is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi=600)
            plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi=600)
            plt.close()
            
        print("  Figure is saved.")
    
    def visualize_pvc_heatmap(
        self,
        save_loc: str | None = None,
        file_name: str | None = "Correlation of PVC",
        is_show: bool = False,
        **kwargs
    ):
        if self.PVC is None:
            raise ValueError("self.PVC is None. Run self.get_pvc() first.")
        
        sessions = self.PVC.shape[0]
        bin_num = self.PVC.shape[2]
        print("    Visualizing...")
        fig = plt.figure(figsize=(4,3))
        ax = Clear_Axes(plt.axes())
        
        self.PVC[np.where(np.isnan(self.PVC))] = 0
        corr = np.zeros((sessions, sessions), dtype=np.float64)
        for i in range(sessions):
            for j in range(sessions):
                for k in range(sessions):
                    for l in range(sessions):
                        corr[i, j] = np.nanmean(self.PVC[i, j][(np.arange(bin_num), np.arange(bin_num))])
        
        im = ax.imshow(corr, **kwargs)
        cbar = fig.colorbar(im, ax=ax)
                
        if is_show or save_loc is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi=600)
            plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi=600)
            plt.close()
            
        print("  Figure is saved.")
        
        return corr
        