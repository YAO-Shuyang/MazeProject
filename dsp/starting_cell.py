import numpy as np
from sklearn.decomposition import PCA
import copy as cp

from mylib.dsp.neural_traj import lda_dim_reduction
from mylib.maze_utils3 import spike_nodes_transform, GetDMatrices, mkdir, Clear_Axes
from mylib.maze_utils3 import GetDMatrices, SP_DSP, DSP_NRG, maze_graphs, Father2SonGraph, CP_DSP
from mazepy.basic._corr import pearsonr_pairwise
from scipy.stats import pearsonr

from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_son_area(area: np.ndarray):
    return np.concatenate([Father2SonGraph[i] for i in area])

def get_initial_traj(trace, limit=10):
    neural_traj = cp.deepcopy(trace['neural_traj'])
    lap_ids = cp.deepcopy(trace['traj_lap_ids'])
    pos_traj = spike_nodes_transform(trace['pos_traj'], 12).astype(np.int64)
    
    idx = np.where(np.ediff1d(lap_ids) != 0)[0]
    beg = np.concatenate([[0], idx+1])
    end = np.concatenate([idx+1, [lap_ids.shape[0]]])
    
    D = GetDMatrices(1, 48)
    
    ego_pos_traj = np.zeros_like(trace['pos_traj'], dtype=np.float64)
    for i in range(ego_pos_traj.shape[0]):
        ego_pos_traj[i] = DSP_NRG[trace['traj_route_ids'][i]][
            pos_traj[i]
        ]

    include_indices = np.where(ego_pos_traj <= limit)[0]
    
    pc_idx = np.unique(
        np.concatenate(
            [np.where(trace[f'node {i}']['is_placecell'] == 1)[0] for i in range(10)]
        )
    )
    
    return {
        "neural_traj": neural_traj[:, include_indices],
        "traj_lap_ids": lap_ids[include_indices],
        "traj_segment_ids": trace['traj_segment_ids'][include_indices],
        "traj_route_ids": trace['traj_route_ids'][include_indices],
        "pos_traj": trace['pos_traj'][include_indices],
        "ego_pos_traj": ego_pos_traj[include_indices],
        "pc_idx": pc_idx,
        "Spikes": trace['Spikes'][:, include_indices],
    }

class StartingCell:
    def __init__(self, place_field_all: list[dict]) -> None:
        """
        Parameters
        ----------
        place_field_all : list[dict]
            A list contains place fields on ten routes for a given neuron, and each of the inside list has k keys,
            where k is the number of fields on that route and the value of keys is the field center.
        """
        self._G = maze_graphs[(1, 48)]
        self._D = GetDMatrices(1, 48)
        self._SP = np.array([
            SP_DSP[0], SP_DSP[1], SP_DSP[2], SP_DSP[3], SP_DSP[0],
            SP_DSP[0], SP_DSP[4], SP_DSP[5], SP_DSP[6], SP_DSP[0]
        ])-1
        self._R = np.array([0, 1, 2, 3, 0, 0, 4, 5, 6, 0])
        self._bin = [
            get_son_area(CP_DSP[self._R[i]])-1 for i in range(10)
        ]
        self._dis = [self._D[self._SP[i], self._bin[i]] for i in range(10)]
        self._max_dis = np.array([np.max(i) for i in self._dis])
        
        self._null = self.get_null_hypo(place_field_all)
        self._place_field_all = place_field_all
        self._init_field_range()
    
    def _get_range(self, field_area: np.ndarray, start_point: int) -> tuple[int, int]:
        dis = self._D[start_point, field_area-1]
        return np.min(dis), np.max(dis)
    
    def _init_field_range(self):
        if len(self._null) <= 1:
            return
        
        max_field = np.max([len(field.keys()) for field in self._place_field_all])
        self._lef_bounds = np.full((10, max_field), np.nan)
        self._rig_bounds = np.full((10, max_field), np.nan)
        
        for i in self._null:
            for j, k in enumerate(self._place_field_all[i].keys()):
                lef, rig = self._get_range(self._place_field_all[i][k], self._SP[i])
                self._lef_bounds[i, j] = lef
                self._rig_bounds[i, j] = rig
        
        self._lef_bounds = self._lef_bounds[self._null, :]
        self._rig_bounds = self._rig_bounds[self._null, :]
        
        self._lef_bounds, self._rig_bounds = self.sort_bounds(self._lef_bounds, self._rig_bounds)
    
    # Sort boundarys based on distance
    def sort_bounds(self, lef_bounds: np.ndarray, rig_bounds: np.ndarray):
        for i in range(lef_bounds.shape[0]):
            sort_idx = np.argsort(lef_bounds[i])
            lef_bounds[i, :], rig_bounds[i, :] = lef_bounds[i, sort_idx], rig_bounds[i, sort_idx]
        return lef_bounds, rig_bounds
    
    def calc_overlapping(self, lef: np.ndarray, rig: np.ndarray):
        overlap = 0
        for d in range(1, lef.shape[0]):
            res_1 = rig[:lef.shape[0]-d] - lef[d:]
            res_2 = rig[d:] - lef[:lef.shape[0]-d]
            
            min_dis = np.min(np.vstack([res_1, res_2, rig[:lef.shape[0]-d] - lef[:lef.shape[0]-d], rig[d:] - lef[d:]]), axis=0)
            overlap += np.sum(min_dis[min_dis > 0])
            
        _range = np.sort(rig-lef)[::-1]
        if np.isnan(overlap):
            print(lef, rig)
        
        return overlap / np.nansum(_range[1:] * np.arange(1, lef.shape[0]))
            
    def get_shift_distance(self, n_shuffle=1000):
        shift_dis = np.zeros((self._null.shape[0], n_shuffle))
        
        for i, n in enumerate(self._null):
            shift_dis[i, :] = np.random.rand(n_shuffle) * self._max_dis[n]
        
        idx = np.where(self._R[self._null] == 0)[0]
        if idx.shape[0] > 1:
            shift_dis[idx, :] = shift_dis[idx[0], :]
        return shift_dis
    
    def shuffle_fields(self, lef_bounds: np.ndarray, rig_bounds: np.ndarray, shift_dis: int):
        lef, rig = lef_bounds + shift_dis[:, np.newaxis], rig_bounds + shift_dis[:, np.newaxis]
        
        max_dis = self._max_dis[self._null][:, np.newaxis]
        idx = np.where((lef - max_dis >= 0))[0]
        lef[idx] -= max_dis[idx]
        rig[idx] -= max_dis[idx]
        
        idx = np.where((rig - max_dis > 0)&(lef < max_dis))[0]
        lef[idx] = 0
        rig[idx] -= max_dis[idx]
        
        lef, rig = self.sort_bounds(lef, rig)
        
        return lef[:, 0], rig[:, 0]
            
    def update(self, place_field_all: list[dict]):
        self._place_field_all = place_field_all
        self._null = self.get_null_hypo(place_field_all)
        self._init_field_range()
    
    def is_starting_cell(self, n_shuffle=1000, is_return_shuf = False):
        if self.null.shape[0] <= 1:
            if is_return_shuf:
                return np.nan, np.nan, np.full(n_shuffle, np.nan)
            else:
                return np.nan, np.nan
            
        lef_bounds, rig_bounds = self._lef_bounds, self._rig_bounds
        real_oi = self.calc_overlapping(lef_bounds[:, 0], rig_bounds[:, 0])

        shuf_oi = np.zeros(n_shuffle)
        shift_dis = self.get_shift_distance(n_shuffle)
        for i in range(n_shuffle):
            lef, rig = self.shuffle_fields(lef_bounds, rig_bounds, shift_dis[:, i])
            shuf_oi[i] = self.calc_overlapping(lef, rig)
            
        P = np.mean(real_oi < shuf_oi)
        
        if is_return_shuf:
            return real_oi, P, shuf_oi
        else:
            return real_oi, P
        
    def get_field_center(self):
        try:
            return np.mean((self._lef_bounds[:, 0] + self._rig_bounds[:, 0])/2)
        except:
            return np.nan
        
    @property
    def null(self):
        return self._null
    
    @staticmethod
    def visualize_shuffle_results(
        trace: dict,
        n: int,
        n_shuffle:int = 1000,
        save_loc:str = None,
        percent: float = 95,
        file_name: str = None,
        is_show: bool = False
    ):
        cell = StartingCell(place_field_all=[trace[f"node {k}"]['place_field_all'][n] for k in range(10)])
        overlap, p, shuf_overlap = cell.is_starting_cell(n_shuffle=n_shuffle, is_return_shuf=True)
    
        v_min = np.percentile(shuf_overlap, (100-percent)/2, axis=0)
        v_max = np.percentile(shuf_overlap, 100 - (100-percent)/2, axis=0)
        
        fig = plt.figure(figsize=(6, 2))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.hist(shuf_overlap, bins=100, range=(0, 1), color='gray')
        ax.axvline(overlap, color='red', linewidth=0.5)
        ax.axvline(v_min, ls=':', color='black', linewidth=0.5)
        ax.axvline(v_max, ls=':', color='black', linewidth=0.5)
        ax.set_xlim([0, 1])
        ax.set_title(f"Cell {n+1} {int(trace['MiceID'])} {int(trace['date'])}\n"
                     f"P-value: {round(p, 3)}  Overlap: {round(overlap, 3)}")
        
        print(f"{int(trace['MiceID'])}_{int(trace['date'])}_Cell {n+1}:\n"
              f"Overlap: {overlap}\n"
              f"P-values: {p}\n"
              f"Null: {cell.null+1}")
        
        if is_show:
            plt.show()
        else:
            if save_loc is None:
                save_loc = os.path.join(trace['p'], "StartingCellShuffle")
            mkdir(save_loc)
        
            if file_name is None:
                file_name = f"{int(trace['MiceID'])}_{int(trace['date'])}_Cell {n+1}" 
            plt.savefig(os.path.join(save_loc, file_name+".png"), dpi=600)          
            plt.savefig(os.path.join(save_loc, file_name+".svg"), dpi=600)      
            plt.close()    
        
    def get_null_hypo(
        self, 
        place_field_all: list[dict]
    ) -> np.ndarray:
        field_center = []
        relative_pos = []
        for i in range(10):
            if len(place_field_all[i].keys()) == 0:
                field_center.append(-1)
                relative_pos.append(np.nan)
            else:
                center_dis = self._D[np.array(list(place_field_all[i].keys()))-1, self._SP[i]]
                field_center.append(list(place_field_all[i].keys())[np.argmin(center_dis)])
                relative_pos.append(np.min(center_dis))

        dis = np.array(relative_pos, dtype=np.float64)
        if np.where(np.isnan(dis) == False)[0].shape[0] <= 1:
            return np.array([])
        
        ref_center = np.nanargmin(dis)
        
        # Check overlap:
        candidate = [ref_center]
        ref_lef, ref_rig = self._get_range(place_field_all[ref_center][int(field_center[ref_center])], self._SP[ref_center])
        for i in range(10):
            if i == ref_center:
                continue
            
            if np.isnan(dis[i]):
                continue
            else:
                test_range = place_field_all[i][int(field_center[i])]
                lef, rig = self._get_range(test_range, self._SP[i])
                if min(rig - ref_lef, ref_rig - lef) > 0:
                    candidate.append(i)
        
        if len(candidate) == 1 or dis[ref_center] > 100:
            return np.array([])
        else:
            return np.array(sorted(candidate), dtype=np.int64)
    
    @staticmethod
    def classify(trace, n_shuffle=10000, p_thre = 0.01):
        n_neuron = trace['n_neuron']
        SC_EncodePath = []
        SC_OI = np.zeros(n_neuron)
        SC_P = np.zeros(n_neuron)
        cell = None
        field_centers = []
        
        for i in tqdm(range(n_neuron)):
            if cell is None:
                cell = StartingCell(place_field_all=[trace[f"node {k}"]['place_field_all'][i] for k in range(10)])
            else:
                cell.update(place_field_all=[trace[f"node {k}"]['place_field_all'][i] for k in range(10)])
                
            SC_OI[i], SC_P[i] = cell.is_starting_cell(n_shuffle=n_shuffle)
            SC_EncodePath.append(cell.null)
            field_centers.append(cell.get_field_center())
        
        trace['SC_OI'] = SC_OI
        trace['SC_P'] = SC_P
        trace['SC_P_thre'] = p_thre
        trace['SC_P_shuffle'] = n_shuffle
        trace['SC'] = np.where((SC_P < p_thre)&(np.isnan(SC_P) == False), 1, 0)
        trace['SC_EncodePath'] = SC_EncodePath
        trace['SC_EncodePath_Num'] = np.array([len(i) for i in SC_EncodePath], np.int64)
        trace['SC_FieldCenter'] = np.array(field_centers, np.float64)
        
        return trace
            
        

if __name__ == '__main__':
    import pickle
    from mylib.local_path import f2
    
    for i in range(len(f2)):
        if i <= 27:
            continue
        
        print(i, f2['MiceID'][i], f2['date'][i])
        
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        trace = StartingCell.classify(trace)
        
        with open(f2['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
        
        n = 15
        print(f"Cell {n}:\n"
              f"  is OI: {trace['SC'][n-1]}\n"
              f"  OI:    {trace['SC_OI'][n-1]}\n"
              f"  P:     {trace['SC_P'][n-1]}\n"
              f"  Encoded Path: {trace['SC_EncodePath'][n-1]+1}")