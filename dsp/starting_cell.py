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
    def __init__(self, smooth_map_all: list[dict]) -> None:
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
        
        self._smooth_maps = [
            smooth_map_all[i][self._bin[i][np.argsort(self._dis[i])]] for i in range(10)
        ]
        """
        from mylib.maze_utils3 import DrawMazeProfile
        for i in range(10):
            plt.figure()
            ax = plt.axes()
            #maps = np.full_like(smooth_map_all[i], np.nan)
            maps = smooth_map_all[i]
            #maps[self._bin[i][np.argsort(self._dis[i])]] = np.arange(self._bin[i].shape[0])
            maps = np.reshape(maps, [48, 48])
            ax.imshow(maps)
            DrawMazeProfile(maze_type=1, axes=ax,color='black', nx=48)
            ax.set_title(f"Route {i+1}")
            plt.show()
        """    
            
    def update_smooth_map(self, smooth_map_all: list[dict]):
        self._smooth_maps = [
            smooth_map_all[i][self._bin[i][np.argsort(self._dis[i])]] for i in range(10)
        ]
        
    def get_map(self):
        return np.vstack([self._smooth_maps[i][:CP_DSP[3].shape[0]*16] for i in range(10)])
        
    def shuffle_maps(self):
        #shuf_maps = []
        shuf_maps = [np.roll(self._smooth_maps[i], np.random.randint(0, self._smooth_maps[i].shape[0])) for i in range(10)]
        """
        for i in range(10):
            if i in [0]:
                shift_route_0 = np.random.randint(0, self._smooth_maps[0].shape[0])
                shuf_maps.append(np.roll(self._smooth_maps[0], shift_route_0))
            elif i in [4, 5, 9]:
                shuf_maps.append(np.roll(self._smooth_maps[i], shift_route_0))
            else:
                shuf_maps.append(np.roll(self._smooth_maps[i], np.random.randint(0, self._smooth_maps[i].shape[0])))
        """
        return np.vstack([shuf_maps[i][:CP_DSP[3].shape[0]*16] for i in range(10)])
    
    def calc_corr(self, smooth_maps: np.ndarray):
        """
        smooth_maps: 10 * length
        """
        n_steps = int(smooth_maps.shape[1]/16)
        corr_steps = np.zeros(n_steps)
        for i in range(n_steps):
            corr = pearsonr_pairwise(
                smooth_maps[:, i*16:(i+1)*16], axis=1
            )
            corr[np.isnan(corr)] = -1
            corr_steps[i] = np.nanmean(corr[np.where(np.triu(corr, k=1)!=0)])

        return corr_steps
    
    def is_starting_cell(self, n_shuffle=1000, is_return_shuf = False, null_hypo: np.ndarray = np.arange(10)):
        smooth_map = self.get_map()
        real_corr = self.calc_corr(smooth_map[null_hypo, :])
        
        """
        x = np.linspace(8, smooth_map.shape[1]-8, int(smooth_map.shape[1]/16))
        ax = plt.axes()
        ax.imshow(smooth_map[null_hypo, :])
        ax.set_aspect("auto")
        ax.plot(x, real_corr, color = 'white')
        ax.invert_yaxis()
        plt.show()
        """
        shuf_corr = np.zeros((n_shuffle, real_corr.shape[0]))
        for i in range(n_shuffle):
            shuf_maps = self.shuffle_maps()
            shuf_corr[i, :] = self.calc_corr(shuf_maps[null_hypo, :])
            
        P = np.array([
            np.where(shuf_corr[:, i] - real_corr[i] > 0)[0].shape[0] / n_shuffle for i in range(real_corr.shape[0])
        ])
        
        if is_return_shuf:
            return real_corr, P, shuf_corr
        else:
            return real_corr, P
    
    @staticmethod
    def visualize_shuffle_results(
        trace: dict,
        n: int,
        n_shuffle:int = 1000,
        save_loc:str = None,
        percent: float = 95,
        file_name: str = None,
        is_show: bool = False,
        null_hypo: np.ndarray = np.arange(10)
    ):
        cell = StartingCell([trace[f"node {j}"]["smooth_map_all"][n, :] for j in range(10)])
        null_hypo = cell.get_null_hypo(field_reg=trace['field_reg_modi'][:, np.where(trace['field_info'][0, :, 0] == n+1)[0]], 
                                       place_field_all=trace['place_field_all'][n])
        print(null_hypo+1)
        corr, p, shuf_corr = cell.is_starting_cell(n_shuffle=n_shuffle, is_return_shuf=True, null_hypo=null_hypo)
    
        v_min = np.percentile(shuf_corr, (100-percent)/2, axis=0)
        v_max = np.percentile(shuf_corr, 100 - (100-percent)/2, axis=0)
        x = np.arange(corr.shape[0])
        
        fig = plt.figure(figsize=(6, 2))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.plot(x, corr, color = 'black', linewidth = 0.5)
        ax.fill_between(x, v_min, v_max, color='gray', edgecolor=None, alpha=0.8)
        
        print(f"{int(trace['MiceID'])}_{int(trace['date'])}_Cell {n+1}:\n"
              f"Correla.: {corr}\n"
              f"P-values: {p}")
        
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
        self, field_reg: np.ndarray, 
        place_field_all: list[dict]
    ) -> np.ndarray:
        print(field_reg.shape[1], len(place_field_all.keys()))
        assert field_reg.shape[1] == len(place_field_all.keys())
        field_reg[field_reg > 1] = 1
        
        field_center = np.array(list(place_field_all.keys()))
        dis = self._D[self._SP, :][:, np.array([k for k in place_field_all.keys()])-1] * field_reg
        dis[dis == 0] = np.nan
        
        nearest = np.nanmin(dis, axis=1)
        nearest_keys = 
        return np.where(nearest <= 30)[0]
    
    @staticmethod
    def classify(trace, n_shuffle=1000):
        is_startingcell = np.zeros(trace['n_neuron'])
        starting_corr = np.zeros((trace['n_neuron'], CP_DSP[3].shape[0]))
        starting_p = np.zeros((trace['n_neuron'], CP_DSP[3].shape[0]))
        cell = None
        
        for i in tqdm(range(trace['n_neuron'])):
            if cell is None:
                cell = StartingCell([trace[f"node {j}"]["smooth_map_all"][i, :] for j in range(10)])
            else:
                cell.update_smooth_map([trace[f"node {j}"]["smooth_map_all"][i, :] for j in range(10)])
            starting_corr[i, :], starting_p[i, :] = cell.is_starting_cell(n_shuffle=n_shuffle)
        
        trace['starting_corr'] = starting_corr
        trace['starting_p'] = starting_p
        return trace
            
        

if __name__ == '__main__':
    import pickle
    from mylib.local_path import f2
    
    for i in range(len(f2)):
        if i != 21:
            continue
        print(i, f2['MiceID'][i], f2['date'][i])
        
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        StartingCell.visualize_shuffle_results(
            trace,
            n=82-1,
            is_show=True,
            null_hypo=np.array([2, 5, 6, 7, 8])
        )