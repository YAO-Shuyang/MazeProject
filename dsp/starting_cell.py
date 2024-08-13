import numpy as np
from sklearn.decomposition import PCA
import copy as cp

from mylib.dsp.neural_traj import lda_dim_reduction
from mylib.maze_utils3 import spike_nodes_transform
from mylib.maze_utils3 import GetDMatrices, SP_DSP, DSP_NRG

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



if __name__ == '__main__':
    import pickle
    from mylib.local_path import f2
    
    for i in range(len(f2)):
        if i != 26:
            continue
        print(i, f2['MiceID'][i], f2['date'][i])
        
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
        
        get_initial_traj(trace)
        