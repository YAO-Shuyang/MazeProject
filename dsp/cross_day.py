from mylib.local_path import f_CellReg_dsp, f2
from mylib.maze_utils3 import ReadCellReg
import numpy as np
import os
from tqdm import tqdm
import pickle
import scipy.stats

def main(
    mouse: int = None
):
    file_index = np.where(f_CellReg_dsp['MiceID'] == mouse)[0][0]
    index_map = ReadCellReg(
        f_CellReg_dsp['cellreg_folder'][file_index]
    )
    
    cell_num = np.sum(np.where(index_map != 0, 1, 0), axis=0)
    
    cell_indices = np.where(cell_num == 7)[0]
    index_map = index_map[:, cell_indices].astype(np.int64)
    
    file_indices = np.where(f2['MiceID'] == mouse)[0]
    
    traces = []
    for i in tqdm(file_indices):
        with open(f2['Trace File'][i], 'rb') as handle:
            traces.append(pickle.load(handle))
    
    neural_trajs = []
    route_ids = []
    lap_ids = []
    segment_ids = []
    pos_trajs = []
    day_ids = []
    lap_ids_cum = []
    
    cum_sum = 0
    for i, trace in enumerate(traces):
        neural_trajs.append(trace['neural_traj'][index_map[i, :]-1, :])
        route_ids.append(trace['traj_route_ids'])
        lap_ids.append(trace['traj_lap_ids'])
        segment_ids.append(trace['traj_segment_ids'])
        pos_trajs.append(trace['pos_traj'])
        day_ids.append(np.repeat(i, trace['pos_traj'].shape[0]))
        lap_ids_cum.append(trace['traj_lap_ids'] + cum_sum)
        
        cum_sum += np.max(trace['traj_lap_ids']) + 1
        
    neural_trajs = np.concatenate(neural_trajs, axis=1)
    route_ids = np.concatenate(route_ids)
    lap_ids = np.concatenate(lap_ids)
    segment_ids = np.concatenate(segment_ids)
    pos_trajs = np.concatenate(pos_trajs)
    day_ids = np.concatenate(day_ids)
    lap_ids_cum = np.concatenate(lap_ids_cum)
    
    # sort neural trajectory
    corr = np.zeros(neural_trajs.shape[0])
    for i in tqdm(range(neural_trajs.shape[0])):
        corr[i] = scipy.stats.pearsonr(
            traces[0]['node 9']['smooth_map_all'][index_map[0, i]-1, :],
            traces[6]['node 9']['smooth_map_all'][index_map[6, i]-1, :]
        )[0]
    
    not_nan = np.where(np.isnan(corr) == False)[0]
    sort_idx = np.argsort(corr[not_nan])[::-1]
    neural_trajs = neural_trajs[not_nan, :]
    neural_trajs = neural_trajs[sort_idx, :]
    print(corr[not_nan][sort_idx])
        
    res = {
        'MiceID': mouse,
        'neural_traj': neural_trajs,
        'traj_route_ids': route_ids,
        'traj_lap_ids': lap_ids,
        'traj_segment_ids': segment_ids,
        'pos_traj': pos_trajs,
        'traj_day_ids': day_ids,
        'traj_lap_ids_cum': lap_ids_cum
    }
    
    with open(f_CellReg_dsp['Trace File'][file_index],'wb') as handle:
        pickle.dump(res, handle)
        
    return res

if __name__ == '__main__':
    main(10209)
    main(10212)
    main(10224)
    main(10227)