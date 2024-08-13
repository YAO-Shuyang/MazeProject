from mylib.local_path import f_CellReg_dsp, f2
from mylib.maze_utils3 import ReadCellReg
import numpy as np
import os
from tqdm import tqdm
import pickle

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
    
    for i, trace in enumerate(traces):
        neural_trajs.append(trace['neural_traj'][index_map[i, :]-1, :])
        route_ids.append(trace['traj_route_ids'])
        lap_ids.append(trace['traj_lap_ids'])
        segment_ids.append(trace['traj_segment_ids'])
        pos_trajs.append(trace['pos_traj'])
        day_ids.append(np.repeat(i, trace['pos_traj'].shape[0]))
        
    neural_trajs = np.concatenate(neural_trajs, axis=1)
    route_ids = np.concatenate(route_ids)
    lap_ids = np.concatenate(lap_ids)
    segment_ids = np.concatenate(segment_ids)
    pos_trajs = np.concatenate(pos_trajs)
    day_ids = np.concatenate(day_ids)
    
    res = {
        'MiceID': mouse,
        'neural_trajs': neural_trajs,
        'route_ids': route_ids,
        'lap_ids': lap_ids,
        'segment_ids': segment_ids,
        'pos_trajs': pos_trajs,
        'day_ids': day_ids
    }
    
    with open(
        os.path.join(
            os.path.join(os.path.dirname(f_CellReg_dsp['cellreg_folder'][file_index])), 
            "trace_mdays.pkl"
        ), 
        'wb'
    ) as handle:
        pickle.dump(res, handle)
        
    return res

if __name__ == '__main__':
    main(10209)
    main(10212)
    main(10224)
    main(10227)