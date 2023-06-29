import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from mylib.maze_utils3 import DrawMazeProfile, Add_NAN, Clear_Axes

def calc_spike_time(spikes: np.ndarray, 
                    behav_time: np.ndarray, 
                    ms_time: np.ndarray):
    idx = np.where(spikes == 1)[0]
    spike_time = ms_time[idx]
    behav_idx = np.zeros(idx.shape[0], dtype=np.float64)
    
    for i in range(idx.shape[0]):
        k = np.where(behav_time<=spike_time[i])[0][-1]
        t1, t2 = behav_time[k], spike_time[i]
        if np.abs(t2-t1) <= 500: # ms
            behav_idx[i] = k
        else:
            behav_idx[i] = np.nan
    behav_idx = behav_idx[np.where(np.isnan(behav_idx) == False)[0]]
    behav_idx = behav_idx.astype(np.int64)
    return behav_idx
    
def TraceMap3D(trace, neuron_id = None, save_loc = None):
    Spikes = trace['Spikes']
    ms_time_behav = trace['ms_time_behav']
    
    pos, behav_time = Add_NAN(trace['processed_pos_new'], trace['behav_time'])

    
    x, y = pos[:, 0]/20 - 0.5, pos[:, 1]/20 - 0.5
    
    
    fig = plt.figure(facecolor=None, edgecolor=None, figsize=(2,6))
    ax: Axes3D = fig.add_subplot(projection='3d', alpha = 0, box_aspect = [4,4,20])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_zlabel("time / s")
    ax.set_aspect('equalxy')
    ax.plot(x, y, behav_time/1000, 'o', color = 'gray', linewidth = 1, markeredgewidth = 0, markersize = 2)
    ax = DrawMazeProfile(axes = ax, color = 'brown', nx = 48, maze_type=trace['maze_type'], linewidth=2)
    
    
    if neuron_id is None:
        for i in range(trace['n_neuron']):
            #rate_map = trace['smooth_map_all'][i, :]
            spike_idx = calc_spike_time(Spikes[i], behav_time=behav_time, ms_time=ms_time_behav)
            ms_x, ms_y = x[spike_idx], y[spike_idx]
            ms_t = behav_time[spike_idx]
            #ax.imshow(np.reshape(rate_map, [48,48]), cmap = 'jet')
            
            ps = ax.plot(ms_x, ms_y, ms_t, '|', color = 'red', markeredgewidth = 0, markersize = 5)
            plt.savefig(os.path.join(save_loc, str(i+1)+'.png'), dpi = 600)
            plt.savefig(os.path.join(save_loc, str(i+1)+'.svg'), dpi = 600)
            for p in ps:
                p.remove()
    else:
        #rate_map = trace['smooth_map_all'][neuron_id-1, :]
        spike_idx = calc_spike_time(Spikes[neuron_id-1], behav_time=behav_time, ms_time=ms_time_behav)
        ms_x, ms_y = x[spike_idx], y[spike_idx]
        ms_t = behav_time[spike_idx]/1000
        
        #ax.imshow(np.reshape(rate_map, [48,48]), cmap = 'jet')
        ps = ax.plot(ms_x, ms_y, ms_t, '|', color = 'red', markeredgewidth = 1, markersize = 3)
        if save_loc is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_loc, str(neuron_id-1)+'.png'), dpi = 600)
            plt.savefig(os.path.join(save_loc, str(neuron_id-1)+'.svg'), dpi = 600)            
    

if __name__ == '__main__':
    import pickle
    with open(r"E:\Data\Cross_maze\11095\20220828\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        
    TraceMap3D(trace, neuron_id = 3, save_loc = None)
        
    


