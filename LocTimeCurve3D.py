import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
from mylib.maze_graph import *
from mylib.maze_utils3 import DrawMazeProfile, Add_NAN, Clear_Axes, spike_nodes_transform, FastDistance, mkdir
from mylib.dp_analysis import BehaviorEvents
from mylib.TraceMap3D import calc_spike_time
from tqdm import tqdm

class Trajactory2D(BehaviorEvents):
    def __init__(self, maze_type: int, behav_nodes: np.ndarray, behav_time: np.ndarray) -> None:
        super().__init__(maze_type, behav_nodes, behav_time)
        self.get_abbr_nodes2d()
        self.get_loc()

    def get_abbr_nodes2d(self):
        W2D = Wrong2DecisionPointGraph1 if self._maze == 1 else Wrong2DecisionPointGraph2
        abbr_node2d = np.zeros((self.abbr_idx.shape[0], 2), dtype=np.int64)
        idx = self.abbr_idx
        for i in range(idx.shape[0]):
            if self.abbr_node[i] in self._cp:
                abbr_node2d[i, 0] = np.where(self._cp == self.abbr_node[i])[0]
                
            else:
                dp = W2D[int(self.abbr_node[i])]
                abbr_node2d[i, 0] = np.where(self._cp == dp)[0]
                abbr_node2d[i, 1] = np.where(np.array(self._wg[dp]) == self.abbr_node[i])[0][0]
                
        self.abbr_node2d = abbr_node2d
        
    def get_loc(self):
        loc = np.zeros((self.node.shape[0], 2), dtype=np.float64)
        idx = self.abbr_idx
        x, y = self.abbr_node2d[:, 0], self.abbr_node2d[:, 1]
        for i in range(idx.shape[0]-1):
            
            n1, n2 = self.abbr_node[i], self.abbr_node[i+1]
            dt = idx[i+1] - idx[i]
            """
            if int(n1) not in self._mg[int(n2)]:
                if n1 == 144:
                    loc[idx[i]:idx[i+1], 0] = np.linspace(x[i], x[i]+1, dt)
                else:
                    loc[idx[i]:idx[i+1], 0] = np.nan
                    loc[idx[i]:idx[i+1], 1] = np.nan
            else:
                  
            """
            loc[idx[i]:idx[i+1], 0] = np.linspace(x[i], x[i+1], dt)
            loc[idx[i]:idx[i+1], 1] = np.linspace(y[i], y[i+1], dt)
            
            if FastDistance(n1, n2, self._maze, nx = 12) >= 5:
                loc[idx[i]:idx[i+1], 0] = np.nan
                loc[idx[i]:idx[i+1], 1] = np.nan  
                
        loc[idx[-1]::, 0] = np.linspace(x[-1], x[-1]+1, self.node.shape[0] - idx[-1])
        
        self.x, self.y = loc[:, 0], loc[:, 1]


def LocTimeCurve3D(trace, start_id = 1, save_loc = None, markeredgewidth = 1, markersize = 4):
    Spikes = trace['Spikes']
    ms_time_behav = trace['ms_time_behav']
    behav_time, behav_nodes = trace['behav_time'], trace['behav_nodes']
    
    pos3d = Trajactory2D(trace['maze_type'], 
                         behav_nodes=spike_nodes_transform(behav_nodes, 12), behav_time=behav_time)

    idx = np.where(trace['is_placecell'] == 1)[0]
    for i in idx:
        if i+1 < start_id:
            continue
        fig = plt.figure(facecolor=None, edgecolor=None, figsize=(2,10))
        ax: Axes3D = fig.add_subplot(projection='3d', alpha = 0, box_aspect = [10,2,20])                
        x, y, t = pos3d.x, pos3d.y, pos3d.time.astype(np.float64)
        t[np.where(np.isnan(t))[0]] = np.nan
        ax.set_zlabel("time / s")
        ax.set_xlabel("Correct path")
        ax.set_ylabel("Incorrect path")
        ax.plot(x, y, t/1000, c = 'black', linewidth = 1)
        
        spike_idx = calc_spike_time(Spikes[i], behav_time=behav_time, ms_time=ms_time_behav)
        ms_x, ms_y = x[spike_idx], y[spike_idx]
        ms_t = behav_time[spike_idx]/1000
        
        #ax.imshow(np.reshape(rate_map, [48,48]), cmap = 'jet')
        ps = ax.plot(ms_x, ms_y, ms_t, '|', color = 'red', markeredgewidth = markeredgewidth, markersize = markersize)
        color = 'red' if trace['is_placecell'][i] == 1 else 'black'
        ax.set_title(f'Cell {i+1}', color = color)
        plt.savefig(os.path.join(save_loc, str(i+1)+'.png'), dpi = 600)
        plt.savefig(os.path.join(save_loc, str(i+1)+'.svg'), dpi = 600)
        plt.show()

                    
if __name__ == '__main__':
    import pickle
    with open(r"E:\Data\Cross_maze\11095\20220828\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    loc = os.path.join(r"E:\Data\Cross_maze\11095\20220828\session 2\LocTimeCurve\3D")
    mkdir(loc)
    LocTimeCurve3D(trace, save_loc=loc)
    