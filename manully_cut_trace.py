import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from mylib.maze_utils3 import Clear_Axes, DrawMazeProfile, mkdir

def plot(trace: dict, save_loc: str) -> None:
    pos = trace['processed_pos_new']
    
    n = pos.shape[0]
    fig = plt.figure(figsize=(3,3))
    ax = Clear_Axes(plt.axes())
    ax.set_aspect('equal')
    ax = DrawMazeProfile(axes=ax, maze_type=trace['maze_type'], nx = 48, linewidth=1,color = 'black')

    for k in tqdm(range(n-10)):
        if k >= 20:
            pre_ps = pos[k-20:k, :]/20 - np.array([0.5, 0.5])
            points = pos[k:k+11, :]/20 - np.array([0.5, 0.5])
        else:
            pre_ps = pos[0:k, :]/20 - np.array([0.5, 0.5])
            points = pos[k:k+11, :]/20 - np.array([0.5, 0.5])

        a = ax.plot(pre_ps[:, 0], pre_ps[:, 1], 'o', color = 'gray', markeredgewidth = 0, markersize = 3)
        b = ax.plot(points[:, 0], points[:, 1], 'o', color = 'red', markeredgewidth = 0, markersize = 3)
        ax.set_title(str(k)+" -> "+str(k+10), fontsize = 8)
        plt.savefig(os.path.join(save_loc, str(k)+'.png'), dpi = 150)
        for e in a+b:
            e.remove()
    
if __name__ == '__main__':
    import pickle

    with open(r"G:\YSY\Dsp_maze\10212\20230602\session 1\trace_behav.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    p = r"G:\YSY\Dsp_maze\10212\20230602\session 1\rectify_traj"
    mkdir(p)
    plot(trace, p)
