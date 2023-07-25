from mylib.maze_utils3 import *

def check(trace = None, trace_old = r'F:\YSY\Cross_maze Backup\11095\20220824\session 2\trace.pkl'):
    with open(trace_old, 'rb') as handle:
        trace_old = pickle.load(handle)

    n_neuron = trace['n_neuron']
    old_map_new = trace['old_map_clear']
    old_map_old = trace_old['old_map_clear']
    smooth_map_new = trace['smooth_map_all']
    smooth_map_old = trace_old['smooth_map_all']

    # old map for newly calculated, old-version and smoothed rate map for newly calculated and old version
    for n in range(n_neuron):
        fig, ax = plt.subplots(ncols = 4, nrows = 1, figsize = (16,3))
        for i in range(4):
            nx = 12 if i in [0,1] else 48
            ax[i] = Clear_Axes(ax[i])
            if trace['maze_type'] in [1,2]:
                DrawMazeProfile(axes = ax[i], nx = nx, linewidth = 2, color = 'white', maze_type = trace['maze_type'])

        im = ax[0].imshow(np.reshape(old_map_new[n],[12,12]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax[0])
        ax[0].set_title("Cell "+str(n+1)+"\nold_map newly calculated.")
        
        im = ax[1].imshow(np.reshape(old_map_old[n],[12,12]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax[1])
        ax[1].set_title("Cell "+str(n+1)+"\nold_map old version")      

        im = ax[2].imshow(np.reshape(smooth_map_new[n],[48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax[2])
        ax[2].set_title("Cell "+str(n+1)+"\nsmooth_map newly calculated.")
        
        im = ax[3].imshow(np.reshape(smooth_map_old[n],[48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = ax[3])
        ax[3].set_title("Cell "+str(n+1)+"\nsmooth_map old version")
        plt.show()