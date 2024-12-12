from scipy.stats import kendalltau
from mylib.statistic_test import *
from mazepy.datastruc.neuact import SpikeTrain, NeuralTrajectory

code_id = "TEST0001 - Kendall Correlation"
loc = os.path.join(figpath, code_id)
mkdir(loc)

with open(r"E:\Data\Cross_maze\10227\20230930\session 2\trace.pkl", "rb") as f:
    trace = pickle.load(f)
    
with open(r"E:\Data\Cross_maze\10227\20230930\session 3\trace.pkl", "rb") as f:
    trace2 = pickle.load(f)
    
with h5py.File(r"E:\Data\Cross_maze\10227\20230930\cross_session\AlignedResults\cellRegistered.mat", 'r') as handle:
    index_map = np.asarray(handle['cell_registered_struct']["cell_to_index_map"])
    idx = np.where((index_map[1, :] > 0) & (index_map[2, :] > 0))[0]
    cell1, cell2 = index_map[1, idx].astype(np.int64)-1, index_map[2, idx].astype(np.int64)-1

y = trace['Kendall_Tau'][np.triu_indices(trace['Kendall_Tau'].shape[0], 1)]
y = y[np.isnan(y) == False]
fig = plt.figure(figsize = (4,3))
ax: Axes = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
ax.hist(y, bins = 80, range=(-0.3, 0.5), density = True, color = '#6FB6B6', edgecolor = 'k', linewidth=0.5)
ax.set_xlim(-0.3, 0.5)
ax.set_xticks(np.linspace(-0.3, 0.5, 9))
plt.savefig(join(loc, 'hist.png'), dpi = 600)
plt.savefig(join(loc, 'hist.svg'), dpi = 600)
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def tau_matrix(trace, thre: float = 50):
    mat = np.zeros((trace['n_neuron'], trace['n_neuron']))
    mask = np.where(trace['ROI_Distance'] >= thre, 1, np.nan)
    
    is_qualified = np.zeros(trace['n_neuron'])
    for i in range(trace['n_neuron']):
        if np.where(trace['neural_traj'][i, :] == 0)[0].shape[0]  < 0.98*trace['neural_traj'].shape[1]:
            is_qualified[i] = 1
        else:
            mask[i, :] = np.nan
            mask[:, i] = np.nan
    
    for i in tqdm(range(trace['n_neuron']-1)):
        for j in range(i+1, trace['n_neuron']):
            if np.isnan(mask[i, j]) or is_qualified[i] == 0 or is_qualified[j] == 0:
                continue
            
            mat[i, j] = kendalltau(trace['neural_traj'][i, :], trace['neural_traj'][j, :])[0]
            mat[j, i] = mat[i, j]
    
    mat = mat * mask
    return mat

def pti_tau_matrix(trace, thre: float = 50):
    # Find Levy et al., 2023
    # They proposed position-tuning independent (PTI) rate
    
    mat = np.zeros((trace['n_neuron'], trace['n_neuron']))
    mask = np.where(trace['ROI_Distance'] >= thre, 1, np.nan)
    
    pti = trace['neural_traj']-trace['smooth_map_all'][:, trace['pos_traj']]

    is_qualified = np.zeros(trace['n_neuron'])
    for i in range(trace['n_neuron']):
        if np.where(trace['neural_traj'][i, :] == 0)[0].shape[0] < 0.98*trace['neural_traj'].shape[1]:
            is_qualified[i] = 1
        else:
            mask[i, :] = np.nan
            mask[:, i] = np.nan
    
    for i in tqdm(range(trace['n_neuron']-1)):
        for j in range(i+1, trace['n_neuron']):
            if np.isnan(mask[i, j]) or is_qualified[i] == 0 or is_qualified[j] == 0:
                continue
            
            mat[i, j] = kendalltau(pti[i, :], pti[j, :])[0]
            mat[j, i] = mat[i, j]
    
    mat = mat * mask
    return mat

"""
for i in range(len(f1)):
    if f1['maze_type'][i] == 0 or f1['include'][i] == 0:
        continue
    
    if f1['MiceID'][i] not in [10209, 10212, 10224, 10227]:
        continue
    
    print(f1['MiceID'][i], f1['date'][i], f1['maze_type'][i])
    
    with open(f1['Trace File'][i], 'rb') as handle:
        trace = pickle.load(handle)
        
    trace['Kendall_Tau'] = tau_matrix(trace)
    trace['Kendall_Tau_PTI'] = pti_tau_matrix(trace)
    
    with open(f1['Trace File'][i], 'wb') as handle:
        pickle.dump(trace, handle)
"""