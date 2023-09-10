import numpy as np
from mylib.divide_laps.lap_split import LapSplit
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import scipy.io
import os

MATLAB_STRUCT = dict

def get_neural_trajectory(trace: dict, save_loc: str, thre: float = 1) -> MATLAB_STRUCT:
    try:
        assert 'correct_time' in trace.keys()
        assert 'ms_time' in trace.keys()
        assert 'DeconvSignal' in trace.keys()
    except:
        raise ValueError("Input dict lack a crucial key!")
    
    beg, end = LapSplit(trace, behavior_paradigm=trace['paradigm'])
    laps = beg.shape[0]
    std_thre = np.nanstd(trace['DeconvSignal'])*thre
    
    D = []
    Data = []
    for i in range(laps):
        data = np.where(trace['DeconvSignal'][:, np.where(trace['ms_time'] >= trace['correct_time'][beg[i]])[0][0]:np.where(trace['ms_time'] <= trace['correct_time'][end[i]])[0][-1]+1] >= std_thre, 1, 0).astype(np.int64)
        trial = np.array([i+1])
        D.append((data, trial))
        Data.append(data)
        
    D = np.array([D], dtype=[('data', 'O'), ('trial', 'O')])
    
    scipy.io.savemat(os.path.join(save_loc, 'neural_trajectory.mat'), {'D':D})
    print(os.path.join(save_loc, 'neural_trajectory.mat')," is saved sucessfully!")

    with open(os.path.join(save_loc, 'neural_trajectory.pkl'), 'wb') as f:
        pickle.dump(Data, f)
    
    return Data

# perform PCA dimentionality reduction
from sklearn.decomposition import PCA

def pca_dim_reduction(Data, n_components=3):
    print("Fitting PCA...")
    pca = PCA(n_components=n_components)
    pca.fit(Data[0].T)
    Y = pca.transform(Data[0].T)
    
    # 3D plot
    plt.
    plt.show()
    return pca

if __name__ == '__main__':
    import pickle
    
    with open(r'E:\Data\Cross_maze\11095\20220830\session 3\trace.pkl', 'rb') as handle:
        trace = pickle.load(handle)
        
    data = get_neural_trajectory(trace, r'E:\Data\Cross_maze\11095\20220830\session 3')
    pca_dim_reduction(data)
