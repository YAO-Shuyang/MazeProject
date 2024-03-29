import numpy as np
from mylib.local_path import DMatrixPath
from mylib.maze_graph import maze_graphs
import sklearn.preprocessing
import os
import pickle

def gaussian(x=0, sigma=2, pi=3.1416, a=1):
    x = x*a
    return 1 / (sigma * np.sqrt(pi * 2)) * np.exp(- x * x / (sigma * sigma * 2))

def gaussian_smooth_matrix2d(maze_type: int, sigma: float = 3, _range: int = 7, nx: int = 48):
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
    
    smooth_matrix = np.zeros((nx*nx,nx*nx), dtype = np.float64)
    
    Path = os.path.join(DMatrixPath, f"D{nx}_{maze_type}.pkl")
    with open(Path, 'rb') as handle:
        D = pickle.load(handle)
        
    scale = nx/12
    
    for curr in range(1,nx*nx+1):
        SurrMap = {}
        SurrMap[0]=[curr]
        Area = [curr]
    
        step = int(_range * 1.5)
        smooth_matrix[curr-1,curr-1] = gaussian(0,sigma = sigma, a = 48/nx)
        for k in range(1,step+1):
            SurrMap[k] = np.array([],dtype = np.int32)
            for s in SurrMap[k-1]:
                for j in range(len(graph[s])):
                    length = D[curr-1, graph[s][j]-1]
                    if graph[s][j] not in Area and length <= _range:
                        Area.append(graph[s][j])
                        SurrMap[k] = np.append(SurrMap[k], graph[s][j])
                        smooth_matrix[curr-1, graph[s][j]-1] = gaussian(length, sigma = sigma, a = 24/nx)

    smooth_matrix = sklearn.preprocessing.normalize(smooth_matrix, norm = 'l1')
    return smooth_matrix

def gaussian_smooth_matrix1d(shape: int, window: int = 10, sigma=3, folder=1, dis_stamp = None):
    M = np.zeros((shape, shape), dtype=np.float64)

    if window % 2 == 1:
        a, b = int((window-1)/2), int((window-1)/2)
    else:
        a, b = int(window/2)-1, int(window/2)

    for i in range(shape):
        lef = i-a if i-a >= 0 else 0
        rig = i+b+1 if i+b+1 <= shape else shape

        if dis_stamp is None:
            weight = gaussian(np.arange(lef, rig) - i, sigma=sigma, a=folder)
        else:
            assert dis_stamp.shape[0] == shape
            weight = gaussian(dis_stamp[lef:rig] - dis_stamp[i], sigma=sigma, a=folder)
    
        weight = weight / np.nansum(weight)
        M[i, lef:rig] = weight
    return M