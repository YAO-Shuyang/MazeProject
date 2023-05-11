import numpy as np
import os
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
from mylib.maze_graph import *
from tqdm import tqdm
import scipy.stats
import sklearn.preprocessing
from scipy.stats import pearsonr
import pickle
import cv2
import h5py
import copy as cp
import time
import shutil
import seaborn as sns
import pandas as pd
import matplotlib
from mylib.AssertError import KeyWordErrorCheck, VariablesInputErrorCheck, ReportErrorLoc, ValueErrorCheck
import warnings

figpath = 'F:\YSY\FinalResults'
figdata = 'F:\YSY\FigData'

def location_to_idx(loc_x, loc_y, nx = 12, MaxLength = 960):
    length = MaxLength / nx
    cell_x = loc_x // length
    cell_y = loc_y // length
    return loc_to_idx(cell_x = cell_x, cell_y = cell_y, nx = nx)

# transform 960cm position data into nx*nx form
def position_transform(processed_pos = None, nx = 48, maxWidth = 960):
    return processed_pos / maxWidth * nx - np.array([0.5, 0.5], np.float64)

def loc_to_idx(cell_x, cell_y, nx = 12):
    idx = cell_x + cell_y * nx + 1
    return idx

def idx_to_loc(idx, nx = 12, ny = 12):
    # convert index to location (x,y), note: location start from 0
    cell_w = (idx-1) % nx
    cell_h = (idx-1) // nx
    return cell_w, cell_h

def isNorthBorder(mazeID, nx = 12):
    if (mazeID-1) // nx == nx - 1:
        return 1
    else:
        return 0

def isEastBorder(mazeID, nx = 12):
    if mazeID % nx == 0:
        return 1
    else:
        return 0

def isWestBorder(mazeID, nx = 12):
    if mazeID % nx == 1:
        return 1
    else:
        return 0

def isSouthBorder(mazeID, nx = 12):
    if mazeID <= nx:
        return 1
    else:
        return 0

def WallMatrix(maze_type = 1):
    # Wall matrix will return 2 matrix: w_m represents verticle walls which has a shape of 12*13; h_m represents horizontal walls 
    # with a shape of 13*12

    vertical_walls = np.ones((12,13), dtype = np.int64)
    horizont_walls = np.ones((13,12), dtype = np.int64)
    nx = 12
    if maze_type == 0:
        graph = OpenField_graph
    elif maze_type == 1:
        graph = maze1_graph
    elif maze_type == 2:
        graph = maze2_graph
    else:
        assert False

    for i in range(1,145):
        if i == 1 and maze_type != 0:
            vertical_walls[0,0] = 0
        if i == 144 and maze_type != 0:
            vertical_walls[11,12] = 0
        
        x, y = idx_to_loc(i, nx = nx, ny = nx)

        surr = graph[i]
        for s in surr:
            if s == i + 1:
                vertical_walls[y, x+1] = 0
            elif s == i - 1:
                vertical_walls[y, x] = 0
            elif s == i + nx:
                horizont_walls[y+1, x] = 0
            elif s == i - nx:
                horizont_walls[y, x] = 0
            else:
                assert False
        
    return vertical_walls, horizont_walls

def DrawMazeProfile(maze_type = 1,axes = None, color = 'white',linewidth = 1, nx = 48, v = [], h = []):
    if len(v) == 0:
        v,h = WallMatrix(maze_type = maze_type)
    else:
        ReportErrorLoc(__file__+'.DrawMazeProfile')
    l = nx / 12

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i,j] == 1:
                axes.plot([j*l-0.5, j*l-0.5],[i*l-0.5,(i+1)*l-0.5], color = color, linewidth = linewidth)

    for j in range(h.shape[0]):
        for i in range(h.shape[1]):
            if h[j,i] == 1:
                axes.plot([i*l-0.5, (i+1)*l-0.5],[j*l-0.5,j*l-0.5], color = color, linewidth = linewidth)
    return axes

# make up dir
def mkdir(path:str):
    '''
    Note
    ----
    Input a directory that you want to make up.
    '''
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print("        "+path + ' is made up successfully!')
        return True
    else:
        print("        "+path + ' is already existed!')
        return False

# calculate path between two points
def DFS(start = 1, goal = 1, maze_type = 1, nx = 48):
    if nx == 48:
        if maze_type == 1:
            graph = maze1_SonGraph 
        elif maze_type == 2:
            graph = maze2_SonGraph
        elif maze_type == 0:
            graph = OpenField_graph1
    elif nx == 12:
        if maze_type == 1:
            graph = maze1_graph 
        elif maze_type == 2:
            graph = maze2_graph
        elif maze_type == 0:
            graph = OpenField_graph
    elif nx == 24:
        if maze_type == 1:
            graph = maze1_QuarterGraph 
        elif maze_type == 2:
            graph = maze2_QuarterGraph
        elif maze_type == 0:
            graph = OpenField_graph4
    else:
        assert False

    if maze_type == 0:
        graph = OpenField_graph1

    Area = [start]
    StepExpand = {0: [start]}
    IndexList = {start: 0}  # {postN:[prev1,prev2,...]}
    prev = 0
    while goal not in Area:
        StepExpand[prev+1] = []
        for j in StepExpand[prev]:
            for k in graph[j]:
                if k not in Area:
                    Area.append(k)
                    StepExpand[prev+1].append(k)
                    IndexList[k] = j
        prev += 1
    
    path = [goal]
    prev = goal
    while start not in path:
        path.append(IndexList[prev])
        prev = path[-1]
    
    path.reverse()
    return path

def FastDistance(start, goal, maze_type = 1, nx = 12):
    if nx != 12:
        print("Warning! This funciton is specifically for old maze (12*12)")
        return
    
    correct_path = CorrectPath_maze_1 if maze_type == 1 else CorrectPath_maze_2
    incorrect_path = IncorrectPath_maze_1 if maze_type == 1 else IncorrectPath_maze_2
    W2DPGraph = Wrong2DecisionPointGraph1 if maze_type == 1 else Wrong2DecisionPointGraph2
    DP2WGraph = DecisionPoint2WrongGraph1 if maze_type == 1 else DecisionPoint2WrongGraph2
    
    if start in incorrect_path:
        if maze_type == 2 and start in [51,50,38,62,39,27,26]:
            vec = [1,2,3,3,2,3,4] # distance from nearest decision point 52
            wrong_start = vec[np.where(np.array([51,50,38,62,39,27,26]) == start)[0][0]]
            dp_start = 52
        else:
            dp_start = W2DPGraph[start]
            wrong_start = np.where(np.array(DP2WGraph[dp_start]) == start)[0][0] + 1
    elif start not in incorrect_path:
        dp_start = start
        wrong_start = 0
    
    if goal in incorrect_path:
        if maze_type == 2 and goal in [51,50,38,62,39,27,26]:
            vec = [1,2,3,3,2,3,4] # distance from nearest decision point 52
            wrong_goal = vec[np.where(np.array([51,50,38,62,39,27,26]) == goal)[0][0]]
            dp_goal = 52
        else:
            dp_goal = W2DPGraph[goal]
            wrong_goal = np.where(np.array(DP2WGraph[dp_goal]) == goal)[0][0] + 1
    elif goal not in incorrect_path:
        dp_goal = goal
        wrong_goal = 0    
        
    idx_start = np.where(np.array(correct_path) == dp_start)[0][0]
    idx_goal = np.where(np.array(correct_path) == dp_goal)[0][0]
    
    return wrong_start + wrong_goal + np.abs(idx_start - idx_goal)



# Smooth
# Define a Gaussian-based smooth method -------------------------------------------------------------------------------------------
def Gaussian(x=0, sigma=2, pi=3.1416, nx = 48):
    x = x * (48 / nx)
    return 1 / (sigma * np.sqrt(pi * 2)) * np.exp(- x * x / (sigma * sigma * 2))

def Cartesian_distance(curr, surr, nx = 48):
    # curr is the node id
    curr_x, curr_y = idx_to_loc(curr, nx = nx, ny = nx)
    surr_x, surr_y = idx_to_loc(surr, nx = nx, ny = nx)
    return np.sqrt((curr_x - surr_x)*(curr_x - surr_x)+(curr_y - surr_y)*(curr_y - surr_y))

def SmoothMatrix(maze_type = 1, sigma = 2, _range = 7, nx = 48):
    if nx == 48:
        if maze_type == 1:
            graph = maze1_SonGraph 
        elif maze_type == 2:
            graph = maze2_SonGraph
        elif maze_type == 0:
            graph = OpenField_graph1
    elif nx == 12:
        if maze_type == 1:
            graph = maze1_graph 
        elif maze_type == 2:
            graph = maze2_graph
        elif maze_type == 0:
            graph = OpenField_graph
    elif nx == 24:
        if maze_type == 1:
            graph = maze1_QuarterGraph 
        elif maze_type == 2:
            graph = maze2_QuarterGraph
        elif maze_type == 0:
            graph = OpenField_graph4
    else:
        assert False
    
    smooth_matrix = np.zeros((nx*nx,nx*nx), dtype = np.float64)
    
    for curr in range(1,nx*nx+1):
        SurrMap = {}
        SurrMap[0]=[curr]
        Area = [curr]
    
        step = int(_range * 1.5)
        smooth_matrix[curr-1,curr-1] = Gaussian(0,sigma = sigma, nx = nx)
        for k in range(1,step+1):
            SurrMap[k] = np.array([],dtype = np.int32)
            for s in SurrMap[k-1]:
                for j in range(len(graph[s])):
                    length = Cartesian_distance(curr, graph[s][j], nx = nx)
                    if graph[s][j] not in Area and length <= _range:
                        Area.append(graph[s][j])
                        SurrMap[k] = np.append(SurrMap[k], graph[s][j])
                        smooth_matrix[curr-1, graph[s][j]-1] = Gaussian(length,sigma = sigma, nx = nx)

    smooth_matrix = sklearn.preprocessing.normalize(smooth_matrix, norm = 'l1')
    return smooth_matrix
    
def smooth(clear_map_all, maze_type = 1, nx = 48, _range = 7, sigma = 2):
    if maze_type in [0,1,2]:
        print("    Generate smooth matrix")
        Ms = SmoothMatrix(maze_type = maze_type, sigma = sigma, _range = _range, nx = nx)
    else:
        print("SmoothError: input variable maze_type error!")
        return clear_map_all

    print("    Begin smooth:")
    smooth_map_all = np.dot(clear_map_all, Ms.T)
    return smooth_map_all

# --------------------------------------------------------------------------------------------------------------------------------
def clear_NAN(rate_map_all:np.ndarray):
    '''
    Note: to clear the NAN value in an numpy.array
    '''
    nanPos = np.where(np.isnan(rate_map_all))
    clear_map_all = cp.deepcopy(rate_map_all)
    clear_map_all[nanPos] = 0

    return clear_map_all, nanPos
    
# generate all subfield. ============================================================================
# place field analysis, return a dict contatins all field. If you want to know the field number of a certain cell, you only need to get it by use 
# len(trace['place_field_all'][n].keys())
def GeneratePlaceField(maze_type = 1, nx = 48, smooth_map = None):
    # rate_map should be one without NAN value. Use function clear_NAN(rate_map_all) to process first.
    MAX = max(smooth_map)
    field_set = np.where(smooth_map >= 0.5*MAX)[0]+1
    search_set = []
    All_field = {}

    while len(np.setdiff1d(field_set, search_set))!=0:
        diff = np.setdiff1d(field_set,search_set)
        point = diff[0]
        subfield = field(rate_map = smooth_map, point = point, maze_type = maze_type, nx = nx, MAX = MAX)
        peak_loc = subfield[0]
        peak = smooth_map[peak_loc-1]
        # find peak idx as keys of place_field_all dict objects.
        for k in subfield:
            if smooth_map[k-1] > peak:
                peak = smooth_map[k-1]
                peak_loc = k
        All_field[peak_loc] = subfield
        search_set = sum([search_set, subfield],[])
    
    return All_field
               
def field(rate_map = None,point = 1, maze_type = 1, nx = 48,MAX = 0):
    if nx == 12:
        if maze_type == 0:
            graph = OpenField_graph1
        elif maze_type == 1:
            graph = maze1_graph
        elif maze_type == 2:
            graph = maze2_graph
    elif nx == 48:
        if maze_type == 0:
            graph = OpenField_graph1
        elif maze_type == 1:
            graph = maze1_SonGraph
        elif maze_type == 2:
            graph = maze2_SonGraph
    else:
        assert False
            
    MaxStep = 300
    step = 0
    Area = [point]
    StepExpand = {0: [point]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:
                if rate_map[j-1] >= 0.5*MAX and j not in Area:
                    StepExpand[step+1].append(j)
                    Area.append(j)
        
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
            
        step += 1
    return Area

# get all cell's place field
def place_field(n_neuron = None, smooth_map_all = None, maze_type = 1):
    place_field_all = []
    smooth_map_all = cp.deepcopy(smooth_map_all)
    for k in tqdm(range(n_neuron)):
        place_field = GeneratePlaceField(smooth_map = smooth_map_all[k], maze_type = maze_type, nx = 48)
        place_field_all.append(place_field)
    print("    Place field has been generated successfully.")
    return place_field_all

#  ========================================================================================================================

def Norm_ratemap(rate_map_all):
    rate_map_all_norm = sklearn.preprocessing.minmax_scale(rate_map_all, feature_range=(0, 1), axis=1, copy=True)
    return rate_map_all_norm

def DateTime(is_print:bool = False):
    '''
    Author: YAO Shuyang
    Date: Jan 26, 2023
    Note: to return current time with certain format: e.g. 2022-09-16 13:50:42

    Parameter
    ---------
    is_print: bool, default (False)
        whether print the time.

    Return
    ------
    str
        the current time with certain form.
    '''
    if is_print:
        t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(t1)
        return t1
    else:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def Clear_Axes(axes = None,close_spines:list = ['top','bottom','left','right'],
               xticks:list|np.ndarray = [], yticks:list|np.ndarray = [], ifxticks:bool = False, ifyticks:bool = False):
    '''
    Author: YAO Shuyang
    Date: Jan 25th, 2022 (Modified) 21:44 UTC +8
    Note: This funciton is to clear the axes and edges of the figure (set them invisible).

    Parameter
    ---------
    - axes: <class 'matplotlib.axes._subplots.AxesSubplot'>, input the canvas axes object whose edges and axes should be cleared.
    - close_spines: list, contains str objects and only 'top', 'bottom','left' and 'right' are valid values. The input edge(s) will be cleared, for example, if input ['top','right'], Only the top and right edges will be cleared and the bottom and left edges will be maintained.
    - xticks: list or numpy.ndarray object. The default value is an empty list and it means no tick of x axis will be shown on the figure. 
        - If you input a list, the number in this list will be shown as x ticks on the figure.
    - yticks: list or numpy.ndarray object. Similar as xticks. The default value is an empty list and it means no tick of y axis will be shown on the figure. 
        - If you input a list, the number in this list will be shown as y ticks on the figure.
    - ifxticks: bool and The default value is False. 
        - If it is False, the xticks will not shown unless the parameter xticks is not an empty list. 
        - If it is ture, the xticks will shown automatically. So if you want to manually set the value of x ticks, you remain this parameter as False and input the xticks list or np.ndarray object that you want to set as xticks through parameter 'xticks'.
    - ifyticks: bool and The default value is False. 
        - If it is False, the yticks will not shown unless the parameter yticks is not an empty list. 
        - If it is ture, the yticks will shown automatically. So if you want to manually set the value of y ticks, you remain this parameter as False and input the yticks list or np.ndarray object that you want to set as yticks through parameter 'yticks'.

    Returns
    -------
    - axes: <class 'matplotlib.axes._subplots.AxesSubplot'>. Return the axes object after clearing.
    '''
    for s in close_spines:
        axes.spines[s].set_visible(False)
    
    if ifxticks == False:
        axes.set_xticks(xticks)
    if ifyticks == False:
        axes.set_yticks(yticks)
    
    return axes


def images_to_video(path, length = 1, name = 'Frame'):
    img_array = []
    
    print("    Read in PNGs.")
    for k in tqdm(range(1,length+1)):
        file_name = os.path.join(path,'Frame '+str(k)+'.png')
        if os.path.exists(file_name) == False:
            print("    "+file_name + " is error!")
        else:
            img = cv2.imread(file_name)
            img_array.append(img)
 
    size = (1800,1800)
    fps = 30
    out = cv2.VideoWriter(os.path.join(path,'Prediction.avi'), cv2.VideoWriter_fourcc('X','V','I','D'), fps, size,True)
    # ('P', 'I', 'M', 'I'),('I', '4', '2', '0')
    
    print("    Write video prediction.avi")
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()

def LocDecodingMovie(y_pred, y_test, maze_type = 1, loc = 'G:\YSY\Cross_maze\11095\20220817\session 2\decoding\decodes_frames'):
    mkdir(loc)
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(axes = ax, maze_type = maze_type, color = 'black', linewidth = 2)
    x_pred, y_pred = idx_to_loc(y_pred, nx = 48, ny = 48)
    x_pred += np.random.rand(y_pred.shape[0]) - 0.5
    y_pred += np.random.rand(y_pred.shape[0]) - 0.5
    x_test, y_test = idx_to_loc(y_test, nx = 48, ny = 48)
    x_test += np.random.rand(y_pred.shape[0]) - 0.5
    y_test += np.random.rand(y_pred.shape[0]) - 0.5
    
    for i in tqdm(range(y_pred.shape[0])):
        a = ax.plot(x_pred[i],y_pred[i], 'o', color = 'cornflowerblue',label = 'Predicted location')
        b = ax.plot(x_test[i],y_test[i], 'o', color = 'red',label = 'Actual location')
        ax.legend(bbox_to_anchor = (0.08,0.98), edgecolor = 'white', facecolor = 'white', ncol = 2)
        plt.savefig(os.path.join(loc,'Frame '+str(i+1)+'.svg'), dpi = 300)
        for j in a:
            j.remove()
        for j in b:
            j.remove()
    
    images_to_video(loc, length = y_pred.shape[0])
    print(os.path.join(loc,'Prediction.avi'),'is successfully generated!')


# Structure similarity between submazes
def rotate(m):
    shapes = m.shape
    n = np.zeros((shapes[1],shapes[0]),dtype = np.int64)
    for i in range(shapes[0]):
        n[:,i] = m[i,:]
    return n

def cover_index(v1,h1,v2,h2):
    walls = (np.sum(v1)+np.sum(v2)+np.sum(h1)+np.sum(h2))/2
    cover_walls = np.sum((v1&v2)) + np.sum((h1&h2))

    ratios = np.zeros(4, np.float64)
    ratios[0] = cover_walls / walls
    for i in range(3):
        c = h1
        h1 = rotate(v1)
        v1 = rotate(h1)
        ratios[i+1] = (np.sum((v1&v2)) + np.sum((h1&h2))) / walls
    return np.nanmax(ratios)

def calc_CIs(maze_type = 1):
    vs, hs = SubMazes(maze_type = maze_type)

    CIs = np.ones((100,100), dtype = np.float64)

    for i in range(99):
        for j in range(i+1, 100):
            CIs[i,j] = cover_index(vs[i],hs[i],vs[j],hs[j])
            CIs[j,i] = cover_index(vs[i],hs[i],vs[j],hs[j])
    return CIs

def SubMazes(maze_type = 1):
    
    v, h = WallMatrix(maze_type = maze_type)
    vs = np.zeros((100,3,4), dtype = np.int64)
    hs = np.zeros((100,4,3), dtype = np.int64)
    for i in range(10):
        for j in range(10):
            k = j + i * 10
            vs[k] = v[i:i+3,j:j+4]
            hs[k] = h[i:i+4,j:j+3]

    return vs, hs

def Draw_SubMazes(maze_type = 1):
    totalpath = 'G:\YSY\Cross_maze'
    loc = os.path.join(totalpath,'Analysis_Results')
    mkdir(loc)
    
    vs, hs = SubMazes(maze_type = maze_type)
    
    fig, axes = plt.subplots(9,12, figsize = (48,36))
    for i in tqdm(range(9)):
        for j in range(12):
            ax = Clear_Axes(axes = axes[i,j])
            k = i * 12 + j
            if k > 99:
                continue
            DrawMazeProfile(axes = ax, color = 'black', maze_type = maze_type, v = vs[k], h = hs[k], linewidth = 3)
    plt.savefig(os.path.join(loc, 'SubMaze'+str(maze_type)+'.svg'),dpi = 600)

def SelfCorrelation(smooth_map):
    smooth_map_2d = np.reshape(smooth_map,[48,48])
    submaps = np.zeros((100,12,12), dtype = np.float64)

    for i in range(10):
        for j in range(10):
            k = i * 10 + j
            submaps[k] = smooth_map_2d[i*4:(i+3)*4,j*4:(j+3)*4]
    
    # Self-Correlation Map
    SCM = np.ones((100,100), dtype = np.float64)
    for i in range(99):
        for j in range(i+1,100):
            r, _ = pearsonr(np.reshape(submaps[i],[144,]),np.reshape(submaps[j],[144,]))
            SCM[i,j] = r
            SCM[j,i] = r
    
    return submaps, SCM

def SpikeType(Transients = None, threshold = 3):
    
    std = np.std(Transients, axis = 1)
    Spikes = np.zeros_like(Transients, dtype = np.int64)
    for i in range(Spikes.shape[0]):
        Spikes[i,:] = np.where(Transients[i,:] >= threshold * std[i], 1, 0)

    return Spikes

def SpikeNodes(Spikes = None, ms_time = None, behav_time = None, behav_nodes = None):
    if behav_nodes.shape[0] != behav_time.shape[0]:
        print("    ERROR! input behav_nodes and behav_time has different shape! They must be the same")
        return
    
    if Spikes.shape[1] != ms_time.shape[0]:
        print("    ERROR! input Spikes and ms_time has different shape! They must be the same")
        return

    if len(np.where(np.isnan(behav_nodes))[0]) != 0:
        print("    Warning! Input behav_nodes contains NAN value!")
        return

    spike_nodes = np.zeros(Spikes.shape[1], dtype = np.float64)
    for i in tqdm(range(spike_nodes.shape[0])):
        if ms_time[i] < behav_time[0]:
            spike_nodes[i] = np.nan
        elif ms_time[i] >= behav_time[-1]:
            spike_nodes[i] = np.nan
        else:
            idx_left = np.where(behav_time <= ms_time[i])[0][-1]
            idx_right = np.where(behav_time >= ms_time[i])[0][0]
            # if the distance to the nearest behav_time stamp is more than 100 ms, we think it will leed to some inaccuracy.
            idx = idx_left if ms_time[i] - behav_time[idx_left] <= behav_time[idx_right] - ms_time[i] else idx_right
            if np.abs(ms_time[i] - behav_time[idx]) <= 100:
                spike_nodes[i] = behav_nodes[idx]
            else:
                spike_nodes[i] = np.nan
    return spike_nodes

def GetDMatrices(maze_type:int, nx:int):
    '''
    Note
    ----
    To return the distance matrix of given maze and given bin number.

    Parameter
    ---------
    - maze_type:int, only 0,1,2 are valid or it will raise an error.
    - nx:int, only, 12,24,48 are valid or it will raise an error.

    YAO Shuyang, Feb 10, 2023
    '''
    ValueErrorCheck(nx, [12,24,48])
    ValueErrorCheck(maze_type, [1,2,0])
    try:
        with open('decoder_DMatrix.pkl', 'rb') as handle:
            D_Matrice = pickle.load(handle)
    except:
        print('decoder_DMatrix.pkl is not exist!')
        assert False

    if nx == 12:
        D = D_Matrice[6+maze_type] / nx * 12
    elif nx == 24:
        D = D_Matrice[3+maze_type] / nx * 12
        #S2FGraph = Quarter2FatherGraph
    elif nx == 48:
        D = D_Matrice[maze_type] / nx * 12
        #S2FGraph = Son2FatherGraph
    else:
        print("nx value error! Report by Decoding ChanceLevel, maze_utils3.")
        assert False
    return D

def DecodingChanceLevel(maze_type:int = 1, nx:int = 48, shuffle_frames:int = 40000, occu_time:np.ndarray = None, Ms:np.ndarray = np.zeros((2304,2304))):
    ValueErrorCheck(nx, [12,24,48])
    ValueErrorCheck(maze_type, [1,2,0])

    D = GetDMatrices(maze_type = maze_type, nx = nx)

    assert occu_time.shape[0] == nx**2

    # Guess by occupation time only.
    occu_time[np.where(np.isnan(occu_time))[0]] = 0
    # probability vector (generated by occupation time. That is if animal stays at a certain bin for a relative longer time, the probability of guessing this bin is
    # correlated higher.)
    if nx == 48:
        occu_time = np.dot(Ms, occu_time)
    
    total_time = np.sum(occu_time)
    prob_vec = occu_time / total_time

    real_vec = np.random.randint(low = nx**2, size = shuffle_frames)
    # guess_vec = np.random.randint(low = nx**2, size = shuffle_frames) # pure random case
    guess_vec = np.random.choice(a = np.arange(nx**2), size = shuffle_frames, p = prob_vec, replace = True)

    # absolute accuracy is the ratio of correct 
    abHit = len(np.where((real_vec - guess_vec) == 0)[0]) / shuffle_frames

    if nx == 12:
        geHit = cp.deepcopy(abHit)
    else:
        real_vec_old = spike_nodes_transform(real_vec+1, nx = 12)
        guess_vec_old = spike_nodes_transform(guess_vec+1, nx = 12)
        geHit = len(np.where((real_vec_old - guess_vec_old) == 0)[0]) / shuffle_frames

    predict_error = D[real_vec, guess_vec] * 8

    MAE = np.nanmean(predict_error)        
    RMSE = np.sqrt(np.nanmean(predict_error**2))
    
    return RMSE, MAE, abHit, geHit

## PeakCurveRegression =============================================================================================================
def dis_to_line(x = 0, y = 0, k = -1, b = 1):
    # k,b: float, to depict a line
    # x,y: float: the location of a point
    theta = np.arctan(k) + np.pi if np.arctan(k)<0 else np.arctan(k)
    return np.abs(x - (y-b)/k) * np.cos(theta  - np.pi/2)

# Sort rate map with peak firing rate.
def sortmap(rate_map_all, order = None):
    maxidx = np.nanargmax(rate_map_all, axis = 1)
    if order is None:
        order = np.array([], dtype = np.int64)
        for i in range(144,0,-1):
            order = np.concatenate([order,np.where(maxidx == i-1)[0]])
    return cp.deepcopy(rate_map_all[order,:])

def sort_peak_curve(rate_map_all, maze_type = 1, delete_idx = None, is_sort_x = True, is_sort_y = True, is_norm_y = True):
    if maze_type == 0:
        x_order = np.arange(144)
    elif maze_type == 1:
        x_order = np.concatenate([CorrectPath_maze_1-1, IncorrectPath_maze_1-1])
    elif maze_type == 2:
        x_order = np.concatenate([CorrectPath_maze_2-1, IncorrectPath_maze_2-1])
    
    # Clear NAN value
    rate_map_all, _ = clear_NAN(rate_map_all = rate_map_all)
    # Sort axis 1 (x axis)
    if is_sort_x == True:
        rate_map_all = rate_map_all[:,x_order]

    # Delete elements in axis 0 (y axis, neuron id)
    if delete_idx is not None:
        rate_map_all = np.delete(rate_map_all, delete_idx, axis = 0)

    if is_norm_y == True:    
        # Normed with axis 0
        rate_map_all = Norm_ratemap(rate_map_all = rate_map_all)

    if is_sort_y == True:    
        # Sort map with axis 0
        rate_map_all = sortmap(rate_map_all)
    return cp.deepcopy(rate_map_all)

# Draw distribution of peak rate location (number of nodes that peak rates locate at), a (144,) np.ndarray array
def PeakDistributionGraph(old_map_all = None, SilentNeuron = None, node_not_occu = None):
    # old_map_all must contain NAN value. Silent cell must be deleted(input SilentNeuron ID and it can be deleted inside this funciton).
    # Sort and norm old_map_all at y axis(axis 0) and 
    old_map_all = sort_peak_curve(rate_map_all = old_map_all, delete_idx = SilentNeuron, is_sort_x = False)
    
    # Count number
    PeakDistribution = np.where(old_map_all == 1, 1, 0)
    PDGraph = np.nansum(PeakDistribution, axis = 0)
    PDGraph = PDGraph.astype(dtype = np.float64)

    # Set value of unoccupied nodes as np.nan
    PDGraph[node_not_occu] = np.nan
    
    # Divided by total neuron number:
    PDGraph = PDGraph / np.nansum(PDGraph)

    return PDGraph

# Calculate PeakDistributionDensity Error by:
#   ErrorGraph = PDGraph - MeanGraph
#   MAE = mean of absolute value of ErrorGraph(mean of absolute error)
#   RMSE = mean of ErrorGraph^2
def PeakDistributionDensity(old_map_all = None, SilentNeuron = np.array([], dtype = np.int64), node_not_occu = np.array([], dtype = np.int64)):
    # Calculate PDGraph from old_map_all, then subtract PDGraph by MeanGraph
    PDGraph = PeakDistributionGraph(old_map_all = old_map_all, SilentNeuron = SilentNeuron, node_not_occu = node_not_occu)

    # Assume peak distribution obeys normal distribution, then the probability of the location at where the peak rate of a certain neuron locates is:
    # 1 / number of nodes that the mouse had at least once been to in a session (abreviation: occu_number). (If all of the 144 nodes had once been stood by mouse 
    # in a session, the probability is 1/144 for each nodes.)
    # MeanGraph is the array (whose length is 144) that if nodes have been occupied, the value is 1/occu_number, else, the value is np.nan
    MeanGraph = np.ones(old_map_all.shape[1], dtype = np.float64) / (old_map_all.shape[1] - node_not_occu.shape[0])
    MeanGraph[node_not_occu] = np.nan

    ErrorGraph = np.abs(PDGraph - MeanGraph)
    RMSE = np.sqrt(np.nanmean(ErrorGraph**2))
    MAE = np.nanmean(ErrorGraph)
    return RMSE, MAE

# Calculate chance level
def PeakDistributionDensityChanceLevel(n_neuron = 400, SilentNeuron = np.array([], dtype = np.int64), node_not_occu = np.array([], dtype = np.int64)):
    # Delete Silent Neuron
    n_neuron = n_neuron - SilentNeuron.shape[0]
    # Delete nodes that were never occupied by mouse in a session.
    select_range = np.delete(np.arange(1,145), node_not_occu)

    rand_center = np.random.choice(select_range, size = n_neuron, replace = True)
    old_map_all = np.zeros((n_neuron,144), dtype = np.float64)
    # Values of nodes that have been randomly selected as peak rate are set as 1 and others a 0, for each neuron. then the Value(i.e. 1) is naturally the peak rate value.
    for i in range(n_neuron):
        old_map_all[i,rand_center[i]-1] = 1
    
    RMSE, MAE  = PeakDistributionDensity(old_map_all = old_map_all, node_not_occu = node_not_occu)
    return RMSE, MAE
# ---------------------------------------------------------------------------------
# Decoding Map

def AlignSpikeLocation(spike_time:np.ndarray = None, pos_time:np.ndarray = None):
    '''
    Author: YAO Shuyang
    Date: Jan 23, 2023
    Note: This function is to aligned spike_time to pos_time.

    Input:
    - spike_time: numpy.ndarray, shape = (T,), T is the number of total frames of ms video.
    - pos_time: numpy.ndarray, shape = (T',), T' is the number of total frames of behavior video.
    
    Output:
    - numpy.ndarray, shape = (T,). Align ms time to behavior time.
    '''
    spike_time = cp.deepcopy(spike_time)
    spike_time[spike_time > pos_time[-1]] = pos_time[-1]
    spike_time[spike_time < pos_time[0]] = pos_time[0]
    
    idxs = np.zeros(spike_time.shape[0], dtype = np.int64)
    for i in range(idxs.shape[0]):
        idx = np.where(spike_time[i] >= pos_time)[0][-1]
        idxs[i] = idx if spike_time[i] - pos_time[idx] >= pos_time[idx+1] -spike_time[i] else idx + 1
    
    return idxs
        

def DecodingMap(BayesModel):
    if BayesModel.res == 48:
        pred = S2F[BayesModel.MazeID_predicted - 1]
        test = S2F[BayesModel.MazeID_test - 1]
    elif BayesModel.res == 24:
        pred = Q2F[BayesModel.MazeID_predicted - 1]
        test = Q2F[BayesModel.MazeID_test - 1]
    elif BayesModel.res == 12:
        pred = BayesModel.MazeID_predicted
        test = BayesModel.MazeID_test

    Hit_seq = np.where(pred == test, 1, 0)
    Hit = np.zeros(144, dtype = np.int64)
    Total = np.zeros(144, dtype = np.int64)
    print("    Decoding Map Generating...")
    for t in range(pred.shape[0]):
        Hit[test[t] - 1] += Hit_seq[t]
        Total[test[t] - 1] += 1
    
    return Hit / Total
    
# encoding turn around
def TurnRound(trace):
    maze_type = trace['maze_type']
    behav_nodes = cp.deepcopy(trace['correct_nodes'])

    with open('decoder_DMatrix.pkl', 'rb') as handle:
        D_Matrice = pickle.load(handle)
        D = D_Matrice[6+maze_type]

    # Turn Around nodes & Index
    TRNODE = []
    TRIDX = []
    dirc = 1
    S2F = Son2FatherGraph
    level = D[S2F[behav_nodes[0]]-1,0]

    for k in range(1,behav_nodes.shape[0]):
        if (D[S2F[behav_nodes[k]]-1,0] - level) * dirc < 0:
            dirc *= -1
            TRNODE.append(behav_nodes[k])
            TRIDX.append(k)

        level = D[S2F[behav_nodes[k]]-1,0]
    return TRNODE, TRIDX

def PeakCurveSplit(trace, p, order_type = 'A'):
    maze_type = trace['maze_type']
    loc = os.path.join(p, 'PeakCurveSplit-Order'+order_type)
    mkdir(loc)
    laps = trace['laps']
    old_map_split = trace['old_map_split']

    if maze_type in [1,2]:
        co_path = correct_path1 if maze_type == 1 else correct_path2
        ic_path = incorrect_path1 if maze_type == 1 else incorrect_path2
        length = co_path.shape[0]
        order = np.concatenate([co_path,ic_path])
    else:
        length = 144
        order = np.array(range(1,145))

    if order_type == 'B':
        for k in range(laps):
            loc = os.path.join(p, 'PeakCurveSplit-Order'+order_type, 'Lap '+str(k+1)+'.svg')
            clear_map_all = clear_NAN(old_map_split[k])[0]
            norm_map_all = Norm_ratemap(clear_map_all)
            norm_map_sort = norm_map_all[:,order-1]
            if k == 0:
                maxidx = np.nanargmax(norm_map_sort, axis = 1)
                order_n = np.array([], dtype = np.int64)
                for i in range(144,0,-1):
                    order_n = np.concatenate([order_n,np.where(maxidx == i-1)[0]])
        
            SortedMap = norm_map_sort[order_n,:]
            title = 'Lap '+str(k+1)
            DrawPeakCurveSplit(SortedMap, length, title, loc)

    elif order_type == 'A':
        for k in range(laps):
            loc = os.path.join(p, 'PeakCurveSplit-Order'+order_type, 'Lap '+str(k+1)+'.svg')
            clear_map_all = clear_NAN(old_map_split[k])[0]
            norm_map_all = Norm_ratemap(clear_map_all)
            norm_map_sort = norm_map_all[:,order-1]     
            SortedMap = sortmap(norm_map_sort)
            title = 'Lap '+str(k+1)
            DrawPeakCurveSplit(SortedMap, length, title, loc)        

def DrawPeakCurveSplit(rate_map, length, title = '', loc = ''):
    fig = plt.figure(figsize = (4,4))
    ax = plt.axes()
    im = ax.imshow(rate_map)
    cbar = plt.colorbar(im, ax = ax)
    ax.set_aspect('auto')
    ax.axvline(length-0.5, color = 'cornflowerblue', linewidth = 2)
    ax.set_xlabel('Orderred MazeID')
    ax.set_ylabel('sorted Neuron ID')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.savefig(loc, dpi = 600)
    plt.show()

# Read Cell Reg
def ReadCellReg(loc, open_type = 'h5py'):
    if open_type == 'h5py':
        with h5py.File(loc, 'r') as f:
            cell_registered_struct = f['cell_registered_struct']
            index_map = np.array(cell_registered_struct['cell_to_index_map'])
        return index_map

    elif open_type == 'scipy':
        f = scipy.io.loadmat(loc)
        cell_registered_struct = f['cell_registered_struct']
        index_map = np.array(cell_registered_struct['cell_to_index_map'])
        return index_map

# Count cell reg number.
def CountCellReg(cellReg = None):
    if cellReg is None:
        print("Warning! Need a path for cellRegistered.mat!")
        return
    
    if os.path.exists(cellReg):
        index_map = ReadCellReg(cellReg)
    else:
        print("Warning! "+cellReg+" is not exist!")
        return

    map01 = np.where(index_map == 0, 0, 1)
    count = np.nansum(map01, axis = 0)
    
    sums = np.zeros(index_map.shape[0]+1, dtype = np.int64)
    for i in range(index_map.shape[0]+1):
        sums[i] = len(np.where(count == index_map.shape[0] - i)[0])
        print("Aligned number = "+str(index_map.shape[0] - i)+": "+str(sums[i]))

    return sums

def spike_nodes_transform(spike_nodes = None, nx = 48):
    spike_nodes = spike_nodes.astype(np.int64)
    if nx == 48:
        return spike_nodes # transformed in vain
    elif nx == 24:
        S2FGraph = S2Q
    elif nx == 12:
        S2FGraph = S2F

    spike_nodes_aft = S2FGraph[spike_nodes-1]
    return spike_nodes_aft

def occu_time_transform(occu_time = None, nx = 48):
    if nx == 48:
        return occu_time # transformed in vain
    elif nx == 24:
        F2SGraph = Quarter2SonGraph
    elif nx == 12:
        F2SGraph = Father2SonGraph
    
    occu_time_aft = np.zeros(nx*nx, dtype = np.float64)
    for i in range(nx*nx):
        for s in F2SGraph[i+1]:
            if np.isnan(occu_time[s-1]) == False:
                occu_time_aft[i] += occu_time[s-1]
    occu_time_aft[np.where(occu_time_aft <= 50)[0]] = np.nan
    return occu_time_aft

# Delete NAN value in positions. Keep the frames of position equal to the frames of behav_time
def Delete_NAN(behav_positions = None, behav_time = None, behav_nodes = None):
    idx = np.where((np.isnan(behav_positions[:,0]) == False)&(np.isnan(behav_positions[:,1]) == False))[0]

    if behav_nodes is None:
        return cp.deepcopy(behav_positions[idx,:]), cp.deepcopy(behav_time[idx])
    elif behav_nodes is not None:
        return cp.deepcopy(behav_positions[idx,:]), cp.deepcopy(behav_time[idx]), cp.deepcopy(behav_nodes[idx])

# Add NAN value at time Gap point between adjacent two lap
def Add_NAN(behav_positions = None, behav_time = None, behav_nodes = None, time_gap_interval = 4000, maze_type = 0):
    if maze_type == 0:
        if behav_nodes is None:
            return behav_positions, behav_time
        elif behav_nodes is not None:
            return behav_positions, behav_time, behav_nodes

    stay_time = np.append(np.ediff1d(behav_time),33)
    idx = np.where(stay_time > time_gap_interval)[0] # gap time index
    time_points = behav_time[idx]
    time_points_add = time_points + 33

    # insert interpolated time point
    behav_time_modi = cp.deepcopy(behav_time)
    behav_time_modi = np.insert(behav_time_modi, idx+1, time_points_add)
    # insert interpolated NAN position point
    behav_positions_modi = cp.deepcopy(behav_positions)
    behav_positions_modi = np.insert(behav_positions_modi, idx+1, np.zeros((len(idx),2))*np.nan, axis = 0)
    
    if behav_nodes is None:
        return behav_positions_modi, behav_time_modi
    elif behav_nodes is not None:
        # insert interpolated NAN position node point
        behav_nodes_modi = cp.deepcopy(behav_nodes)
        behav_nodes_modi = np.insert(behav_nodes_modi, idx+1, np.repeat(np.nan, len(idx)))
        return behav_positions_modi, behav_time_modi, behav_nodes_modi

def Generate_Venn3_Subset(Group_A, Group_B, Group_C):
    # Only A
    A =  len(np.where((Group_A == 1)&(Group_B == 0)&(Group_C == 0))[0])
    # Only B
    B = len(np.where((Group_A == 0)&(Group_B == 1)&(Group_C == 0))[0])
    # Only C
    C = len(np.where((Group_A == 0)&(Group_B == 0)&(Group_C == 1))[0])
    # Both A and B, but not C
    A_B = len(np.where((Group_A == 1)&(Group_B == 1)&(Group_C == 0))[0])
    # Both A and C, but not B
    A_C = len(np.where((Group_A == 1)&(Group_B == 0)&(Group_C == 1))[0])
    # Both B and C, but not A
    B_C = len(np.where((Group_A == 0)&(Group_B == 1)&(Group_C == 1))[0])
    # All A,B,C
    A_B_C = len(np.where((Group_A == 1)&(Group_B == 1)&(Group_C == 1))[0])
    return [A,B,A_B,C,A_C,B_C,A_B_C]

# Get a colorbar ticks or y axis ticks according to peak value.
def ColorBarsTicks(peak_rate:float = 10, intervals:float = 1, is_auto:bool = False, tick_number:int = 8):
    '''
    Author: YAO Shuyang
    Date: Sept 1st, 2022
    Note: This funciton is to set ticks for figures, including xticks/yticks for figure and ticks of colorbars (in ratemap). Although usually the ticks could automatically be set by matplotlib package itself, it cannot show the peak value of plotted data. So I develop this method to return a vector with peak rate and other ticks value automatically.

    Input:
    - peak_rate: float, the peak rate you want to return and show in ticks.
    - intervals: float, the intervals between 2 neighbored ticks. If you choose to automatically (is_auto = True) set the interval, you do not need to input this parameter.
    - is_auto: bool, to determine whether set the ticks automatically.
    - tick_number: int, a parameter to set the number of ticks you want. Note that the finally number of ticks may be not equal to the number you input (e.g. +1 or -1).

    Output:
    - <class 'numpy.ndarray'>
    '''
    # Automatically set the stick value
    if is_auto == True:
        # A tick unit vector
        round_edge = np.array([0.1, 0.2, 0.4, 0.5, 0.8, 1, 2, 4, 5, 8, 10, 12, 15, 20, 40, 50, 80, 100, 150, 200, 400, 500, 800, 1000], dtype = np.float64)
        assert peak_rate >= round_edge[0] and peak_rate <= 30 * round_edge[-1] # the peak rate is not suit for this function. Add new number in round_edge or give up using this function.
        # distance vector
        dis_vec = np.abs(peak_rate/int(tick_number) - round_edge)
        fittest_unit_index = np.argmin(dis_vec)
        # Set intervals as the fittest unit
        intervals = round_edge[fittest_unit_index]

    num = int(peak_rate // intervals)
    ticks = np.linspace(0, num * intervals, num+1)

    if peak_rate - num*intervals <= 0.4*intervals:
        ticks[-1] = peak_rate
    else:
        ticks = np.append(ticks, peak_rate)

    return ticks

# plot shadows on some area.
def ax_shadow(ax = None, x1_list = np.array([]), x2_list = np.array([]), y = np.array([]), palette = 'muted', alpha = 0.5, **kwargs):
    if len(x1_list) != len(x2_list):
        print("Error! The length of x1_list is not equal to the length of x2_list.")
        return ax

    # Number of shadow areas.
    areas_num = len(x1_list)
    # length (or density) of y axis points.
    y_len = len(y)
    colors = sns.color_palette(palette = palette, n_colors = areas_num)

    for i in range(areas_num):
        # plot shadows areas.
        ax.fill_betweenx(y = y, x1 = np.repeat(x1_list[i], y_len), x2 = np.repeat(x2_list[i], y_len), color = colors[i], alpha = alpha, **kwargs)

    return ax

def CopyFile(source_path = None, targ_path = None, copyfile = 'file'):
    if source_path is None or targ_path is None:
        print("    InputError! Both 'source_path' and 'targ_path' are key args.")
        return
    
    if copyfile == 'file':
        if os.path.exists(source_path):
            dir = os.path.dirname(targ_path)
            if os.path.exists(dir) == False:
                mkdir(dir)
            shutil.copy(source_path, targ_path)
        else:
            print(source_path, 'is not exist!')
    elif copyfile == 'dir':
        if os.path.exists(source_path):
            if os.path.exists(targ_path):
                shutil.rmtree(targ_path)
            shutil.copytree(source_path, targ_path)
        else:
            print(source_path, 'is not exist!') 


# plot animal's trajactory.
def plot_trajactory(x, y, maze_type = 1, is_DrawMazeProfile = False, is_ExistAxes = False, ax = None, color = None, 
                    nx = 48, save_loc = None, file_name = None, figsize = (6,6), linewidth = 2, is_show = False, is_GetAxes = False,
                    is_inverty = False, clearax_args = {}):
    if save_loc is None and is_GetAxes == False and is_show == False:
        print("Argument save_loc is required!")
        return
    if file_name is None and is_GetAxes == False and is_show == False:
        print("Argument file_name is required!")
        return

    if is_ExistAxes == False:
        fig = plt.figure(figsize = figsize)
        ax = plt.axes()
    else:
        if ax is None:
            print("Error! ax must not be empty if you want input an axes.")
            return

    # Draw figures
    ax = Clear_Axes(axes = ax, **clearax_args)
    if is_DrawMazeProfile == True:
        ax = DrawMazeProfile(axes = ax, maze_type = maze_type, nx = nx, linewidth = linewidth, color = 'black')
    
    if color is not None:
        ax.plot(x,y,linewidth = linewidth, color = color)
    else:
        ax.plot(x,y,linewidth = linewidth)

    if is_inverty == True:
        ax.invert_yaxis()

    # return an axes if you want
    if is_GetAxes == True:
        return ax

    if is_show == True:
        plt.show()
    else:
        if os.path.exists(save_loc) == False:
            mkdir(save_loc)
        plt.savefig(os.path.join(save_loc,file_name+'.png'), dpi = 600)
        plt.savefig(os.path.join(save_loc,file_name+'.svg'), dpi = 600)
        plt.close()

# In some cases, if main field center of two cells are close to each other in a certain environment which we tentatively term it as A, they their main field center in 
# another environment may be close to each other as well. To test whether the phenomenone truely happen when we switch the environment from open field to maze 1 and 
# from maze 1 to maze 2, function InterFieldCenterDistance() are inplemented.
def InterFieldCenterDistance(center_list1:np.ndarray, center_list2:np.ndarray, maze_list = np.array([0,1,2], dtype = np.int64)):
    if len(center_list1) < 2 or len(center_list2) < 2:
        print("Warning! Only 2 environment are identified.")
    
    if len(center_list1) != len(center_list2) or center_list1.shape[0] != maze_list.shape[0]:
        print("LengthError! Two center lists must have the same length, but the inputted two are not satisfied with the requirement.")
        return None

    # Distance list in each environment (maze_list)
    IFCD = np.zeros(center_list1.shape[0])

    # Calculate center distance.
    for i in range(center_list1.shape[0]):
        if maze_list[i] != 0:
            IFCD[i] = Cartesian_distance(curr = center_list1[i], surr = center_list2[i], nx = 12)#FastDistance(start = center_list1[i], goal = center_list2[i], maze_type = maze_list[i], nx = 12)
        else:
            IFCD[i] = Cartesian_distance(curr = center_list1[i], surr = center_list2[i], nx = 12)
    
    return IFCD

# Return Field Number Vector of A Session
def field_number_session(trace:dict, is_placecell = True, spike_thre = None):
    '''
    Parameter
    ---------
    is_placecell: bool

    spike_thre: int
        Default None, the number of spikes. If a cell has total spikes less than spike_thre, delete it.
        - 30 is recommended.
    '''
    # Check if crucial keys exist.
    KeyWordErrorCheck(trace, __file__, keys = ['is_placecell', 'place_field_all', 'SilentNeuron'],)
    
    if is_placecell == True:
        idx = np.where(trace['is_placecell'] == 1)[0]
    else:
        idx = np.arange(trace['is_placecell'].shape[0])

    if spike_thre:
        spike_num = np.nansum(trace['Spikes'], axis = 1)
        cell_idx = np.where(spike_num >= spike_thre)[0]
        idx = np.intersect1d(cell_idx, idx)
    
    field_number = np.zeros(len(idx), dtype = np.float64)
    for i in range(idx.shape[0]):
        if idx[i] in trace['SilentNeuron']:
            field_number[i] = np.nan
        else:
            field_number[i] = len(trace['place_field_all'][idx[i]].keys())
    
    return field_number

def SubDict(trace:dict, keys:list = [], idx:list|np.ndarray = None):
    '''
    Author: YAO Shuyang
    Date: Jan 26th, 2023
    Note: Return a subset of trace. Note that the length of values of each key must have be same, or it will raise an error!!!!

    # Input:
    - trace: dict, the father dict.
    - keys: list, contains keys that you want to keep in subset.
    - idx: the index of elements that you want to keep in subset.

    # Output:
    - A dict.
    '''
    KeyWordErrorCheck(trace, __file__, keys = keys)
    assert len(keys) != 0
    assert idx is not None

    l = len(idx)
    subdic = {}

    for k in keys:
        try: subdic[k] = cp.deepcopy(trace[k][idx])
        except:
            assert False
    
    return subdic

def print_nan(l:list[list]|list[np.ndarray]):
    '''
    Date: Jan 27th, 2023
    Note: input an numpy.ndarray or list, it will return whether it contains NAN value. 
    '''
    
    for vec in l:
        if type(vec) is list:
            vec = np.array(vec)
    
        print(np.where(np.isnan(vec)))

        
def Get_X_Order(maze_type:int):
    '''
    Author: YAO Shuyang
    Date: Jan 27th, 2023
    Note: Get the x order that used to sort the x axis of peak curve and old_rate_map by correct path and incorrect path.

    Parameter
    ---------
    - maze_type: int, only {0,1,2} are valid data.

    Returns
    -------
    - NDarray[int], (144,)
    - Separate Point, int
    '''
    ValueErrorCheck(maze_type, [0,1,2])

    order_list = [np.arange(144), np.concatenate([CorrectPath_maze_1, IncorrectPath_maze_1])-1, np.concatenate([CorrectPath_maze_2, IncorrectPath_maze_2])-1]
    separate_list = [143.5, CorrectPath_maze_1.shape[0]-0.5, CorrectPath_maze_2.shape[0]-0.5]
    return order_list[int(maze_type)], separate_list[int(maze_type)]

def calc_PVC(rate_map_all1:np.ndarray, rate_map_all2:np.ndarray):
    '''
    Author: YAO Shuyang
    Date:Jan 27th, 2023
    Note: to calculate the population vector correlation(PVC) of two vector. It requires the input two matrix have the same shape, that is, the same number of rows and columns, or it will raise an error!!!!!

    Parameter
    -----------
    - rate_map_all1: np.ndarray, (number of neuron, number of spatial bins), in our maze cases it was (n-neuron, 144). Note that (n-neuron, 2304) and any other shape is deprecated and if input matrice with this type of shape will raise an error!!!!
    - rate_map_all2: np.ndarray. Obeys the same requirement as 'rate_map_all1'.

    Returns
    ---------
    - A NDarray[float] matrix, (144, 144).
    '''
    assert rate_map_all1.shape == rate_map_all2.shape and rate_map_all1.shape[1] == 144 # the input 2 matrice should have same shape and the number spatial bins in our case should be and only be 144. (2304 is deprecated.)

    PVC = np.zeros((144,144), dtype=np.float64)

    for i in range(144):
        for j in range(144):
            # when meat a location that mice have not been to, set the correlation as np.nan
            if np.nansum(rate_map_all1[:,i]) == 0 or np.nansum(rate_map_all2[:,j]) == 0:
                PVC[i,j] = np.nan
            else:
                PVC[i,j], _ = pearsonr(rate_map_all1[:,i], rate_map_all2[:,j])

    return PVC 

def calc_pearsonr(rate_map_all1:np.ndarray, rate_map_all2:np.ndarray = None):
    '''
    Author: YAO Shuyang
    Date:Jan 27th, 2023
    Note: to calculate the population vector correlation(PVC) of two vector. It requires the input two matrix have the same shape, that is, the same number of rows and columns, or it will raise an error!!!!!

    Parameter
    -----------
    - rate_map_all1: np.ndarray, (n_neuron1, number of spatial bins), in our maze cases it was (n-neuron, 144). Note that (n-neuron, 2304) and any other shape is deprecated and if input matrice with this type of shape will raise an error!!!!
    - rate_map_all2: np.ndarray, (n_neuron2, number of spatial bins), optional. If it is None, we will set it as rate_map_all1.

    Returns
    ---------
    - A NDarray[float] matrix, (n_neuron1, n_neuron2).
    '''
    if rate_map_all2 is None:
        rate_map_all2 = cp.deepcopy(rate_map_all1)
    assert rate_map_all1.shape[1] == rate_map_all2.shape[1] # the input 2 matrice should have same number of spatial bins.

    PearsonC = np.zeros((rate_map_all1.shape[0], rate_map_all2.shape[0]), dtype=np.float64)

    for i in range(rate_map_all1.shape[0]):
        for j in range(rate_map_all2.shape[0]):
            # when meat a location that mice have not been to, set the correlation as np.nan
            if np.nansum(rate_map_all1[i,:]) == 0 or np.nansum(rate_map_all2[j,:]) == 0:
                PearsonC[i,j] = np.nan
            else:
                PearsonC[i,j], _ = pearsonr(rate_map_all1[i,:], rate_map_all2[j,:])

    return PearsonC

# Fit equal poison distribution:
from scipy.optimize import leastsq
import math
from scipy.special import factorial
from scipy.stats import anderson # Anderson-Darling goodness of fit test, Dylan et al 2014.

def EqualPoisson(x:np.int64, l:float):
    '''
    Parameters
    ----------
    l: The lambda
    x: The number
    '''
    x = np.int64(x)
    return (l**(x)) * (np.exp(-l)) / factorial(x)

def EqualPoissonResiduals(l:float, x:np.int64, y:np.float64):
    return y - EqualPoisson(x,l)

def EqualPoissonFit(x, y, l0:float = 5):
    para = leastsq(EqualPoissonResiduals, x0 = l0, args = (x, y))
    return para[0][0]

def sort_dlc_file(dir_name: str):
    """
    Sort dlc files with recording order before concatenating.

    Parameters
    ----------
    dir_name : str
        the directory of where saves the dlc files.
    """

    files = os.listdir(dir_name)

    return sorted(files, key=lambda file: os.path.getctime(os.path.join(dir_name, file)))

    

def DLC_Concatenate(directory: str, find_chars: str = '.csv', body_part: list = ['bodypart1', 'bodypart2', 'bodypart3', 'bodypart4'],
                    **kwargs):
    """
    concatenate dlc excel files together

    Parameter
    ---------
    directory: str, required
        The directory that saves all of the files that needed to be concatenate. An order is needed.
    """

    try:
        bp = ['bodypart1', 'bodypart2', 'bodypart3', 'bodypart4']

        Data = {}
        for b in bp:
            Data[b] = np.array([[np.nan, np.nan]], dtype = np.float64)

        if_no_file = True

        if os.path.exists(directory):
            loc = os.path.join(directory, 'dlc_process_file')
            mkdir(loc)
            files = sort_dlc_file(directory)   # Sort dlc files according to their established time.
            for file in files:
                if find_chars in file:
                    if '.csv' in file:
                        if_no_file = False
                        f = pd.read_csv(os.path.join(directory, file), **kwargs)
                        for b in bp:
                            coord = np.zeros((len(f), 2), dtype=np.float64)
                            coord[:,0] = f[b, 'x']
                            coord[:,1] = f[b, 'y']
                            Data[b] = np.concatenate([Data[b], coord], axis=0)

                    shutil.move(os.path.join(directory, file), os.path.join(loc, file))
    except:
        bp = ['bodypart1', 'bodypart2', 'bodypart3', 'objectA']

        Data = {}
        for b in bp:
            Data[b] = np.array([[np.nan, np.nan]], dtype = np.float64)

        if_no_file = True

        if os.path.exists(directory):
            loc = os.path.join(directory, 'dlc_process_file')
            mkdir(loc)
            files = sort_dlc_file(directory)   # Sort dlc files according to their established time.
            for file in files:
                if find_chars in file:
                    if '.csv' in file:
                        if_no_file = False
                        f = pd.read_csv(os.path.join(directory, file), **kwargs)
                        for b in bp:
                            coord = np.zeros((len(f), 2), dtype=np.float64)
                            coord[:,0] = f[b, 'x']
                            coord[:,1] = f[b, 'y']
                            Data[b] = np.concatenate([Data[b], coord], axis=0)

                    shutil.move(os.path.join(directory, file), os.path.join(loc, file))

    for b in bp:
        Data[b] = Data[b][1::, :]

    if if_no_file == False:
        with open(os.path.join(directory, 'dlc_coord.pkl'), 'wb') as f:
            pickle.dump(Data, f)
    else:
        warnings.warn(f"There's no DLC processed csv files in directory '{directory}'")

def peak_velocity(behav_speed: np.ndarray, behav_nodes: np.ndarray, idx: int):
    """
    Parameter
    ---------
    behav_speed: numpy.ndarray
        The mice's speed recorded at each frame.

    behav_nodes: numpy.ndarray
        The mice's location (Spatial bin ID) recorded at each frame.

    idx: int
        The index of the spatial bin. We want to have a look at the velocity of mice when they passed the bin.
    """
    assert behav_nodes.shape[0] == behav_speed.shape[0]

    behav_nodes = spike_nodes_transform(spike_nodes=behav_nodes, nx = 12)

    idx = np.where(behav_nodes == idx)[0]
    return behav_speed[idx]


if __name__ == '__main__':
    loc = r"E:\CC\MAZE_2\2022_08_30\11095\15_19_57\My_WebCam\dlc_process_file"
    
    print(os.listdir(loc))
    print("---------------------------------------------")
    print(sort_dlc_file(loc))