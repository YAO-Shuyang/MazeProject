from mylib.preprocessing_behav import *
from mylib.preprocessing_ms import *
from mylib.Interfaces import *
import seaborn as sns
import scipy.io
from scipy.stats import ttest_1samp, ttest_ind, levene
from mylib.local_path import *
from scipy.stats import linregress, pearsonr
from matplotlib.gridspec import GridSpec
import gc

CONVERGE_MODEL_COLOR, EQUALRATE_MODEL_COLOR = '#D9A6A9', '#9C8FBC'

def star(p:str):
    '''
    Note: notations of significance according to the input p value.

    # Input:
    - p:str, the p value.

    # Output:
    - str, the notation.
    '''
    if p > 0.05:
        return 'ns'
    elif p <= 0.05 and p > 0.01:
        return '*'
    elif p <= 0.01 and p > 0.001:
        return '**'
    elif p <= 0.001 and p > 0.0001:
        return '***'
    elif p <= 0.0001:
        return '****'

def plot_star(left:list = [0.1], right:list = [0.9], height:list = [1.], delt_h:list = [0.1], ax = None, p:list = [0.5], 
              color:str = 'black', barwidth:float = 2., fontsize:float = 12., **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 26, 2023 (Noted)
    Note: this function is used to plot results of significance test(stars or ns) on figure. parameter left/right/height/delt_h/p must have same length, or it will raise an error.

    # Input:
    - left: list[float], contains left locations of horizontal bars that under the stars/ns.
    - right: list[float], contains right locations of horizontal bars that under the stars/ns.
    - height: list[float], contains the y locations of each horizontal bar that under the stars/ns.
    - delt_h: list[float], contains the length of the vertical protuberances that on the two ends of each horizontal bars.
    - ax: matplotlib axes object. This paraemter is required
    - p: list[float], contains the results of significance test.
    - color: str, the color of bars.
    - barwidth: float, the width of horizontal bars.
    - fontsize: float, the size of stars/ns.

    # Output:
    - matplotlib axes object.
    '''
    dim = len(left)
    if len(right) != dim or len(height) != dim or len(delt_h) != dim or len(p) != dim:
        print('Input list length error!')
        return ax

    for i in range(dim):
        ax.plot([left[i],right[i]],[height[i],height[i]], color = color, linewidth = barwidth, **kwargs)
        ax.text((left[i]+right[i])/2, height[i] + delt_h[i], star(p[i]), ha = 'center', fontsize = fontsize)
    return ax

# Get a collection of trace object(dict).
def TraceFileSet(
    idx:np.ndarray, 
    f:pd.DataFrame, 
    is_behav: bool = False
) -> list:
    '''
    Author: YAO Shuyang
    Date: Sept 26th, 2023
    Note: To combine trace into a list for the convinence to call.

    Input:
    - f: <class 'pandas.DataFrame'>, the guide f that saves basic information of each session.
    - idx: <class 'numpy.ndarray'>, the indexes (of row) of sessions that you want to choose to combine a list.
    - is_behav: bool, whether the input is behavioral trace or not.

    Output:
    - A list, contains several dicts (trace of each session) and the length of this list is equal to the length of idx.
    '''

    trace_set = []
    
    if is_behav == False:
        key = 'Trace File'
    else:
        key = 'Trace Behav File'
    
    for i in tqdm(idx):
        if os.path.exists(f[key][i]):
            with open(f[key][i], 'rb') as handle:
                trace = pickle.load(handle)
            trace_set.append(trace)
        else:
            trace_set.append(None)
    
    return trace_set

# Index matrix is generated for the convience to plot sample cells.
def Generate_IndexMatrix(f, dateset = ['20220830'], beg_idx = None, end_idx = None):
    if beg_idx is None or beg_idx < 0:
        beg_idx = 0
    if end_idx is None or end_idx > len(f):
        end_idx = len(f)
    IndexMatrix = np.zeros((end_idx - beg_idx, len(dateset)), dtype = np.int64)
    for d in range(len(dateset)):
        IndexMatrix[:,d] = np.array(f[dateset[d]][beg_idx:end_idx], dtype = np.int64)
    return IndexMatrix



# According to figure 4, there's a gap between each block of incorrect path. Now we split old_map incorrect part into several 
# blocks with NAN gap.
def IncorrectMap(old_map_all = None, maze_type = 1, is_norm = True, is_sort = True):
    SIM = Split_IncorrectPath_Map1 if maze_type == 1 else Split_IncorrectPath_Map2
    n_neuron = old_map_all.shape[0]
    
    incorrect_map = cp.deepcopy(old_map_all[:, SIM[0,1:1+SIM[0,0]]-1])
    x_ticks = [SIM[0,0]/2 - 0.5]

    for i in range(1,SIM.shape[0]):
        incorrect_map = np.concatenate([incorrect_map, np.zeros((n_neuron,2))*np.nan], axis = 1)
        x_ticks.append(incorrect_map.shape[1] - 0.5 + SIM[i,0]/2)
        incorrect_map = np.concatenate([incorrect_map, old_map_all[:, SIM[i,1:1+SIM[i,0]]-1]], axis = 1)
    
    x_labels = np.linspace(0, SIM.shape[0]-1, SIM.shape[0])+1
    x_labels = x_labels.astype(np.int64)

    if is_norm:
        incorrect_map = Norm_ratemap(incorrect_map)
        
    if is_sort:
        incorrect_map = sortmap(incorrect_map)
    
    return incorrect_map, x_ticks, x_labels

# Add line to represent correct path
def Add_NAN_Line(ax = None, incorrect_map = None, is_single = False, linewidth = 1):

    if is_single == False:
        idx = np.where(np.isnan(incorrect_map[0,:]))[0]
        n = incorrect_map.shape[0]/2
        for i in idx:
            ax.plot([i-0.5,i+0.5], [n,n], color = 'black', linewidth = linewidth)
    else:
        idx = np.where(np.isnan(incorrect_map))[0]
        for i in idx:
            ax.plot([i-0.5,i+0.5], [0,0], color = 'black', linewidth = linewidth)        
    return ax


# Generate data about some key variables. Generate all data from a behavior paragidm.
def DataFrameEstablish(variable_names: list = [], f:pd.DataFrame = f1, function = None, 
                       file_name:str = 'default', behavior_paradigm:str = 'CrossMaze', 
                       legal_maze_type:list = [0,1,2,3], f_member: list|None = None, 
                       file_idx:np.ndarray|list = None, func_kwgs:dict = {}, 
                       is_behav: bool = False):
    '''
    Author: YAO Shuyang
    Date: Jan 25th, 2023 (Modified)
    Note: This function is to calculate some variables and concatenate data cross a long training period together to form a DataFrame, and this data frame can be used to plot figures.

    Input: 
    - variable_names: list, represent the variables that will be saved in data.
    - f: pandas.DataFrame, files that save information of sessions.
    - function: processing function. the function must have args 'trace', 'spike_threshold'.
    - file_name: str, file_name of saved pkl file.
    - behavior_paradigm: str, 'CrossMaze','SimpleMaze','ReverseMaze' are 3 valid data.
    - f_member: str or None. If it gets an input (str), it delivers the member of list f_member to f, and get correlated value from f.

    Output:
    - A dict
    '''
    ValueErrorCheck(behavior_paradigm, ['CrossMaze', 'ReverseMaze', 'DSPMaze', 'HairpinMaze', 'SimpleMaze', 'decoding', 'CellReg CrossMaze'])

    # Initiate data dic
    data = {'MiceID':np.array([], np.int64), 'Training Day':np.array([]), 'Maze Type':np.array([]), 'Stage': np.array([]), 'date': np.array([], dtype=np.int64)}
    
    # Initiate additive member:
    if f_member is not None:
        for m in f_member:
            data[m] = np.array([])

    for c in variable_names:
        data[c] = np.array([], np.float64)

    if file_idx is None:
        follow = True
        file_idx = np.arange(len(f))
    else:
        follow = False

    if behavior_paradigm in ['CrossMaze', 'DSPMaze', 'ReverseMaze', 'HairpinMaze', 'CellReg CrossMaze']:
        if is_behav:
            keyw = 'Trace Behav File'
        else:
            keyw = 'Trace File'
    elif behavior_paradigm in ['decoding']:
        keyw = 'Results File'
    else:
        raise ValueError(f'behavior_paradigm should be in ["CrossMaze", "ReverseMaze", "DSPMaze", "HairpinMaze", "decoding", "CellReg CrossMaze"], while {behavior_paradigm} is not supported.')
        

    for i in tqdm(file_idx):
        # delete abnormal sessions.
        if f['include'][i] == 0:
            continue
        
        p = f[keyw][i]
        if os.path.exists(p):
            with open(p, 'rb') as handle:
                trace = pickle.load(handle)
        else:
            print(p,' is not exist!')
            continue

        # if maze_type is not we want, continue
        
        if 'maze_type' in trace.keys():
            if trace['maze_type'] not in legal_maze_type and follow:
                continue
        else:
            trace['maze_type'] = int(f['maze_type'][i])
        
        # Running funcitons to get variables we want to analysis.
        results = function(trace, variable_names = variable_names, **func_kwgs)
        # length of each variables in dictionary 'data' must be the same with others.
        if len(variable_names) == 1:
            length  = len(results)
            results = [results]
        else:
            length = len(results[0])

        training_day = str(f['training_day'][i])
        stage = str(f['Stage'][i])

        # Generating data.
        if behavior_paradigm in ['HairpinMaze']:
            mazes = 'HairpinMaze'
        else:
            mazes = 'Maze '+str(trace['maze_type']) if trace['maze_type'] in [1,2] else 'Open Field'
        data['MiceID'] = np.concatenate([data['MiceID'], np.repeat(int(f['MiceID'][i]), length)])
        data['Maze Type'] = np.concatenate([data['Maze Type'], np.repeat(mazes, length)])
        data['Training Day'] = np.concatenate([data['Training Day'], np.repeat(training_day, length)])
        data['Stage'] = np.concatenate([data['Stage'], np.repeat(stage, length)])
        data['date'] = np.concatenate([data['date'], np.repeat(int(f['date'][i]), length)])

        for c in range(len(variable_names)):
            data[variable_names[c]] = np.concatenate([data[variable_names[c]], results[c]])
        
        # Add additive values
        if f_member is not None:
            for m in f_member:
                data[m] = np.concatenate([data[m], np.repeat(f[m][i], length)])
                
        del trace
        gc.collect()
        
        
    print(np.shape(data['MiceID']))

    d = pd.DataFrame(data)
    try:
        d.to_excel(os.path.join(figdata, file_name+'.xlsx'), sheet_name = 'data', index = False)
    except:
        pass

    with open(os.path.join(figdata, file_name+'.pkl'), 'wb') as f:
        pickle.dump(data,f)

    return data
 
# get a sub collection of a dataframe
def DivideData(data:dict, index:list|np.ndarray, keys:list[str] = None):
    '''
    Date: Jan 10st, 2023
    
    Parameters
    ----------
    data: dict, required. 
            Note that each values corresponding to each key of data should have the same length or it will impossible to divide them into a subset by index.
    index: list or numpy.ndarray, required. 
            The index what you want to keep in the subset.
    keys: list[str], optional, default is None.
            If None, get all of the keys.

    Return
    ------
    A Dict
    '''
    if keys is None:
        keys = data.keys()
    
    subGroup = {}
    for k in keys:
        subGroup[k] = data[k][index]
    
    return subGroup


def plot_diagonal_figure(f:pd.DataFrame = None, row:int = None, map1:int = 1, map2:int = 2, save_loc:str = None, f_trace:pd.DataFrame = f1, function = None,
                         residents:float = 5, add_noise:bool = False, noise_amplitude:float = 1., **kwargs):
    '''
    Author: YAO Shuyang
    Date: Jan 26th, 2023
    Note: To plot a diagonal figure to show whether a property of cells that are shown in map1 will tend to show in map2. (Like more field number or more active in peak rate)

    # Input:
    - f: <class 'pandas.DataFrame'>, default value is f_CellReg which saves directories of all cross_session cellRegistered.mat file
    - row: int, the row id of the line want to read in f. Row should not be larger than the length of f. If f is default file(f_CellReg), row should not be bigger than 17.
    - map1: int, input a maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively)
    - map2: int, input another maze you want to choose. Only 0,1,2,3 are valid (0 and 3 represent Open Field 1 and Open Field 2, respectively, while 1 and 2 represent Maze 1/2 respectively). Map2 Should be different with Map1 (and are required to be bigger than map 1), or it will report an error!!!!!!!
    - save_loc: str, the location you want to save the diagonal figure.
    - f_trace: <class 'pandas.DataFrame'>, default value is f1 which saves basic information of cross_maze paradigm corresponding to f's default value f_CellReg.
    - function: the function is to generate data matrix.
    - residents: float, this parameter is to make a bit space to contain the legend.
    - add_noise: bool, if the data are discrete values, in order to distinguish points have same value, we can add some noise to distinguish same-value points.
    - noise_amplitude: float, if add noise, we should determine the amplitude of this noise.

    # Output:
    - (bool, dict). If the function has successfully run, return True or it will stop the funciton by 'AssertError' or return a False.
    '''
    assert row is not None and function is not None
    assert row < len(f) # row should not be bigger than the length of f, or it will overflow.
    assert map1 < map2 # map 1 should be bigger than map 2
    ValueErrorCheck(map1, [0,1,2,3])
    ValueErrorCheck(map2, [0,1,2,3])

    # Read and Sort Index Map
    print("Step 1 - Read And Sort Index Map")
    if os.path.exists(f['Cell Reg Path'][row]):
        index_map = Read_and_Sort_IndexMap(path = f_CellReg['Cell Reg Path'][row], occur_num = 2, align_type = 'cross_session')
    else:
        print(f['Cell Reg Path'][row], 'is not exist!')
        return False, None
    
    # Select Cell Pairs that Both exist in index_map in map1 and map2
    is_cell_detected = np.where(index_map == 0, 0, 1)
    cellpair = np.where(np.nansum(is_cell_detected[[map1,map2],:], axis = 0) == 2)[0]
    index_map = index_map[:, cellpair]
    index_map = index_map.astype(np.int64)

    if index_map.shape[1] == 0: # did not find cell pair that satisfy our requirement.
        return False, None

    # Get Trace File Set
    print("Step 2 - Get Trace File Set")
    idx = np.where((f_trace['MiceID'] == f['MiceID'][row])&(f_trace['date'] == f['date'][row]))[0]
    trace_set = TraceFileSet(idx = idx, file = f_trace, Behavior_Paradigm = 'Cross Maze')

    # Generate place fields number matrix
    print("Step 3 - Get Data From The Input Function")
    data = function(trace_set = trace_set, index_map = index_map, map1 = map1, map2 = map2)
    data_original = cp.deepcopy(data)
    # Add noise:
    if add_noise:
        assert noise_amplitude is not None
        data = data + np.random.rand(2, index_map.shape[1]) * noise_amplitude

    # Generate place cell list
    is_placecell = np.zeros((2, index_map.shape[1]), dtype = np.int64)
    is_placecell[0,:] = trace_set[map1]['is_placecell'][index_map[map1,:]-1]
    is_placecell[1,:] = trace_set[map2]['is_placecell'][index_map[map2,:]-1]

    # Plot diagonal figure
    print("Step 4 - Plot Figure")
    fig = plt.figure(figsize=(6,6))
    ax = Clear_Axes(axes = plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
    ax.set_aspect('equal')
    ticks = ColorBarsTicks(peak_rate = np.nanmax(data)+residents, is_auto = True)
    labels = ['Open Field 1', 'Maze 1', 'Maze 2', 'Open Field 2']
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(labels[map1])
    ax.set_ylabel(labels[map2])
    ax.axis([0, np.nanmax(data)+residents, 0, np.nanmax(data)+residents])
    
    # plot diagonal dot line
    ax.plot([0,np.nanmax(data)],[0,np.nanmax(data)],':', color = 'gray')

    # init classified data
    classified_data = {'Cell Pair Type':np.array([]), 'differences':np.array([], dtype = np.float64)}

    # Plot cell pair point at the cass:it was a place cell both in map1 and map2
    idx = np.where((is_placecell[0,:] == 1)&(is_placecell[1,:] == 1))[0]
    ax.plot(data[0,idx], data[1,idx], marker_list[0], color = 'red', label = 'pc-pc', **kwargs)
    classified_data['Cell Pair Type'] = np.concatenate([classified_data['Cell Pair Type'], np.repeat('pc-pc', len(idx))])
    classified_data['differences'] = np.concatenate([classified_data['differences'], data_original[1,idx] - data_original[0,idx]])

    # Plot cell pair point at the cass:it was a place cell in map1 but not in map2
    idx = np.where((is_placecell[0,:] == 1)&(is_placecell[1,:] == 0))[0]
    ax.plot(data[0,idx], data[1,idx], marker_list[1], color = 'orange', label = 'pc-npc', **kwargs) 
    classified_data['Cell Pair Type'] = np.concatenate([classified_data['Cell Pair Type'], np.repeat('pc-npc', len(idx))])
    classified_data['differences'] = np.concatenate([classified_data['differences'], data_original[1,idx] - data_original[0,idx]])  

    # Plot cell pair point at the cass:it was a place cell in map2 but not in map1
    idx = np.where((is_placecell[0,:] == 0)&(is_placecell[1,:] == 1))[0]
    ax.plot(data[0,idx], data[1,idx], marker_list[2], color = 'limegreen', label = 'npc-pc', **kwargs)
    classified_data['Cell Pair Type'] = np.concatenate([classified_data['Cell Pair Type'], np.repeat('npc-pc', len(idx))])
    classified_data['differences'] = np.concatenate([classified_data['differences'], data_original[1,idx] - data_original[0,idx]])

    # Plot cell pair point at the cass:it wasn't a place cell in both environment
    idx = np.where((is_placecell[0,:] == 0)&(is_placecell[1,:] == 0))[0]
    ax.plot(data[0,idx], data[1,idx], marker_list[3], color = 'black', label = 'npc-npc', **kwargs)
    classified_data['Cell Pair Type'] = np.concatenate([classified_data['Cell Pair Type'], np.repeat('npc-npc', len(idx))])
    classified_data['differences'] = np.concatenate([classified_data['differences'], data_original[1,idx] - data_original[0,idx]])

    ax.legend(facecolor = 'white', edgecolor = 'white', ncol = 4, title = 'Cell Pair (pc: place cell/npc: non-place cell)')
    plt.savefig(save_loc+'.png', dpi = 600)
    plt.savefig(save_loc+'.svg', dpi = 600)
    plt.close()
    print("Done.", end = '\n\n')

    if len(np.where(classified_data['Cell Pair Type'] == 'pc-pc')[0]) + len(np.where(classified_data['Cell Pair Type'] == 'pc-npc')[0]) <= 20:
        return False, classified_data
    else:
        return True, classified_data

def get_trace(i: int, f: pd.DataFrame, env1: str = 'op', env2: str = 'm1', f1: pd.DataFrame = f1, work_flow: str = r'G:\YSY\Cross_maze'):
    j, k = int(f[env1][i]), int(f[env2][i])

    idx1 = np.where((f1['MiceID'] == f['MiceID'][i])&(f1['date'] == f['date'][i])&(f1['session'] == j+1))[0][0]
    idx2 = np.where((f1['MiceID'] == f['MiceID'][i])&(f1['date'] == f['date'][i])&(f1['session'] == k+1))[0][0]

    if exists(f1['Trace File'][idx1]):
        with open(f1['Trace File'][idx1], 'rb') as handle:
            trace1 = pickle.load(handle)
    else:
        trace1 = None

    if exists(f1['Trace File'][idx2]):
        with open(f1['Trace File'][idx2], 'rb') as handle:
            trace2 = pickle.load(handle)
    else:
        trace2 = None
    
    return trace1, trace2

def get_placecell_pair(i: int, f: pd.DataFrame, env1: str = 'op', env2: str = 'm1', f1: pd.DataFrame = f1, work_flow: str = r'G:\YSY\Cross_maze'):
    trace1, trace2 = get_trace(i, f, env1, env2, f1, work_flow)
    if trace1 is None or trace2 is None:
        return np.array([[],[]], dtype=np.float64)

    index_map = ReadCellReg(f['Cell Reg Path'][i])
    j, k = int(f[env1][i]), int(f[env2][i])

    idx = np.where((index_map[j, :]!=0)&(index_map[k, :]!=0))[0]
    index_map_pair = index_map[:, idx].astype(np.int64)

    idx = np.where((trace1['is_placecell'][index_map_pair[j, :]-1]==1)&(trace2['is_placecell'][index_map_pair[k, :]-1]==1))[0]
    index_map_pcpair = index_map_pair[:, idx]
    return index_map_pcpair[np.array([j, k]), :]


def Read_and_Sort_IndexMap(
    path:str = None, 
    occur_num:int = 6, 
    align_type:str = 'cross_day', 
    name_label:str = 'SFP2022',
    order = np.array(['20220820', '20220822', '20220824', '20220826', '20220828', '20220830']) # if align_type == 'cross_day' else np.array(['1','2','3','4'])
) -> np.ndarray:
    '''
    Author: YAO Shuyang
    Date: Jan 25th, 2023 (Modified)
    Note: This function is written to
    1. Read index_map from cellRegistered.mat.
    2. Sort them with certain order.
    3. Select cell sequences that satisfy certain requirement. The requirement is the number of cells in the cell sequence. If the number >= occur_numm, it is satisfied.

    Input:
    - path: str, the directory of cellRegistered.mat f
    - occur_num: int. The threshold to select cell sequences that satisfy this threshold. The default value for 'cross_day' file is 6 and default value for 'cross_session' is 4.
    - align_type: str. Determines whether sessions that are aligned are recorded in different days (cross_day) or different sessions in one day (cross_session). Only 'cross_day' and 'cross_session' are valid value.
    - name_label: str, the cell_reg name label in logFile.txt

    Output:
    - <class 'numpy.ndarray'>
    '''
    ValueErrorCheck(align_type, ['cross_day','cross_session']) # InputContentError! Only 'cross_day' and 'cross_session' are valid value!

    if align_type == 'cross_session' and occur_num == 6:
        occur_num = 4

    # ReadCellReg
    index_map = ReadCellReg(loc = path)

    # Index_map does not have the proper length as the input order!
    assert order.shape[0] <= index_map.shape[0]

    nrow = index_map.shape[0]
    index_map_reorder = np.zeros((order.shape[0], index_map.shape[1]), dtype = np.int64)
    # read log.txt file to reorder the index_map, with the order of date or session, according to align_type.
    dir = os.path.dirname(path)
    with open(os.path.join(dir, 'logFile.txt'), 'r') as f:
        lines = f.readlines()[2:2 + nrow]

        if align_type == 'cross_day':
            for i in range(nrow):
                idx = lines[i].find(name_label)
                # Get date, e.g. SFP20200826, the 3rd char object to the 11st char is 20220826.
                th = np.where(order == lines[i][idx+3:idx+11])[0]
                if len(th) != 0:
                    index_map_reorder[th[0], :] = index_map[i, :]

        elif align_type == 'cross_session':
            for i in range(nrow):
                idx = lines[i].find(name_label)
                # Get session, e.g. SFP2020082601, the 12nd char object is the number of session.
                th = np.where(order == lines[i][idx+12])[0]
                if len(th) != 0:
                    index_map_reorder[th[0], :] = index_map[i, :]           

    # Delete those cell number less than occur_num, for example, 'occur_num = 5' means that only those cells that are detected in at least 5 days 
    # are kept and others are deleted.
    isfind_map = np.where(index_map_reorder == 0, 0, 1)
    count_num = np.nansum(isfind_map, axis = 0)
    kept_idx = np.where(count_num >= occur_num)[0]

    return index_map_reorder[:, kept_idx]       

def GetMultidayIndexmap(
    mouse: int = None,
    stage: str = None,
    session: int = None,
    i: int = None,
    occu_num: int = None,
    f: pd.DataFrame | None = None
):  
    if f is None:
        f = f_CellReg_day

        if i is None:
            idx = np.where((f['MiceID'] == mouse)&(f['Stage'] == stage)&(f['session'] == session))[0]
    
            if len(idx) == 0:
                print(f"    Mouse {mouse} does not have {stage} session {session} data.")
                return np.array([], dtype = np.int64)
            i = idx[0]
    
        if occu_num is None:
            occu_num = 6
            
        with open(os.path.join(CellregDate, f['dates'][i]), 'rb') as handle:
            order = pickle.load(handle)
    
        return Read_and_Sort_IndexMap(path = f['cellreg_folder'][i], occur_num = occu_num, align_type = 'cross_day', name_label = f['label'][i], order=order)
    else:
        if i is None:
            idx = np.where((f['MiceID'] == mouse)&(f['Stage'] == stage)&(f['session'] == session)&(f['Type'] == 'Real'))[0]
    
            if len(idx) == 0:
                print(f"    Mouse {mouse} does not have {stage} session {session} data.")
                return np.array([], dtype = np.int64)
            i = idx[0]
    
        if occu_num is None:
            occu_num = 6
        
        if f['File Type'][i] == 'PKL':
            with open(f['cellreg_folder'][i], 'rb') as handle:
                index_map = pickle.load(handle)
        else:
            index_map = ReadCellReg(f['cellreg_folder'][i])
            
        print(f['cellreg_folder'])

        cellnum = np.where(index_map == 0, 0, 1)
        idx = np.where(np.nansum(cellnum, axis=0) >= occu_num)[0]
        return index_map[:, idx]
    
def GetSFPSet(
    cellreg_path: str,
    f: pd.DataFrame,
    file_indices: np.ndarray
):
    sfps = []
    for i in file_indices:
        path = os.path.dirname(os.path.dirname(cellreg_path))
        sfp_path = os.path.join(path, f"SFP{int(f['date'][i])}.mat")
    
        if os.path.exists(sfp_path):
            with h5py.File(sfp_path, 'r') as handle:
                sfp = np.array(handle['SFP'])
            sfps.append(sfp)
        else:
            warnings.warn(f"SFP{int(f['date'][i])}.mat does not exist.")
            sfps.append(np.array([]))
        
    return sfps

def print_estimator(Data, **kwargs):
    print(f"  Mean: {np.nanmean(Data)}, STD: {np.nanstd(Data)}, Max: {np.nanmax(Data)}, Min: {np.nanmin(Data)}, Median: {np.nanmedian(Data)}", **kwargs)

def cohen_d(x, y):
    return (np.nanmean(x) - np.nanmean(y))/ np.nanstd(x), (np.nanmean(x) - np.nanmean(y)) / np.nanstd(y)

if __name__ == '__main__':
    idx = np.where((f_CellReg_day['MiceID'] == 10227)&(f_CellReg_day['Stage'] == 'Stage 1+2'))[0][0]
    
    stat_dir = f_CellReg_day['stat_folder'][idx]
    print(ReadSTAT(stat_dir, open_type='scipy'))