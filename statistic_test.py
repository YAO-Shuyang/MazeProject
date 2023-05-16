from mylib.preprocessing_behav import *
from mylib.preprocessing_ms import *
from mylib.Interfaces import *
import seaborn as sns
import scipy.io
from mylib.Interfaces import FiringRateProcess_Interface, SpatialInformation_Interface, LearningCurve_Interface, PlaceFieldNumber_Interface, PlaceCellPercentage_Interface, NeuralDecodingResults_Interface, PeakDistributionDensity_Interface

figpath = 'E:\Data\FinalResults'
figdata = 'E:\Data\FigData'

CM_path = 'E:\Data\Cross_maze'
CM_path2 = 'E:\Data\Cross_maze2'
#SM_path = 'E:\Data\Simple_maze'
f1 = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'calcium')
f1_behav = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'behavior')
#f2 = pd.read_excel(os.path.join(SM_path,'simple_maze_paradigm.xlsx'), sheet_name = 'calcium')

#f_CellReg = pd.read_excel(os.path.join(CM_path, 'cell_reg_path.xlsx'), sheet_name = 'CellRegPath')

cellReg_95_maze1 = r'E:\Data\Cross_maze\11095\Maze1-footprint\Cell_reg\cellRegistered.mat'
cellReg_95_maze2 = r'E:\Data\Cross_maze\11095\Maze2-footprint\Cell_reg\cellRegistered.mat'
order_95_maze1 = np.array([], dtype = np.int64)


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
def TraceFileSet(idx:np.ndarray = None, file:pd.DataFrame = f1, Behavior_Paradigm:str = 'CrossMaze', tp: str = r"G:\YSY\Cross_maze"):
    '''
    Author: YAO Shuyang
    Date: Jan 26th, 2023
    Note: To combine trace into a list for the convinence to call.

    Input:
    - file: <class 'pandas.DataFrame'>, the guide file that saves basic information of each session.
    - idx: <class 'numpy.ndarray'>, the indexes (of row) of sessions that you want to choose to combine a list.
    - Behavior_Paradigm: str, the behavior paradigm of file. Only 'Cross Maze', 'Simple Maze', 'Reverse Maze' are valid value.

    Output:
    - A list, contains several dicts (trace of each session) and the length of this list is equal to the length of idx.
    '''
    assert idx is not None and file is not None  #Error! Need an input idx list and a DataFrame!
    ValueErrorCheck(Behavior_Paradigm, ['CrossMaze', 'SimpleMaze', 'ReverseMaze'])

    if Behavior_Paradigm == 'CrossMaze':
        KeyWordErrorCheck(file, ['MiceID', 'date', 'session'])
        trace_set = []
        print("Read trace file:")
        for i in tqdm(idx):
            loc = os.path.join(tp, str(int(file['MiceID'][i])), str(int(file['date'][i])), 'session '+str(int(file['session'][i])), 'trace.pkl')
            if os.path.exists(loc) == False:
                print(loc, ' is not exist! Abort function TraceFileSet().')
                return []
            else:
                with open(loc, 'rb') as handle:
                    trace = pickle.load(handle)
            trace_set.append(trace)
        return trace_set

    elif Behavior_Paradigm == 'SimpleMaze':
        KeyWordErrorCheck(file, ['MiceID', 'date'])
        trace_set = []
        print("Read trace file:")
        for i in tqdm(idx):
            loc = os.path.join(tp, str(int(file['MiceID'][i])), str(int(file['date'][i])), 'trace.pkl')
            if os.path.exists(loc) == False:
                print(loc, ' is not exist! Abort function TraceFileSet().')
                return []
            else:
                with open(loc, 'rb') as handle:
                    trace = pickle.load(handle)
            trace_set.append(trace)
        return trace_set
    
    elif Behavior_Paradigm == 'ReverseMaze':
        KeyWordErrorCheck(file, ['MiceID', 'date'])
        trace_set = []
        print("Read trace file:")
        for i in tqdm(idx):
            loc = os.path.join(tp, str(int(file['MiceID'][i])), str(int(file['date'][i])), 'trace.pkl')
            if os.path.exists(loc) == False:
                print(loc, ' is not exist! Abort function TraceFileSet().')
                return []
            else:
                with open(loc, 'rb') as handle:
                    trace = pickle.load(handle)
            trace_set.append(trace)
        return trace_set

# Index matrix is generated for the convience to plot sample cells.
def Generate_IndexMatrix(file, dateset = ['20220830'], beg_idx = None, end_idx = None):
    if beg_idx is None or beg_idx < 0:
        beg_idx = 0
    if end_idx is None or end_idx > len(file):
        end_idx = len(file)
    IndexMatrix = np.zeros((end_idx - beg_idx, len(dateset)), dtype = np.int64)
    for d in range(len(dateset)):
        IndexMatrix[:,d] = np.array(file[dateset[d]][beg_idx:end_idx], dtype = np.int64)
    return IndexMatrix

def Read_and_Sort_IndexMap(path:str = None, occur_num:int = 6, align_type:str = 'cross_day', name_label:str = 'SFP2022'):
    '''
    Author: YAO Shuyang
    Date: Jan 25th, 2023 (Modified)
    Note: This function is written to
    1. Read index_map from cellRegistered.mat.
    2. Sort them with certain order.
    3. Select cell sequences that satisfy certain requirement. The requirement is the number of cells in the cell sequence. If the number >= occur_numm, it is satisfied.

    Input:
    - path: str, the directory of cellRegistered.mat file
    - occur_num: int. The threshold to select cell sequences that satisfy this threshold. The default value for 'cross_day' file is 6 and default value for 'cross_session' is 4.
    - align_type: str. Determines whether sessions that are aligned are recorded in different days (cross_day) or different sessions in one day (cross_session). Only 'cross_day' and 'cross_session' are valid value.
    - name_label: str, the cell_reg name label in logFile.txt

    Output:
    - <class 'numpy.ndarray'>
    '''
    ValueErrorCheck(align_type, ['cross_day','cross_session']) # InputContentError! Only 'cross_day' and 'cross_session' are valid value!
    order = np.array(['20220820', '20220822', '20220824', '20220826', '20220828', '20220830']) if align_type == 'cross_day' else np.array(['1','2','3','4'])

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
                       legal_maze_type:list = [0,1,2], f_member: list|None = None, 
                       file_idx:np.ndarray|list = None, func_kwgs:dict = {}, 
                       is_behav: bool = False):
    '''
    Author: YAO Shuyang
    Date: Jan 25th, 2023 (Modified)
    Note: This function is to calculate some variables and concatenate data cross a long training period together to form a DataFrame, and this data frame can be used to plot figures.

    Input: 
    - variable_names: list, represent the variables that will be saved in data.
    - f: pandas.DataFrame, files that save information of sessions.
    - funciton: processing function. the function must have args 'trace', 'spike_threshold'.
    - file_name: str, file_name of saved pkl file.
    - behavior_paradigm: str, 'CrossMaze','SimpleMaze','ReverseMaze' are 3 valid data.
    - f_member: str or None. If it gets an input (str), it delivers the member of list f_member to f, and get correlated value from f.

    Output:
    - A dict
    '''
    ValueErrorCheck(behavior_paradigm, ['CrossMaze', 'ReverseMaze', 'SimpleMaze', 'decoding'])

    # Initiate data dic
    data = {'MiceID':np.array([]), 'Training Day':np.array([]), 'Maze Type':np.array([])}
    
    # Initiate additive member:
    if f_member is not None:
        for m in f_member:
            data[m] = np.array([])

    for c in variable_names:
        data[c] = np.array([], np.float64)

    if file_idx is None:
        file_idx = np.arange(len(f))

    for i in tqdm(file_idx):
        try: 
            KeyWordErrorCheck(f, __file__, keys = ['Trace File', 'Trace Behav File'])
            if is_behav:
                p = f['Trace Behav File'][i]
            else:
                p = f['Trace File'][i]
        except:
            KeyWordErrorCheck(f, __file__, keys = ['Data File'])
            p = f['Data File'][i]

        if os.path.exists(p):
            with open(p, 'rb') as handle:
                trace = pickle.load(handle)
        else:
            print(p,' is not exist!')
            continue

        # delete abnormal sessions.
        if str(trace['date']) == '20220817' and trace['maze_type'] == 1:
            continue
        
        # if maze_type is not we want, continue
        if trace['maze_type'] not in legal_maze_type:
            continue

        # Running funcitons to get variables we want to analysis.
        results = function(trace, spike_threshold = 30, variable_names = variable_names, **func_kwgs)
        # length of each variables in dictionary 'data' must be the same with others.
        if len(variable_names) == 1:
            length  = len(results)
            results = [results]
        else:
            length = len(results[0])
        # Generating data.
        mazes = 'Maze '+str(trace['maze_type']) if trace['maze_type'] in [1,2] else 'Open Field'
        data['MiceID'] = np.concatenate([data['MiceID'], np.repeat(str(trace['MiceID']), length)])
        data['Maze Type'] = np.concatenate([data['Maze Type'], np.repeat(mazes, length)])
        data['Training Day'] = np.concatenate([data['Training Day'], np.repeat(f['training_day'][i], length)])
        for c in range(len(variable_names)):
            data[variable_names[c]] = np.concatenate([data[variable_names[c]], results[c]])
        
        # Add additive values
        if f_member is not None:
            for m in f_member:
                data[m] = np.concatenate([data[m], np.repeat(f[m][i], length)])
        
    print(np.shape(data['MiceID']))

    d = pd.DataFrame(data)
    try:
        d.to_excel(os.path.join(figdata, file_name+'.xlsx'), sheet_name = 'data', index = False)
    except:
        a = 1

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


def ExclusivePeakRate_Interface(trace:dict = {}, spike_threshold:int = 30, variable_names:str = None, f_CellReg:pd.DataFrame = None, f_trace:pd.DataFrame = f1):
    '''
    Parameters
    ----------
    trace: dict
        The trace file that saves elements.
    spike_threshold: int, default value is 30
        If total number of spikes of a cell is less than the threshold, this cell is not included in statistic.
    variable_names: str
        The variable's name.
    f_CellReg: pd.DataFrame

    Note
    ----
    To return peak rate exclude maintained place cells.

    Author
    ------
    YAO Shuyang
    
    Date
    ----
    Jan 30th, 2023
    '''
    trace = FiringRateProcess(trace, map_type = 'old', spike_threshold = spike_threshold)
    KeyWordErrorCheck(trace, __file__, ['is_placecell', 'peak_rate','MiceID','date'])
    VariablesInputErrorCheck(variable_names, ['Peak Rate', 'Cell Type'])
    
    MiceSet = {'11095':11095, '11092':11092}
    DateSet = {'20220808':20220808, '20220809':20220809, '20220810':20220810, '20220811':20220811, '20220812':20220812, '20220813':20220813, '20220815':20220815, 
               '20220817':20220817, '20220820':20220820, '20220822':20220822, '20220824':20220824, '20220826':20220826, '20220828':20220828, '20220830':20220830}

    if trace['MiceID'] not in MiceSet.keys() or trace['date'] not in DateSet.keys() or (trace['maze_type'] == 1 and DateSet[trace['date']] < 20220813):
        return np.array([]), np.array([])

    if trace['maze_type'] == 0:
        return trace['peak_rate'][np.where(trace['is_placecell'] == 1)[0]], np.repeat("Original PC", trace['peak_rate'][np.where(trace['is_placecell'] == 1)[0]].shape[0])
    
    try:
        i = np.where((f_CellReg['MiceID'] == MiceSet[trace['MiceID']])&(f_CellReg['date'] == DateSet[trace['date']]))[0][0]
        print(f_CellReg['Cell Reg Path'][i])
    except:
        print(i)

    # Get Index MAP
    if os.path.exists(f_CellReg['Cell Reg Path'][i]):
        index_map = Read_and_Sort_IndexMap(path = f_CellReg['Cell Reg Path'][i], occur_num = 2, align_type = 'cross_session')
    else:
        print(f_CellReg['Cell Reg Path'][i], 'is not exist!')
        return np.array([]), np.array([])
    
    # Select Cell Pairs that Both exist in index_map in map1 and map2
    is_cell_detected = np.where(index_map == 0, 0, 1)
    cellpair = np.where(np.nansum(is_cell_detected[[0,trace['maze_type']],:], axis = 0) == 2)[0]
    index_map = index_map[:, cellpair]
    index_map = index_map.astype(np.int64)

    idx = np.where((f_trace['MiceID'] == f_CellReg['MiceID'][i])&(f_trace['date'] == f_CellReg['date'][i]))[0]
    trace_set = TraceFileSet(idx = idx, file = f_trace, Behavior_Paradigm = 'Cross Maze')
    m = trace['maze_type'] # map

    # non-place cell index
    spikes_num = np.nansum(trace['Spikes'], axis = 1)
    pc_idx = np.where((spikes_num >= spike_threshold)&(trace['is_placecell'] == 1))[0]

    mpc_idx_idx = np.where((trace_set[0]['is_placecell'][index_map[0, :]-1] == 1)&(trace_set[m]['is_placecell'][index_map[m, :]-1] == 1))[0]
    mpc_idx = index_map[m, mpc_idx_idx]-1

    # Get the intersection:
    mpc_idx = np.intersect1d(pc_idx, mpc_idx)
    other_pc_idx = np.setdiff1d(pc_idx, mpc_idx)

    return np.concatenate([trace['peak_rate'][mpc_idx], trace['peak_rate'][other_pc_idx]]), np.repeat(['Maintained PC', 'Newly Recuited PC'], [mpc_idx.shape[0], other_pc_idx.shape[0]])

#if __name__ == '__main__':
    #data = DataFrameEstablish(variable_names = ['peak_rate','mean_rate','cor_mean_rate','inc_mean_rate','cor_peak_rate','inc_peak_rate'], 
    #                           f = f1, function = FiringRateProcess_Interface)


if __name__ == '__main__':
    print(1)