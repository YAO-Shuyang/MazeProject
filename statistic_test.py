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
SpatialMapPalette = ['#D4C9A8', '#8E9F85', '#527C5A', '#C3AED6', '#66C7B4', '#A7D8DE']
DSPCorrectTrackPalette = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9']
DSPIncorrectTrackPalette = ['#A9CCE3', '#F2E2C5', '#647D91', '#C06C84']
DSPPalette = ['#A9CCE3', '#A8DADC', '#9C8FBC', '#D9A6A9', '#F2E2C5', '#647D91', '#C06C84']#['#873D38', "#C67E32", "#BFA834", "#7DB27A", "#6376B1", "#764271", "#647D91"]
ModelPalette = (
    sns.color_palette("rocket", 2) + 
    sns.color_palette("Greens", 4) + 
    sns.color_palette("Blues", 4) + 
    sns.color_palette("Purples", 3) +
    sns.color_palette("Reds", 4) +
    sns.color_palette("rainbow", 9)
)

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


# Generate data about some key variables. Generate all data from a behavior paragidm.
def DataFrameEstablish(variable_names: list = [], f:pd.DataFrame = f1, function = None, 
                       file_name:str = 'default', behavior_paradigm:str = 'CrossMaze', 
                       legal_maze_type:list = [0,1,2,3], f_member: list|None = None, 
                       file_idx:np.ndarray|list = None, func_kwgs:dict = {}, 
                       is_behav: bool = False, figdata: str = figdata):
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
    print(f"  Mean: {np.nanmean(Data)}, STD: {np.nanstd(Data)}, Max: {np.nanmax(Data)}, Min: {np.nanmin(Data)}, Median: {np.nanmedian(Data)}, df: {len(Data)-1}", **kwargs)

def cohen_d(x, y):
    return (np.nanmean(x) - np.nanmean(y))/ np.nanstd(x), (np.nanmean(x) - np.nanmean(y)) / np.nanstd(y)

if __name__ == '__main__':
    idx = np.where((f_CellReg_day['MiceID'] == 10227)&(f_CellReg_day['Stage'] == 'Stage 1+2'))[0][0]
    
    stat_dir = f_CellReg_day['stat_folder'][idx]
    print(ReadSTAT(stat_dir, open_type='scipy'))