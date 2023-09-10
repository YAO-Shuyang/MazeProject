from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *

# Fig0011 Session duration
def SessionDuration_Interface(trace: dict, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Duration'])
    KeyWordErrorCheck(trace, __file__, ['correct_time'])
    
    return np.array([trace['correct_time'][-1]/1000/60], dtype=np.float64)

def TotalPathLength_Interface(trace: dict, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Path Length', 'Lap'])
    KeyWordErrorCheck(trace, __file__, ['correct_pos', 'lap_begin_index', 'lap_end_index'])
    
    lap = trace['lap_begin_index'].shape[0]
    dis = np.zeros(lap, np.float64)
    for i in range(lap):
        beg, end = trace['lap_begin_index'][i], trace['lap_end_index'][i]
        dx = np.ediff1d(trace['correct_pos'][beg:end, 0])
        dy = np.ediff1d(trace['correct_pos'][beg:end, 1])
        dis[i] = np.nansum(np.sqrt(dx**2+dy**2))
    
    return dis, np.arange(1, lap+1)   

# Fig0015/14
# FiringRateProcess's interface for data analysis. Fig0015.
def FiringRateProcess_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['peak_rate','mean_rate'])
    KeyWordErrorCheck(trace, __file__, ['is_placecell', 'Spikes'])
    
    trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = spike_threshold)

    Spikes = trace['Spikes']
    # delete those neurons with spike number less than 30
    spikes_num = np.nansum(Spikes, axis = 1)
    # &(trace['is_placecell_isi'] == 1)
    if is_placecell == True:
        idx = np.where((spikes_num >= spike_threshold)&(trace['is_placecell'] == 1))[0]
    else:
        idx = np.where((spikes_num >= spike_threshold))[0]
    
    return cp.deepcopy(trace['peak_rate'][idx]), cp.deepcopy(trace['mean_rate'][idx])

# Fig0016&17
# Generate spatial information map Fig0016
def SpatialInformation_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    KeyWordErrorCheck(trace, __file__, ['SI_all','is_placecell','is_placecell_isi','Spikes'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell','SI'])
    
    is_pc = trace['is_placecell']
    Spikes = trace['Spikes']
    # delete those neurons with spike number less than 30
    spikes_num = np.nansum(Spikes, axis = 1)
    # &(trace['is_placecell_isi'] == 1)
    if is_placecell == True:
        idx = np.where((spikes_num >= spike_threshold)&(is_pc == 1))[0]
    else:
        idx = np.where((spikes_num >= spike_threshold))[0]

    return idx+1, cp.deepcopy(trace['SI_all'][idx])

def NavigateLap(trace):
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    beg_point = StartPoints[int(trace['maze_type'])]
    end_point = EndPoints[int(trace['maze_type'])]
    
    indices = np.where((behav_nodes==beg_point)|(behav_nodes==end_point))[0]
    two_ends = behav_nodes[indices]
    dn = np.ediff1d(two_ends)
    
    idx = np.where(dn > 0)[0]
    return indices[idx], indices[idx+1]
    

# Generate learning curve for cross maze paradigm. Fig0020
def LearningCurve_Interface(trace = {}, spike_threshold = 30, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'paradigm'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Lap ID', 'Lap-wise time cost'])

    behav_time = trace['correct_time']
    beg_idx, end_idx = NavigateLap(trace)
    navigating_time = (behav_time[end_idx] - behav_time[beg_idx])/1000
    
    laps_id = np.array([i for i in range(1, beg_idx.shape[0] + 1)])
    return laps_id, navigating_time

from mylib.behavior.correct_rate import calc_behavioral_score
def LearningCurveBehavioralScore_Interface(trace: dict, variable_names: list):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Correct Rate', 'Pass Number', 'Error Number'])
    
    err_num, pass_num = calc_behavioral_score(trace)
    
    return np.array([1-err_num/pass_num], np.float64), np.array([pass_num], np.float64), np.array([err_num], np.float64)



# Fig0021
# Place Field Number Counts.
def PlaceFieldNumber_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
    KeyWordErrorCheck(trace, __file__, ['place_field_all','is_placecell','is_placecell_isi', 'Spikes','session','maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell','field_number','maze_type'])

    is_pc = trace['is_placecell']
    Spikes = trace['Spikes']
    # delete those neurons with spike number less than 30
    spikes_num = np.nansum(Spikes, axis = 1)
    # &(trace['is_placecell_isi'] == 1)
    if is_placecell == True:
        idx = np.where((spikes_num >= spike_threshold)&(is_pc == 1))[0]
    else:
        idx = np.where((spikes_num >= spike_threshold))[0]

    field_number = np.zeros(len(idx), dtype = np.int64)
    for i in range(len(idx)):
        field_number[i] = len(trace['place_field_all'][idx[i]].keys())

    if trace['session'] == 1:
        maze_type = 'Open Field 1'
    elif trace['session'] == 2:
        if trace['maze_type'] == 1:
            maze_type = 'Maze 1'
        elif trace['maze_type'] == 0:
            maze_type = 'Open Field 2'
    elif trace['session'] == 3:
        maze_type = 'Maze 2'
    elif trace['session'] in [4,5]:
        maze_type = 'Open Field 2'

    return idx+1, field_number, np.repeat(maze_type, len(idx))


# Fig0022 Peak Curve Regression
def PeakDistributionDensity_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False, shuffle_times = 1000):
    KeyWordErrorCheck(trace, __file__, ['old_map_clear','SilentNeuron','occu_time_old'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['MAE', 'RMSE', 'data_type'])

    MAE = np.zeros(1 + shuffle_times, dtype = np.float64)
    RMSE = np.zeros(1 + shuffle_times, dtype = np.float64)
    
    old_map_all = cp.deepcopy(trace['old_map_clear'])
    RMSE[0], MAE[0] = PeakDistributionDensity(old_map_all = old_map_all, SilentNeuron = trace['SilentNeuron'], 
                                              node_not_occu = np.where(np.isnan(trace['occu_time_old']))[0])

    # chance level
    for i in range(1, shuffle_times+1):
        RMSE[i], MAE[i] = PeakDistributionDensityChanceLevel(n_neuron = old_map_all.shape[0], SilentNeuron = trace['SilentNeuron'], 
                                                                   node_not_occu = np.where(np.isnan(trace['occu_time_old']))[0])

    return RMSE, MAE, np.concatenate([['Experiment value'], np.repeat('Chance Level', shuffle_times)])


# Fig0023 Place Cell Percentage
def PlaceCellPercentage_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = False):
    KeyWordErrorCheck(trace, __file__, ['is_placecell'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['percentage', 'place cell num', 'total cell num'])

    return [np.nanmean(trace['is_placecell'])], [np.nansum(trace['is_placecell'])], [trace['n_neuron']]

# Fig0029 Field Centers to Start (distance, unit: m)
from mazepy.behav.graph import Graph
def FieldCentersToStart_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = True, Gs: None = None):
    assert Gs is not None
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Distance To Start', 'Cell'])
    
    dis = []
    cells = []
    G = Gs[trace['maze_type']-1]
    
    idx = np.where(trace['is_placecell'] == 1)[0]
    
    for i in tqdm(idx):
        for k in trace['place_field_all'][i].keys():
            x, y = ((k - 1)%48 + 0.5)/4, ((k - 1)//48 + 0.5)/4
            
            dis.append(G.shortest_distance((x, y), (0.125, 0.125))*8)
            cells.append(i+1)
    
    return np.array(dis, dtype = np.float64), np.array(cells, dtype = np.int64)   


# Fig0030 Decoding Error Figure
def NeuralDecodingResults_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
    KeyWordErrorCheck(trace, __file__, ['maze_type','occu_time'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['RMSE c.l.', 'MAE c.l.', 'abHit c.l.', 'geHit c.l.'])

    # Shuffle 100 times
    shf_t = 1000    
    # initiate shuffle data
    RMSE = np.zeros(shf_t, dtype = np.float64)
    MAE = np.zeros(shf_t, dtype = np.float64)
    abHit = np.zeros(shf_t, dtype = np.float64)
    geHit = np.zeros(shf_t, dtype = np.float64)

    Ms = SmoothMatrix(maze_type = trace['maze_type'], sigma = 3, _range = 7, nx = nx)

    for i in range(shf_t):
        RMSE[i], MAE[i], abHit[i], geHit[i] = DecodingChanceLevel(maze_type = trace['maze_type'], nx = 48, shuffle_frames = 40000, occu_time = trace['occu_time'],
                                                                  Ms = Ms)
    return RMSE, MAE, abHit, geHit


# Fig0033 Peak Velocity
def PeakVelocity_Interface(trace: dict, spike_threshold: int or float = 10, 
                           variable_names: list or None = None, 
                           is_placecell: bool = False):
    KeyWordErrorCheck(trace, __file__, ['behav_nodes', 'behav_speed', 'n_neuron', 'old_map_clear'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell', 'velocity'])

    is_pc = trace['is_placecell']
    Spikes = trace['Spikes']
    # delete those neurons with spike number less than 30
    spikes_num = np.nansum(Spikes, axis = 1)
    # &(trace['is_placecell_isi'] == 1)
    if is_placecell == True:
        idx = np.where((spikes_num >= spike_threshold)&(is_pc == 1))[0]
    else:
        idx = np.where((spikes_num >= spike_threshold))[0]

    velocity = []
    cell_id = []

    for i in idx:
        peak_bin = np.nanargmax(trace['old_map_clear'][i, :]) + 1
        v = peak_velocity(behav_nodes=trace['behav_nodes'], behav_speed=trace['behav_speed'], idx=peak_bin)
        velocity.append(np.nanmean(v))
        cell_id.append(i+1)

    return np.array(cell_id, dtype=np.int64), np.array(velocity, dtype=np.float64)


def Coverage_Interface(trace: dict, spike_threshold: int or float = 10, 
                           variable_names: list or None = None, 
                           is_placecell: bool = False):   
    KeyWordErrorCheck(trace, __file__, ['processed_pos_new'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Coverage', 'bin size', 'date'])

    coverage = np.zeros(5, dtype=np.float64)
    
    coverage[0] = calc_coverage(trace['processed_pos_new'], 12, 12)*100
    coverage[2] = calc_coverage(trace['processed_pos_new'], 19, 19)*100
    coverage[1] = calc_coverage(trace['processed_pos_new'], 24, 24)*100
    coverage[3] = calc_coverage(trace['processed_pos_new'], 36, 36)*100
    coverage[4] = calc_coverage(trace['processed_pos_new'], 48, 48)*100
    
    return coverage, np.array(['8 cm','5 cm','4 cm','2.67 cm','2 cm']), np.repeat(trace['date'], 5)


def Speed_Interface(trace: dict, spike_threshold: int or float = 10, 
                           variable_names: list or None = None, 
                           is_placecell: bool = False):
    if 'smooth_speed' not in trace.keys():
        behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
        trace['smooth_speed'] = uniform_smooth_speed(behav_speed)
        
    KeyWordErrorCheck(trace, __file__, ['smooth_speed', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Speed', 'Maze Bin'])
    
    nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)
    smooth_speed = trace['smooth_speed']
    
    if trace['maze_type'] != 0:
        CP = CorrectPath_maze_1 if trace['maze_type'] == 1 else CorrectPath_maze_2
    else:
        CP = np.arange(1, 145)
    
    mean_speed = np.zeros(CP.shape[0], dtype = np.float64)    
    for i in range(CP.shape[0]):
        idx = np.where(nodes==CP[i])[0]
        mean_speed[i] = np.nanmean(smooth_speed[idx])
        
    
    return mean_speed, np.arange(1, CP.shape[0]+1)

def InterSessionCorrelation_Interface(trace: dict, spike_threshold: int or float = 10, 
                           variable_names: list or None = None, 
                           is_placecell: bool = False):
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        return np.array([]), np.array([]), np.array([])
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
    
    print(np.nanmean(trace['odd_even_corr']), np.nanmean(trace['fir_sec_corr']), 'Maze '+str(trace['maze_type']))    
    KeyWordErrorCheck(trace, __file__, ['fir_sec_corr', 'odd_even_corr', 'is_placecell'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Half-half Correlation', 'Odd-even Correlation', 'Cell Type'])
    
    return trace['fir_sec_corr'], trace['odd_even_corr'], trace['is_placecell']

from scipy.stats import poisson
# Fig0039
def KSTestPoisson_Interface(trace: dict, spike_threshold: int or float = 10, 
                           variable_names: list or None = None, 
                           is_placecell: bool = True):
    
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Statistic', 'PValue'])   
    field_number_pc = field_number_session(trace, is_placecell = True, spike_thre = spike_threshold)
    MAX = np.nanmax(field_number_pc)
    density = plt.hist(field_number_pc, range=(0.5, MAX+0.5), bins = int(MAX), density=True)[0]
    plt.close()
    lam = EqualPoissonFit(np.arange(1,MAX+1), density)
    sta, p = scipy.stats.kstest(field_number_pc, poisson.rvs(lam, size=1000), alternative='two-sided')
    return [sta], [p]

# Fig0040
def FieldNumber_InSessionStability_Interface(trace: dict, spike_threshold: int or float = 10, 
                                             variable_names: list or None = None, 
                                             is_placecell: bool = True):
    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        return np.array([]), np.array([]), np.array([])
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
        
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Field Number', 'In-session OEC', 'In-session FSC'])
    idx = np.where(trace['is_placecell']==1)[0]
    field_number_pc = field_number_session(trace, is_placecell = False)
    
    return field_number_pc[idx], trace['odd_even_corr'][idx], trace['fir_sec_corr'][idx]


# Fig0041
def InFieldCorrelation_Interface(trace: dict, spike_threshold: int or float = 10, 
                                             variable_names: list or None = None, 
                                             is_placecell: bool = True):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Center ID', 'Field Size', 'Center Rate', 
                                                                                'In-field OEC', 'In-field FSC', 'Path Type'])
    
    n = trace['n_neuron']
    
    trace = field_specific_correlation(trace)
    id, size, rate, OEC, FSC, path = [], [], [], [], [], []
    
    CP = Correct_SonGraph1 if trace['maze_type'] == 1 else Correct_SonGraph2
    
    for i in range(n):
        if trace['is_placecell'][i] == 0:
            continue
        ks = trace['place_field_all'][i].keys()
        for k in ks:
            id.append(k)
            size.append(len(trace['place_field_all'][i][k]))
            rate.append(trace['smooth_map_all'][i][k-1])
            OEC.append(trace['in_field_corr'][i][k][0])
            FSC.append(trace['in_field_corr'][i][k][1])
            if k in CP:
                path.append(1)
            else:
                path.append(0)
            
    
    return (np.array(id, dtype = np.int64), 
            np.array(size, dtype = np.int64), 
            np.array(rate, dtype = np.float64), 
            np.array(OEC, dtype = np.float64), 
            np.array(FSC, dtype = np.float64),
            np.array(path, dtype = np.int64))

# Fig0044
def PVCorrelations_Interface(trace: dict, spike_threshold: int or float = 10, 
                                             variable_names: list or None = None, 
                                             is_placecell: bool = True):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Std OEC', 'Mean OEC', 'CP Std OEC', 'CP Mean OEC', 'IP Std OEC', 'IP Mean OEC',
                                                                                'Std FSC', 'Mean FSC', 'CP Std FSC', 'CP Mean FSC', 'IP Std FSC', 'IP Mean FSC'])

    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)
    if 'odd_even_corr' not in trace.keys():
        trace = odd_even_correlation(trace)
    
    if is_placecell:
        idx = np.where(trace['is_placecell'] == 1)[0]
    else:
        idx = np.arange(trace['n_neuron'])
    
    SpatialPVector_OEC = np.zeros(48**2, np.float64) 
    for i in range(48**2):
        SpatialPVector_OEC[i], _ = pearsonr(trace['smooth_map_fir'][idx, i], 
                                            trace['smooth_map_sec'][idx, i])
    
    STD_OEC = np.nanstd(SpatialPVector_OEC)
    MEAN_OEC = np.nanmean(SpatialPVector_OEC)
        
    SpatialPVector_FSC = np.zeros(48**2, np.float64) 
    for i in range(48**2):
        SpatialPVector_FSC[i], _ = pearsonr(trace['smooth_map_odd'][idx, i], 
                                            trace['smooth_map_evn'][idx, i])
        
    STD_FSC = np.nanstd(SpatialPVector_FSC)
    MEAN_FSC = np.nanmean(SpatialPVector_FSC)
        
    if trace['maze_type'] != 0:
        CP = Correct_SonGraph1 if trace['maze_type'] == 1 else Correct_SonGraph2
        IP = Incorrect_SonGraph1 if trace['maze_type'] == 1 else Incorrect_SonGraph2
        
        STD_OEC_CP = np.nanstd(SpatialPVector_OEC[CP-1])
        MEAN_OEC_CP = np.nanmean(SpatialPVector_OEC[CP-1])
        STD_OEC_IP = np.nanstd(SpatialPVector_OEC[IP-1])
        MEAN_OEC_IP = np.nanmean(SpatialPVector_OEC[IP-1])
        
        STD_FSC_CP = np.nanstd(SpatialPVector_FSC[CP-1])
        MEAN_FSC_CP = np.nanmean(SpatialPVector_FSC[CP-1])
        STD_FSC_IP = np.nanstd(SpatialPVector_FSC[IP-1])
        MEAN_FSC_IP = np.nanmean(SpatialPVector_FSC[IP-1])
        return (np.array([STD_OEC]), np.array([MEAN_OEC]),
                np.array([STD_OEC_CP]), np.array([MEAN_OEC_CP]), 
                np.array([STD_OEC_IP]), np.array([MEAN_OEC_IP]), 
                np.array([STD_FSC]), np.array([MEAN_FSC]), 
                np.array([STD_FSC_CP]), np.array([MEAN_FSC_CP]), 
                np.array([STD_FSC_IP]), np.array([MEAN_FSC_IP]))
        
    else:
        return (np.array([STD_OEC]), np.array([MEAN_OEC]),
                np.array([np.nan]), np.array([np.nan]), 
                np.array([np.nan]), np.array([np.nan]), 
                np.array([STD_FSC]), np.array([MEAN_FSC]), 
                np.array([np.nan]), np.array([np.nan]), 
                np.array([np.nan]), np.array([np.nan]))


from mylib.calcium.field_criteria import GetPlaceField
#Fig0048 Place Field Criteria
def PlaceFieldNumberWithCriteria_Interface(
    trace: dict, 
    spike_threshold: int or float = 10,
    variable_names: list or None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Field Number', 'criteria', 'x'])

    smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
    pc_idx = np.where(trace['is_placecell'] == 1)[0]

    field_numberA = np.zeros((34, pc_idx.shape[0]), dtype=np.int64)
    x = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 
                  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 
                  0.14, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 
                  0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 
                  0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)
    x1 = np.repeat(x, pc_idx.shape[0])
    criteria1 = np.repeat('A', pc_idx.shape[0]*34)

    for i, j in enumerate(x):
        for k, n in enumerate(pc_idx):
            fields = GetPlaceField(trace['maze_type'], smooth_map=smooth_map_all[n, :], thre_type = 1, parameter=j)
            field_numberA[i, k] = len(fields.keys())

    field_numberB = np.zeros((27, pc_idx.shape[0]), dtype=np.int64)
    x = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0])
    x2 = np.repeat(x, pc_idx.shape[0])
    criteria2 = np.repeat('B', pc_idx.shape[0]*27)

    for i, j in enumerate(x):
        for k, n in enumerate(pc_idx):
            fields = GetPlaceField(trace['maze_type'], smooth_map=smooth_map_all[n, :], thre_type = 2, parameter=j)
            field_numberB[i, k] = len(fields.keys())

    return np.concatenate([field_numberA.flatten(), field_numberB.flatten()]), np.concatenate([criteria1, criteria2]), np.concatenate([x1, x2])
    
# Fig0048-2 First-second half stability
def PlaceFieldFSCStabilityWithCriteria_Interface(
    trace: dict, 
    spike_threshold: int or float = 10,
    variable_names: list or None = None, 
    is_placecell: bool = True
):

    if 'laps' not in trace.keys():
        trace = CrossLapsCorrelation(trace, behavior_paradigm = trace['paradigm'])
        trace = OldMapSplit(trace)
    if trace['laps'] == 1:
        return np.array([]), np.array([]), np.array([])
    if 'fir_sec_corr' not in trace.keys():
        trace = half_half_correlation(trace)

    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['FSC Stability', 'criteria', 'x'])

    smooth_map_all = cp.deepcopy(trace['smooth_map_all'])
    smooth_map_fir = cp.deepcopy(trace['smooth_map_fir'])
    smooth_map_sec = cp.deepcopy(trace['smooth_map_sec'])
    pc_idx = np.where(trace['is_placecell'] == 1)[0]

    FSCStabilityA = []
    x = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 
                  0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 
                  0.14, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 
                  0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 
                  0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float64)
    x1 = []
    criteria1 = []
    correct_path = Correct_SonGraph1 if trace['maze_type'] == 1 else Correct_SonGraph2

    for i, j in enumerate(x):
        for n in pc_idx:
            fields = GetPlaceField(trace['maze_type'], smooth_map=smooth_map_all[n, :], thre_type = 1, parameter=j)
            for k in fields.keys():
                if k not in correct_path:
                    continue

                if len(fields[k]) <= 1:
                    corr = np.nan
                else:
                    corr, _ = pearsonr(smooth_map_fir[n, np.array(fields[k])-1], smooth_map_sec[n, np.array(fields[k])-1])
                FSCStabilityA.append(corr)
                x1.append(j)
                criteria1.append('A')

    FSCStabilityB = []
    x = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0])
    x2 = []
    criteria2 = []

    for i, j in enumerate(x):
        for n in pc_idx:
            fields = GetPlaceField(trace['maze_type'], smooth_map=smooth_map_all[n, :], thre_type = 2, parameter=j)
            for k in fields.keys():
                if k not in correct_path:
                    continue

                father_bin = np.unique(S2F[np.array(fields[k])-1])
                area = np.concatenate([Father2SonGraph[b] for b in father_bin])

                if len(fields[k]) <= 1:
                    corr = np.nan
                else:
                    corr, _ = pearsonr(smooth_map_fir[n, area-1], smooth_map_sec[n, area-1])

                FSCStabilityB.append(corr)
                x2.append(j)
                criteria2.append('B')

    return np.array(FSCStabilityA+FSCStabilityB, np.float64), np.array(criteria1+criteria2), np.array(x1+x2, np.float64)
    