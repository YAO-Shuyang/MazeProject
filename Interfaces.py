from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *

# Fig0015/14
# FiringRateProcess's interface for data analysis. Fig0015.
def FiringRateProcess_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
    trace = FiringRateProcess(trace, map_type = 'old', spike_threshold = spike_threshold)
    Spikes = trace['Spikes']
    # delete those neurons with spike number less than 30
    spikes_num = np.nansum(Spikes, axis = 1)
    # &(trace['is_placecell_isi'] == 1)
    if is_placecell == True:
        idx = np.where((spikes_num >= spike_threshold)&(trace['is_placecell'] == 1))[0]
    else:
        idx = np.where((spikes_num >= spike_threshold))[0]

    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['peak_rate','mean_rate','peak_rate_on_path','mean_rate_on_path','path_type'])
    KeyWordErrorCheck(trace, __file__, ['peak_rate','main_field_all','peak_rate', 'mean_rate','is_placecell','Spikes'])

    # generate field pattern
    main_field_all = trace['main_field_all'][idx]
    path_type = []
    for n in range(main_field_all.shape[0]):
        if main_field_all[n] == 0:
            path_type.append('Correct path')
        elif main_field_all[n] == 1:
            path_type.append('Incorrect path')

    if trace['maze_type'] in [1,2]:
        return cp.deepcopy(trace['peak_rate'][idx]), cp.deepcopy(trace['mean_rate'][idx]), cp.deepcopy(trace['peak_rate_on_path'][idx]), cp.deepcopy(trace['mean_rate_on_path'][idx]), np.array(path_type)
    else:
        return cp.deepcopy(trace['peak_rate'][idx]), cp.deepcopy(trace['mean_rate'][idx]), np.repeat(np.nan, len(idx)), np.repeat(np.nan, len(idx)), np.repeat(np.nan, len(idx))

# Fig0016&17
# Generate spatial information map Fig0016
def SpatialInformation_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
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

# Generate learning curve for cross maze paradigm. Fig0020
def LearningCurve_Interface(trace = {}, spike_threshold = 30, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['laps','lap_begin_index','lap_end_index', 'correct_time'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['laps_id','explore_time'])

    correct_time = trace['correct_time']
    beg_idx, end_idx = CrossMazeLapSplit(trace)
    laps = len(beg_idx)

    explore_time = np.zeros(laps, dtype = np.float64)
    for i in range(laps):
        explore_time[i] = (correct_time[end_idx[i]] - correct_time[beg_idx[i]]) / 1000
    
    laps_id = np.array(['Lap '+str(i) for i in range(1, laps + 1)])
    return laps_id, explore_time


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
def PlaceCellPercentage_Interface(trace = {}, spike_threshold = 30, variable_names = None, is_placecell = False):
    KeyWordErrorCheck(trace, __file__, ['is_placecell'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['percentage', 'place cell'])

    return [np.nanmean(trace['is_placecell'])], [np.nansum(trace['is_placecell'])]


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
def PeakVelocity_Interface(trace: dict, spike_threshold: int or float = 30, 
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


def Coverage_Interface(trace: dict, spike_threshold: int or float = 30, 
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


def Speed_Interface(trace: dict, spike_threshold: int or float = 30, 
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

if __name__ == '__main__':
    with open(r'G:\YSY\Cross_maze\11095\20220830\session 3\trace.pkl', 'rb') as handle:
        trace = pickle.load(handle)
    
    print(trace.keys())
    print(trace['behav_speed'].shape)
    print(trace['behav_nodes'].shape)
