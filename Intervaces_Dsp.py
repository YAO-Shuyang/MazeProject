from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *
from mylib.local_path import *
from scipy.stats import poisson, norm, nbinom, gamma, kstest, ks_2samp, anderson, anderson_ksamp 
from scipy.stats import linregress, pearsonr, chisquare, ttest_1samp, ttest_ind
from scipy.stats import ttest_rel, levene, spearmanr
from mylib.stats.ks import poisson_kstest, normal_discrete_kstest, nbinom_kstest
from mylib.behavior.correct_rate import lapwise_behavioral_score
from mylib.stats.indept import indept_field_properties, indept_field_properties_mutual_info, indept_field_evolution_CI
from mylib.stats.ks import lognorm_kstest
from mylib.behavior.correct_rate import calc_behavioral_score_dsp
from mylib.calcium.dsp_ms import classify_lap

# Generate learning curve for cross maze paradigm. Fig0020
def LearningCurve_DSP_Interface(trace: dict, spike_threshold = 30, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'paradigm'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Lap ID', 'Route', 'Lap-wise time cost'])

    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    behav_time = trace['correct_time']
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    navigating_time = (behav_time[end_idx] - behav_time[beg_idx])/1000
    
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx, trace['start_from'])
    routes[routes == 4] = 0
    
    if trace['start_from'] == 'correct':
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    else:
        routes[np.where(routes != 0)[0]] += 3
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
        routes_used[routes_used != 0] = routes_used[routes_used != 0] + 3
    
    laps_id = np.array([i for i in range(1, beg_idx.shape[0] + 1)])
    return laps_id, routes_id, navigating_time

def LearningCurveBehavioralScore_DSP_Interface(trace: dict, variable_names: list):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Route', 'Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'])
    
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], 12)
    behav_time = cp.deepcopy(trace['correct_time'])
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx, trace['start_from'])
    routes[routes == 4] = 0
    
    if trace['start_from'] == 'correct':
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    else:
        routes[np.where(routes != 0)[0]] += 3
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    
    correct_rates, pass_nums, err_nums, pureguess = [], [], [], []
    
    for i in np.unique(routes_used):
        idx = np.where(routes_used == i)[0]
        
        err_num, pass_num, std_err = calc_behavioral_score_dsp(
            route=i,
            behav_nodes=np.concatenate([behav_nodes[beg_idx[j]:end_idx[j]+1] for j in idx]),
            behav_time=np.concatenate([behav_time[beg_idx[j]:end_idx[j]+1] for j in idx])
        )
        
        correct_rates.append(1-err_num/pass_num)
        err_nums.append(err_num)
        pass_nums.append(pass_num)
        pureguess.append(1-std_err)
    
    return np.unique(routes_id), np.array(correct_rates, np.float64), np.array(pass_nums, np.int64), np.array(err_nums, np.int64), np.array(pureguess, np.float64)

def MazeSegmentsPVC_DSP_Interface(trace: dict, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['segments_pvc'])
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segments', 'Mean PVC', 'Compare Groups']
    )
    return (
        np.concatenate([np.arange(trace['segments'].shape[0]) for i in range(4)]),
        trace['segments_pvc'].flatten(),
        np.array(['0-4', '4-5', '5-9', '0-9'] * trace['segments'].shape[0])
    )