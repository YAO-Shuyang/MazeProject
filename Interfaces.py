from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *
from mylib.local_path import *
from scipy.stats import poisson, norm, nbinom, gamma, kstest, ks_2samp, anderson, anderson_ksamp 
from scipy.stats import linregress, pearsonr, chisquare, ttest_1samp, ttest_ind
from scipy.stats import ttest_rel, levene, spearmanr
from mylib.stats.kstest import poisson_kstest, normal_kstest, nbinom_kstest
from mylib.behavior.correct_rate import lapwise_behavioral_score
from mylib.stats.indeptest import indept_field_properties, indept_field_properties_mutual_info

# Fig0007 Cell Number
def CellNumber_Interface(trace: dict, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell Number'])
    
    return np.array([trace['n_neuron']], dtype=np.int64)

# Fig0011 Session duration
def SessionDuration_Interface(trace: dict, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Duration'])
    KeyWordErrorCheck(trace, __file__, ['correct_time'])
    
    return np.array([trace['correct_time'][-1]/1000/60], dtype=np.float64)

def TotalPathLength_Interface(trace: dict, spike_threshold = 30, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Path Length'])
    
    if trace['maze_type'] == 0:
        dx = np.ediff1d(trace['correct_pos'][:, 0])
        dy = np.ediff1d(trace['correct_pos'][:, 1])
        dis = np.nansum(np.sqrt(dx**2+dy**2))/10
        return np.array([dis], dtype=np.float64)
        
    beg, end = LapSplit(trace, trace['paradigm'])
    
    dis = np.zeros(beg.shape[0], np.float64)
    for i in range(beg.shape[0]):
        dx = np.ediff1d(trace['correct_pos'][beg[i]:end[i], 0])
        dy = np.ediff1d(trace['correct_pos'][beg[i]:end[i], 1])
        dis[i] = np.nansum(np.sqrt(dx**2+dy**2))/10
    
    
    return np.array([np.sum(dis)], dtype=np.float64) 

# Fig0015/14
# FiringRateProcess's interface for data analysis. Fig0015.
def FiringRateProcess_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['peak_rate','mean_rate'])
    KeyWordErrorCheck(trace, __file__, ['is_placecell', 'Spikes'])
    
    #trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = spike_threshold)
    if trace['maze_type'] == 0:
        occu_time = cp.deepcopy(trace['occu_time_spf'])
        Spikes = trace['Spikes']
        rate_map_all = cp.deepcopy(trace['rate_map_all'])
        is_placecell = trace['is_placecell']
    else:
        occu_time = cp.deepcopy(trace['LA']['occu_time_spf'])
        Spikes = trace['LA']['Spikes']
        rate_map_all = cp.deepcopy(trace['LA']['rate_map_all'])
        is_placecell = trace['LA']['is_placecell']
    
    peak_rate = np.nanmax(rate_map_all[np.where(is_placecell == 1)[0], :], axis = 1)
    mean_rate = np.nansum(Spikes[np.where(is_placecell == 1)[0], :], axis = 1) / np.nansum(occu_time) * 1000
    
    return np.array([np.mean(peak_rate)]), np.array([np.mean(mean_rate)])

# Fig0015/14
# FiringRateProcess's interface for data analysis. Fig0015.
def FieldPeakRateStatistic_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['peak_rate'])
    KeyWordErrorCheck(trace, __file__, ['is_placecell', 'Spikes'])
    
    #trace = FiringRateProcess(trace, map_type = 'smooth', spike_threshold = spike_threshold)
    peak_rate = []
    for i in range(trace['n_neuron']):
        if trace['is_placecell'][i] == 1:
            for k in trace['place_field_all'][i].keys():
                peak_rate.append(trace['smooth_map_all'][i, k-1])
    
    return np.array(peak_rate, np.float64)

# Fig0016&17
# Generate spatial information map Fig0016
def SpatialInformation_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    KeyWordErrorCheck(trace, __file__, ['SI_all','is_placecell','is_placecell_isi','Spikes'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['SI'])
    
    if trace['maze_type'] == 0:
        return np.array([np.mean(trace['SI_all'][np.where(trace['is_placecell'] == 1)[0]])])
    else:
        return np.array([np.mean(trace['LA']['SI_all'][np.where(trace['LA']['is_placecell'] == 1)[0]])])
    

# Generate learning curve for cross maze paradigm. Fig0020
def LearningCurve_Interface(trace: dict, spike_threshold = 30, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'paradigm'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Lap ID', 'Lap-wise time cost'])

    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    behav_time = trace['correct_time']
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    navigating_time = (behav_time[end_idx] - behav_time[beg_idx])/1000
    
    laps_id = np.array([i for i in range(1, beg_idx.shape[0] + 1)])
    return laps_id, navigating_time

# Fig0065
def LearningCurve_Reverse_Interface(trace: dict, spike_threshold = 10, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'paradigm'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Lap ID', 'Direction', 'Lap-wise time cost'])

    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    behav_time = trace['correct_time']
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    navigating_time = (behav_time[end_idx] - behav_time[beg_idx])/1000
    
    direction = np.zeros(beg_idx.shape[0])
    laps_id = np.zeros(beg_idx.shape[0])
    cis, trs = 0, 0
    for i in range(beg_idx.shape[0]):
        if trace['correct_nodes'][end_idx[i]] < trace['correct_nodes'][beg_idx[i]]:
            direction[i] = 0
            trs += 1
            laps_id[i] = trs
        else:
            direction[i] = 1
            cis += 1
            laps_id[i] = cis
        
    return laps_id, direction, navigating_time

from mylib.behavior.correct_rate import calc_behavioral_score
def LearningCurveBehavioralScore_Interface(trace: dict, variable_names: list):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'])
    
    err_num, pass_num, std_err = calc_behavioral_score(trace)
    
    return np.array([1-err_num/pass_num], np.float64), np.array([pass_num], np.float64), np.array([err_num], np.float64), np.array([1-std_err], np.float64)

def model_test(data, compared_data, a, y):
    # Kolmogorov-Smirnov Test for Poisson Distribution
    ks_sta, pvalue = scipy.stats.kstest(data, compared_data, alternative='two-sided')
    print("KS Test:", ks_sta, pvalue)
    ks_sta2, pvalue2 = scipy.stats.ks_2samp(data, compared_data, alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2, pvalue2)
    chista, chip = scipy.stats.chisquare(a, y)
    print("Chi-Square Test:", chista, chip)
    adsta, c, adp = scipy.stats.anderson_ksamp([data, compared_data])
    print("Anderson-Darling Test 2 sample:", adsta, c, adp, end='\n\n')
    
    is_rejected = pvalue < 0.05 and pvalue2 < 0.05
    
    return ks_sta, pvalue, ks_sta2, pvalue2, chista, chip, adsta, adp, is_rejected

def fit_model(field_number: np.ndarray, a, dist: str = 'poisson'):
    
    xmax = np.max(field_number)
    # Fit Poisson Distribution
    prob = a / np.nansum(a)
        
    if dist == 'poisson':
        lam = EqualPoissonFit(np.arange(1, xmax+1), prob)
        y = EqualPoisson(np.arange(1, xmax+1), l = lam)
        y = y / np.sum(y) * np.sum(a)
        abs_res = np.mean(np.abs(a-y))
        return lam, a, y, abs_res
    elif dist == 'normal':
        mu, sigma = NormalFit(np.arange(1, xmax+1), prob)
        y = Normal(np.arange(1, xmax+1), mu, sigma)
        y = y / np.sum(y) * np.sum(a)
        abs_res = np.mean(np.abs(a-y))
        return mu, sigma, a, y, abs_res
    elif dist == 'nbinom':
        r, p = NegativeBinomialFit(np.arange(1, xmax+1), prob)
        y = NegativeBinomial(np.arange(1, xmax+1), r, p)
        y = y / np.sum(y) * np.sum(a)
        abs_res = np.mean(np.abs(a-y))
        return r, p, a, y, abs_res
    else:
        raise ValueError(f"dist {dist} is not valid! Only poisson, normal, nbinom are supported.")


# Fig0020 Influence of Field 
def FieldDistributionStatistics_DiverseCriteria_Interface(
    trace: dict,
    spike_threshold = 10,
    variable_names = None,
    is_placecell = True,
    i: int = None
):
    VariablesInputErrorCheck(input_variable = variable_names, 
                             check_variable = ['lambda', 'Poisson-residual', 'Poisson-ChiS Statistics', 'Poisson-ChiS P-Value',
                                                'Poisson-KS Statistics', 'Poisson-KS P-Value', 
                                                'Poisson-KS 2sample Statistics', 'Poisson-KS 2sample P-Value',
                                                'Poisson-AD 2sample Statistics', 'Poisson-AD 2sample P-Value', 
                                                'Poisson-Is Rejected',
                                                
                                                'Mean', 'Sigma', 'Normal-residual', 'Normal-ChiS Statistics', 'Normal-ChiS P-Value',
                                                'Normal-KS Statistics', 'Normal-KS P-Value', 
                                                'Normal-KS 2sample Statistics', 'Normal-KS 2sample P-Value',
                                                'Normal-AD 2sample Statistics', 'Normal-AD 2sample P-Value', 
                                                'Normal-Is Rejected',
                                                
                                                'r', 'p', 'NBinom-residual', 'NBinom-ChiS Statistics', 'NBinom-ChiS P-Value',
                                                'NBinom-KS Statistics', 'NBinom-KS P-Value', 
                                                'NBinom-KS 2sample Statistics', 'NBinom-KS 2sample P-Value',
                                                'NBinom-AD 2sample Statistics', 'NBinom-AD 2sample P-Value', 
                                                'NBinom-Is Rejected',
                                                
                                                'Events Threshold', 'Field Number'])
    saveloc = os.path.join(trace['p'], "Field_Criteria.pkl")
    hist_saveloc = os.path.join(figdata, r"E:\Data\FinalResults\0026 - Draw Place Field Distribution Huge Panel\Field_Criteria")
    print(saveloc)
    if os.path.exists(os.path.dirname(saveloc)) == False:
        mkdir(os.path.dirname(saveloc))
        
    if os.path.exists(hist_saveloc) == False:
        mkdir(hist_saveloc)
    
    lams = np.zeros(16, dtype = np.float64)
    means, sigmas = np.zeros(16, dtype = np.float64), np.zeros(16, dtype = np.float64)
    rs, ps = np.zeros(16, dtype = np.float64), np.zeros(16, dtype = np.float64)
    
    residuals = np.zeros((3, 16), dtype = np.float64)
    chi_s, chi_p = np.zeros((3, 16), dtype = np.float64), np.zeros((3, 16), dtype = np.float64)
    ks_s, ks_p = np.zeros((3, 16), dtype = np.float64), np.zeros((3, 16), dtype = np.float64)
    ks2_s, ks2_p = np.zeros((3, 16), dtype = np.float64), np.zeros((3, 16), dtype = np.float64)
    ad2_s, ad2_p = np.zeros((3, 16), dtype = np.float64), np.zeros((3, 16), dtype = np.float64)
    is_rejected = np.zeros((3, 16), dtype = np.float64)
    
    thres = np.zeros((3, 16), dtype = np.float64)
    field_nums = np.zeros((3, 16), dtype = np.float64)
    
    data = {}
    fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(4*4,3*4))    
    for i in range(1, 17):
        try:
            data[i] = place_field(trace['LA'], 2, 0.4, events_num_crit=i)
        except:
            data[i] = place_field(trace, 2, 0.4, events_num_crit=i)
            
        idx = np.where(trace['is_placecell'] == 1)[0] if trace['maze_type'] == 0 else np.where(trace['LA']['is_placecell'] == 1)[0]
        
        field_number = np.zeros(len(idx), dtype = np.int64)
        for j, k in enumerate(idx):
            field_number[j] = len(data[i][j].keys())

        xmax = int(np.max(field_number))
        ax = Clear_Axes(axes[int(i-1)//4, (i-1)%4], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        a = ax.hist(field_number, bins=xmax, range=(0.5, xmax+0.5), color='gray', rwidth=0.8)[0]
        ax.set_xlim((0.5, 25+0.5))
            
        lams[i-1], a, y, residuals[0, i-1] = fit_model(field_number, a, dist='poisson')
        ax.plot(np.arange(1, xmax+1), y, linewidth=0.5, label = 'Poisson')
        ks_s[0, i-1], ks_p[0, i-1], ks2_s[0, i-1], ks2_p[0, i-1], chi_s[0, i-1], chi_p[0, i-1], ad2_s[0, i-1], ad2_p[0, i-1], is_rejected[0, i-1] = model_test(field_number, poisson.rvs(lams[i-1], size=int(np.nansum(a))), a, y)
        means[i-1], sigmas[i-1], a, y, residuals[1, i-1] = fit_model(field_number, a, dist='normal')
        ax.plot(np.arange(1, xmax+1), y, linewidth=0.5, label = 'Normal')
        ks_s[1, i-1], ks_p[1, i-1], ks2_s[1, i-1], ks2_p[1, i-1], chi_s[1, i-1], chi_p[1, i-1], ad2_s[1, i-1], ad2_p[1, i-1], is_rejected[1, i-1] = model_test(field_number, norm.rvs(loc=means[i-1], scale=sigmas[i-1], size=int(np.nansum(a))), a, y)
        rs[i-1], ps[i-1], a, y, residuals[2, i-1] = fit_model(field_number, a, dist='nbinom')
        ax.plot(np.arange(1, xmax+1), y, linewidth=0.5, label = 'Negative Binomial')
        ks_s[2, i-1], ks_p[2, i-1], ks2_s[2, i-1], ks2_p[2, i-1], chi_s[2, i-1], chi_p[2, i-1], ad2_s[2, i-1], ad2_p[2, i-1], is_rejected[2, i-1] = model_test(field_number, nbinom.rvs(n=rs[i-1], p=ps[i-1], size=int(np.nansum(a))), a, y)
        ax.legend()
        
        thres[:, i-1] = i
        field_nums[:, i-1] = np.nansum(a)

    plt.savefig(os.path.join(hist_saveloc, f"{trace['MiceID']}_session_{int(f1['session'][i])}_{trace['date']}.png"), dpi=600)
    plt.savefig(os.path.join(hist_saveloc, f"{trace['MiceID']}_session_{int(f1['session'][i])}_{trace['date']}.svg"), dpi=600)
    plt.close()
    
    with open(saveloc, 'wb') as handle:
        pickle.dump(data, handle)
    
    return (
        lams, residuals[0, :], chi_s[0, :], chi_p[0, :], ks_s[0, :], ks_p[0, :], ks2_s[0, :], ks2_p[0, :], ad2_s[0, :], ad2_p[0, :], is_rejected[0, :],
        means, sigmas, residuals[1, :], chi_s[1, :], chi_p[1, :], ks_s[1, :], ks_p[1, :], ks2_s[1, :], ks2_p[1, :], ad2_s[1, :], ad2_p[1, :], is_rejected[1, :],
        rs, ps, residuals[2, :], chi_s[2, :], chi_p[2, :], ks_s[2, :], ks_p[2, :], ks2_s[2, :], ks2_p[2, :], ad2_s[2, :], ad2_p[2, :], is_rejected[2, :],
        thres[0, :], field_nums[0, :]
    )

# Fig0021
# Place Field Number Counts.
def PlaceFieldNumber_Interface(trace, spike_threshold = 10, variable_names = None, is_placecell = True):
    KeyWordErrorCheck(trace, __file__, ['place_field_all'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell','Field Number'])

    field_number = trace['place_field_num']
    idx = np.where((np.isnan(field_number) == False)&(field_number != 0))[0]

    return idx+1, field_number[idx]

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
    
    if trace['maze_type'] in [0, 3]:
        return [np.mean(trace['is_placecell'])], [np.where(trace['is_placecell']==1)[0].shape[0]], [trace['is_placecell'].shape[0]]
    else:
        return [np.mean(trace['LA']['is_placecell'])], [np.where(trace['LA']['is_placecell']==1)[0].shape[0]], [trace['LA']['is_placecell'].shape[0]]

def PlaceCellPercentageCP_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = False):
    KeyWordErrorCheck(trace, __file__, ['is_placecell'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['percentage', 'place cell num', 'total cell num'])
    
    return [np.mean(trace['is_placecell'])], [np.where(trace['is_placecell']==1)[0].shape[0]], [trace['is_placecell'].shape[0]]

# Fig0024 Place Field Number Change
def PlaceFieldNumberChange_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    KeyWordErrorCheck(trace, __file__, ['place_field_all'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Field Number', 'Path Type'])

    if trace['maze_type'] == 0:
        idx = np.where(trace['is_placecell'] == 1)[0]
        place_field_all = trace['place_field_all']
        
        idx2 = np.where(trace['is_placecell'] == 1)[0]
        place_field_all2 = trace['place_field_all']
    else:
        idx = np.where(trace['LA']['is_placecell'] == 1)[0]
        place_field_all = trace['LA']['place_field_all']
        
        idx2 = np.where(trace['is_placecell'] == 1)[0]
        place_field_all2 = trace['place_field_all']
        
    field_number = np.zeros(len(idx), dtype = np.int64)
    field_number2 = np.zeros(len(idx2), dtype = np.int64)
    for i, k in enumerate(idx):
        field_number[i] = len(place_field_all[k].keys())
        
    for i, k in enumerate(idx2):
        field_number2[i] = len(place_field_all2[k].keys())

    if trace['maze_type'] == 0:
        return [np.nanmean(field_number)], ["OP"]
    else:
        return [np.nanmean(field_number), np.nanmean(field_number2)], np.array(['AP', 'CP'])

# Fig0025 Percentage of PCsf
def PercentageOfPCsf_Interface(trace = {}, spike_threshold = 10, variable_names = None, is_placecell = True):
    KeyWordErrorCheck(trace, __file__, ['place_field_all'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Percentage', 'Path Type'])

    if trace['maze_type'] == 0:
        idx = np.where(trace['is_placecell'] == 1)[0]
        place_field_all = trace['place_field_all']
        
        idx2 = np.where(trace['is_placecell'] == 1)[0]
        place_field_all2 = trace['place_field_all']
    else:
        idx = np.where(trace['LA']['is_placecell'] == 1)[0]
        place_field_all = trace['LA']['place_field_all']
        
        idx2 = np.where(trace['is_placecell'] == 1)[0]
        place_field_all2 = trace['place_field_all']
    
    field_number = np.zeros(len(idx), dtype = np.int64)
    field_number2 = np.zeros(len(idx2), dtype = np.int64)
    for i, k in enumerate(idx):
        field_number[i] = len(place_field_all[k].keys())
        
    for i, k in enumerate(idx2):
        field_number2[i] = len(place_field_all2[k].keys())

    if trace['maze_type'] == 0:
        return np.array([len(np.where(field_number == 1)[0])/len(np.where(field_number != 0)[0])]), np.array(['OP'])
    else:
        return np.array([len(np.where(field_number == 1)[0])/len(np.where(field_number != 0)[0]), len(np.where(field_number2 == 1)[0])/field_number2.shape[0]]), np.array(["AP", "CP"])

def FieldDistributionStatistics_TestAll_Interface(
    trace: dict,
    spike_threshold: int = 10,
    variable_names: list = None,
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['lam', 'Poisson KS Statistics', 'Poisson KS P-Value', 
                                                      'r', 'p', 'nbinom KS Statistics', 'nbinom KS P-Value',
                                                      'mean', 'sigma', 'Normal KS Statistics', 'Normal KS P-Value'])
    
    idx = np.where(trace['is_placecell'] == 1)[0] if 'LA' not in trace.keys() else np.where(trace['LA']['is_placecell'] == 1)[0]
    field_number = cp.deepcopy(trace['place_field_num']) if 'LA' not in trace.keys() else cp.deepcopy(trace['LA']['place_field_num'])
    field_number = field_number[np.where((np.isnan(field_number) == False)&(field_number != 0))[0]]
    
    #field_number = trace['place_field_num'][idx] if 'LA' not in trace.keys() else trace['LA']['place_field_num'][idx]

    if len(field_number) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    xmax = int(np.max(field_number))
    # Fit Poisson Distribution
    a = np.histogram(field_number, bins=xmax, range=(0.5, xmax+0.5))[0]
    prob = a / np.nansum(a)
    lam = EqualPoissonFit(np.arange(1,xmax+1), prob)
    mean, sigma = NormalFit(np.arange(1,xmax+1), prob)
    r, p = NegativeBinomialFit(np.arange(1,xmax+1), prob)
    
    maze = 'Open Field' if trace['maze_type'] == 0 else 'Maze '+str(trace['maze_type'])
    print(trace['MiceID'], trace['date'], maze)
    # Kolmogorov-Smirnov Test for three alternative test.
    print(f"Test for Poisson Distribution: lam = {lam}")
    poisson_ks, poison_pvalue = poisson_kstest(field_number)
    print(poisson_ks, poison_pvalue)
    print(f"Test for Normal Distribution: mean = {mean}, sigma = {sigma}")
    normal_ks, normal_pvalue = normal_kstest(field_number)
    print(normal_ks, normal_pvalue)
    print(f"Test for Negative Binomial Distribution: r = {r}, p = {p}")
    nbinom_ks, nbinom_pvalue = nbinom_kstest(field_number, monte_carlo_times=1000)
    print(nbinom_ks, nbinom_pvalue)
    print()
    
    return (np.array([lam]), np.array([poisson_ks]), np.array([poison_pvalue]), 
            np.array([r]), np.array([p]), np.array([nbinom_ks]), np.array([nbinom_pvalue]), 
            np.array([mean]), np.array([sigma]), np.array([normal_ks]), np.array([normal_pvalue]))
    
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
def NeuralDecodingResults_Interface(trace: dict, spike_threshold = 10, variable_names = None, is_placecell = False):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Error'])
    
    pred, test = trace['y_pred'].astype(np.int64), trace['y_test'].astype(np.int64)
    maze_type = trace['maze_type']
    D48 = GetDMatrices(maze_type=maze_type, nx=48)
    
    return [np.median(D48[(pred-1, test-1)])]


# Fig0033 Peak Velocity
def PeakVelocity_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, 
                           is_placecell: bool = False):
    KeyWordErrorCheck(trace, __file__, ['behav_nodes', 'behav_speed', 'n_neuron', 'old_map_clear'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Cell', 'velocity'])

    is_pc = trace['is_placecell']

    if is_placecell == True:
        idx = np.where(is_pc == 1)[0]
    else:
        idx = np.arange(trace['is_placecell'].shape[0])

    velocity = []
    cell_id = []

    for i in idx:
        peak_bin = np.nanargmax(trace['old_map_clear'][i, :]) + 1
        v = peak_velocity(behav_nodes=trace['behav_nodes'], behav_speed=trace['behav_speed'], idx=peak_bin)
        velocity.append(np.nanmean(v))
        cell_id.append(i+1)

    return np.array(cell_id, dtype=np.int64), np.array(velocity, dtype=np.float64)


def Coverage_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, 
                           is_placecell: bool = False):   
    KeyWordErrorCheck(trace, __file__, ['processed_pos_new'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Coverage', 'Bin Number'])

    coverage = np.zeros(5, dtype=np.float64)
    
    coverage[0] = calc_coverage(trace['processed_pos_new'], 12, 12)*100
    coverage[2] = calc_coverage(trace['processed_pos_new'], 20, 20)*100
    coverage[1] = calc_coverage(trace['processed_pos_new'], 24, 24)*100
    coverage[3] = calc_coverage(trace['processed_pos_new'], 36, 36)*100
    coverage[4] = calc_coverage(trace['processed_pos_new'], 48, 48)*100
    
    return coverage, np.array([12, 20, 24, 36, 48])



def Speed_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, 
                           is_placecell: bool = False):
    if 'correct_speed' not in trace.keys():
        correct_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
    else:
        correct_speed = trace['correct_speed']
    
    KeyWordErrorCheck(trace, __file__, ['correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Speed', 'Maze Bin'])
    
    nodes = spike_nodes_transform(trace['correct_nodes'], nx = 12)
    
    if trace['maze_type'] != 0:
        CP = CorrectPath_maze_1 if trace['maze_type'] == 1 else CorrectPath_maze_2
    else:
        CP = np.arange(1, 145)
    
    mean_speed = np.zeros(CP.shape[0], dtype = np.float64)
    for i in range(CP.shape[0]):
        idx = np.where(nodes==CP[i])[0]
        mean_speed[i] = np.nanmean(correct_speed[idx])
        
    return mean_speed, np.arange(1, CP.shape[0]+1)

def InterSessionCorrelation_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, 
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
      
    KeyWordErrorCheck(trace, __file__, ['fir_sec_corr', 'odd_even_corr', 'is_placecell'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Half-half Correlation', 'Odd-even Correlation'])
    
    return np.array([np.mean(trace['fir_sec_corr'][np.where(trace['is_placecell'] == 1)[0]])]), np.array([np.mean(trace['odd_even_corr'][np.where(trace['is_placecell'] == 1)[0]])])

from scipy.stats import poisson, norm
# Fig0039
def KSTestPoisson_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, cell_num_bound: int = 50,
                           is_placecell: bool = True):
    
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Statistic', 'PValue'])   
    field_number_pc = field_number_session(trace, is_placecell = True, spike_thre = spike_threshold)
    if len(field_number_pc) < cell_num_bound:
        return [np.nan], [np.nan]
    MAX = np.nanmax(field_number_pc)
    density = plt.hist(field_number_pc, range=(0.5, MAX+0.5), bins = int(MAX), density=True)[0]
    plt.close()
    lam = EqualPoissonFit(np.arange(1,MAX+1), density)
    sta, p = scipy.stats.kstest(field_number_pc, poisson.rvs(lam, size=len(field_number_pc)), alternative='two-sided')
    return [sta], [p]

#Fig0039-2
def KSTestNormal_Interface(trace: dict, spike_threshold: int | float = 10, 
                           variable_names: list | None = None, cell_num_bound: int = 50,
                           is_placecell: bool = True):
    
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Statistic', 'PValue'])   
    field_number_pc = field_number_session(trace, is_placecell = True, spike_thre = spike_threshold)
    if len(field_number_pc) < cell_num_bound:
        return [np.nan], [np.nan]
    
    MAX = np.nanmax(field_number_pc)
    density = plt.hist(field_number_pc, range=(0.5, MAX+0.5), bins = int(MAX), density=True)[0]
    plt.close()
    sigma, miu = NormalFit(np.arange(1,MAX+1), density)
    sta, p = scipy.stats.kstest(field_number_pc, norm.rvs(loc=miu, scale=sigma, size=len(field_number_pc)), alternative='two-sided')
    return [sta], [p]

# Fig0040
def FieldNumber_InSessionStability_Interface(trace: dict, spike_threshold: int | float = 10, 
                                             variable_names: list | None = None, 
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
def InFieldCorrelation_Interface(trace: dict, spike_threshold: int | float = 10, 
                                             variable_names: list | None = None, 
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
def PVCorrelations_Interface(trace: dict, spike_threshold: int | float = 10, 
                                             variable_names: list | None = None, 
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
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
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
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
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

#Fig 0050 Lapwise travel distance
def LapwiseTravelDistance_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True,
    field_num_bound: int = 10
):
    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Lap ID', 'Lap-wise Distance'])
    
    try:
        beg, end = LapSplit(trace, trace['paradigm'])
    except:
        print(trace['MiceID'], trace['date'], trace['maze_type'])
    
    dis = np.zeros(end.shape[0], np.float64)
    
    for i in range(beg.shape[0]):
        pos = trace['correct_pos'][beg[i]:end[i], :]
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        
        dis[i] = np.nansum(np.sqrt(dx**2 + dy**2))
    
    return np.arange(1, beg.shape[0]+1), dis

#Fig0051
def AverageVelocity_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True,
    field_num_bound: int = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Average Velocity'])
    if trace['maze_type'] == 0:
        pos = trace['correct_pos']
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dis = np.nansum(np.sqrt(dx**2 + dy**2))
        
        dt = (trace['correct_time'][-1] - trace['correct_time'][0])/1000
        return np.array([dis/dt], np.float64)
        
        
    try:
        beg, end = LapSplit(trace, trace['paradigm'])
    except:
        print(trace['MiceID'], trace['date'], trace['maze_type'])
    
    dis = np.zeros(end.shape[0], np.float64)
    dt = np.zeros(end.shape[0], np.float64)
    
    for i in range(beg.shape[0]):
        pos = trace['correct_pos'][beg[i]:end[i], :]
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        
        dis[i] = np.nansum(np.sqrt(dx**2 + dy**2))
        
        dt[i] = (trace['correct_time'][end[i]-1] - trace['correct_time'][beg[i]])/1000
        
    return np.array([np.nansum(dis)/np.nansum(dt)], np.float64)


def FirstExposure_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Average Velocity', 'Time'])
    
    if trace['maze_type'] == 0:
        pos = trace['correct_pos']
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dis = np.nansum(np.sqrt(dx**2 + dy**2))
        
        dt = (trace['correct_time'][-1] - trace['correct_time'][0])/1000
        return np.array([dis/dt], np.float64), np.array([dt], np.float64)
    
    try:
        beg, end = LapSplit(trace, trace['paradigm'])
    except:
        print(trace['MiceID'], trace['date'], trace['maze_type'])
    
    dis = np.zeros(end.shape[0], np.float64)
    dt = np.zeros(end.shape[0], np.float64)
    
    for i in range(beg.shape[0]):
        pos = trace['correct_pos'][beg[i]:end[i], :]
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        
        dis[i] = np.nansum(np.sqrt(dx**2 + dy**2))
        
        dt[i] = (trace['correct_time'][end[i]-1] - trace['correct_time'][beg[i]])/1000
        
    return np.array([dis[0]/dt[0]], np.float64), np.array([dt[0]], np.float64)

# Fig 0053
def LapNum_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Lap Num'])
    try:
        beg, end = LapSplit(trace, trace['paradigm'])
        return np.array([beg.shape[0]], np.int64)
    except:
        print(trace['MiceID'], trace['date'], trace['maze_type'])
        return np.array([], np.int64)
        
# Fig 0055
def LapwiseAverageVelocity_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Lap-wise Average Velocity', 'Lap ID'])

    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    beg, end = LapSplit(trace, trace['paradigm'])
    velocity = np.zeros(end.shape[0], np.float64)
    
    for i in range(beg.shape[0]):
        pos = trace['correct_pos'][beg[i]:end[i], :]
        x, y = pos[:, 0]/10, pos[:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        
        dt = (trace['correct_time'][end[i]-1] - trace['correct_time'][beg[i]])/1000
        velocity[i] = np.nansum(np.sqrt(dx**2 + dy**2))/dt # cm/s
        
    return velocity, np.arange(1, beg.shape[0]+1)


# Fig0057 Mean Rate on Diff Path
def MeanRateDiffPath_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Mean Rate', 'Path Type'])
    
    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    if 'LA' not in trace.keys():
        Spikes = cp.deepcopy(trace['Spikes'])
        spike_nodes = spike_nodes_transform(cp.deepcopy(trace['spike_nodes']), 12)
        dt = np.ediff1d(trace['ms_time_behav'])
        dt = np.append(dt, np.median(dt))
        dt[dt>100] = 100
    else:
        Spikes = cp.deepcopy(trace['LA']['Spikes'])
        spike_nodes = spike_nodes_transform(cp.deepcopy(trace['LA']['spike_nodes']), 12)
        dt = np.ediff1d(trace['LA']['ms_time_behav'])
        dt = np.append(dt, np.median(dt))
        dt[dt>100] = 100
    
    CP = correct_paths[int(trace['maze_type'])]
    IP = incorrect_paths[int(trace['maze_type'])]
    
    cpidx = np.concatenate([np.where(spike_nodes == i)[0] for i in CP])
    ipidx = np.concatenate([np.where(spike_nodes == i)[0] for i in IP])
    
    cptime = np.nansum(dt[cpidx])/1000
    iptime = np.nansum(dt[ipidx])/1000
    
    cpnum = np.nansum(Spikes[:, cpidx], axis=1)
    ipnum = np.nansum(Spikes[:, ipidx], axis=1)
    
    return np.concatenate([cpnum/cptime, ipnum/iptime]), np.concatenate([np.repeat('CP', Spikes.shape[0]), np.repeat('IP', Spikes.shape[0])])
    
# Fig0058 Occupation Time on Correct&Incorrect Track
def OccupationTimeDiffPath_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Occupation Time Percentage', 'Ratio', 'Bin-wise Mean Time', 'Bin-wise Ratio', 'Path Type'])
    
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], 12)
    CP = correct_paths[int(trace['maze_type'])]
    IP = incorrect_paths[int(trace['maze_type'])]
    
    dt = np.ediff1d(trace['correct_time'])
    dt = np.append(dt, np.median(dt))
    dt[dt>100] = 100
    
    cpidx = np.concatenate([np.where(behav_nodes == i)[0] for i in CP])
    ipidx = np.concatenate([np.where(behav_nodes == i)[0] for i in IP])
    
    cptime = np.nansum(dt[cpidx])/1000
    iptime = np.nansum(dt[ipidx])/1000
    
    return np.array([cptime/(cptime+iptime), iptime/(cptime+iptime)], np.float64), np.array([cptime/iptime, cptime/iptime], np.float64), np.array([cptime/len(CP), iptime/len(IP)], np.float64), np.array([cptime*len(IP)/len(CP)/iptime, np.nan], np.float64), np.array(['CP', 'IP'])

# Fig0059 - Naive - PVC Half-Half Correlation
def Naive_PVCHalfHalfCorrelation_Interface(
    trace: dict,
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = False
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['PVC Half-Half Correlation', 'Bin'])
    
    if 'LA' not in trace.keys() or trace['maze_type'] == 0:
        smooth_map_fir = trace['smooth_map_fir']
        smooth_map_sec = trace['smooth_map_sec']
    else:
        smooth_map_fir = trace['LA']['smooth_map_fir']
        smooth_map_sec = trace['LA']['smooth_map_sec']
        
    PVC = np.zeros(smooth_map_fir.shape[1])
    for i in range(smooth_map_fir.shape[1]):
        PVC[i] = pearsonr(smooth_map_fir[:, i], smooth_map_sec[:, i])[0]
    
    return PVC, np.arange(1, PVC.shape[0]+1)
    
    
# Fig04+ Field Pool
from mylib.field.within_field import within_field_half_half_correlation, within_field_odd_even_correlation
def WithinFieldBasicInfo_Interface(
    trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['FSC Stability', 'OEC Stability', 'Field Size', 'Field Length', 'Peak Rate', 'Position'])
    
    FSCList = within_field_half_half_correlation(
        trace['smooth_map_fir'],
        trace['smooth_map_sec'],
        trace['place_field_all']
    )
    
    OECList = within_field_odd_even_correlation(
        trace['smooth_map_odd'],
        trace['smooth_map_evn'],
        trace['place_field_all']
    )
    
    D = GetDMatrices(trace['maze_type'], 48)
    if trace['maze_type'] != 0:
        CP = correct_paths[int(trace['maze_type'])]
    FSC, OEC, SIZE, RATE, POSITION, LENGTH = [], [], [], [], [], []
    
    for i in range(len(FSCList)):
        if trace['is_placecell'][i] == 1:
            for k in FSCList[i].keys():
                POSITION.append(D[0, k-1]) 
                    
                FSC.append(FSCList[i][k])
                OEC.append(OECList[i][k])
                SIZE.append(len(trace['place_field_all'][i][k]))
                LENGTH.append(np.max(D[0, trace['place_field_all'][i][k]-1]) - np.min(D[0, trace['place_field_all'][i][k]-1]))
                RATE.append(trace['smooth_map_all'][i, k-1])
                
    FSC = np.array(FSC, np.float64)
    OEC = np.array(OEC, np.float64)
    SIZE = np.array(SIZE, np.int64)
    RATE = np.array(RATE, np.float64)
    LENGTH = np.array(LENGTH, np.float64)
    POSITION = np.array(POSITION, np.float64)
    
    return FSC, OEC, SIZE, LENGTH, RATE, POSITION


#Fig 04+ Place Cells' Fields Independent Test
def WithinCellFieldStatistics_Interface(trace: dict, 
    spike_threshold: int | float = 10,
    variable_names: list | None = None, 
    is_placecell: bool = True,
    field_num_bound: int = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Mean FSC', 'Std. FSC', 'Median FSC', 'Error FSC',
                                                                            'Mean OEC', 'Std. OEC', 'Median OEC', 'Error OEC',
                                                                            'Mean Size', 'Std. Size', 'Median Size', 'Error Size',
                                                                            'Mean Length', 'Std. Length', 'Median Length', 'Error Length',
                                                                            'Mean Rate', 'Std. Rate', 'Median Rate', 'Error Rate',
                                                                            'Mean Position', 'Std. Position', 'Median Position', 'Error Position',
                                                                            'Mean Interdistance', 'Std. Interdistance', 'Median Interdistance', 'Error Interdistance',
                                                                            'Cell ID', 'Field Number'])

    FSCList = within_field_half_half_correlation(
        trace['smooth_map_fir'],
        trace['smooth_map_sec'],
        trace['place_field_all']
    )
    
    OECList = within_field_odd_even_correlation(
        trace['smooth_map_odd'],
        trace['smooth_map_evn'],
        trace['place_field_all']
    )
    
    D = GetDMatrices(trace['maze_type'], 48)
    if trace['maze_type'] != 0:
        CP = correct_paths[int(trace['maze_type'])]
        
    mean_fsc, std_fsc, median_fsc, err_fsc = [], [], [], []
    mean_oec, std_oec, median_oec, err_oec = [], [], [], []
    mean_size, std_size, median_size, err_size = [], [], [], []
    mean_length, std_length, median_length, err_length = [], [], [], []
    mean_rate, std_rate, median_rate, err_rate = [], [], [], []
    mean_position, std_position, median_position, err_position = [], [], [], []
    mean_interdistance, std_interdistance, median_interdistance, err_interdistance = [], [], [], []
    CellID, FieldNumber = [], []
    
    for i in range(len(trace['place_field_all'])):
        if trace['is_placecell'][i] == 1:
            
            if len(trace['place_field_all'][i].keys()) <= 1:
                continue
            
            CellID.append(i)
            FieldNumber.append(len(trace['place_field_all'][i].keys()))
            
            FSC, OEC, SIZE, LENGTH, RATE, POSITION = [], [], [], [], [], []
            
            for j, k in enumerate(trace['place_field_all'][i].keys()):
                POSITION.append(D[0, k-1])  
                FSC.append(FSCList[i][k])
                OEC.append(OECList[i][k])
                SIZE.append(len(trace['place_field_all'][i][k]))
                LENGTH.append(np.max(D[0, trace['place_field_all'][i][k]-1]) - np.min(D[0, trace['place_field_all'][i][k]-1]))
                RATE.append(trace['smooth_map_all'][i, k-1])
                POSITION.append(D[0, k-1])
                
            FSC = np.array(FSC, np.float64)
            OEC = np.array(OEC, np.float64)
            SIZE = np.array(SIZE, np.float64)
            LENGTH = np.array(LENGTH, np.float64)
            RATE = np.array(RATE, np.float64)
            POSITION = np.array(POSITION, np.float64)
            
            SortedPosition = np.sort(POSITION)
            interdistance = np.abs(np.ediff1d(SortedPosition))
                
            mean_fsc.append(np.nanmean(FSC)), std_fsc.append(np.nanstd(FSC)), median_fsc.append(np.nanmedian(FSC)), err_fsc.append(np.nansum(np.abs(FSC-np.nanmean(FSC))))
            mean_oec.append(np.nanmean(OEC)), std_oec.append(np.nanstd(OEC)), median_oec.append(np.nanmedian(OEC)), err_oec.append(np.nansum(np.abs(OEC-np.nanmean(OEC))))
            mean_size.append(np.nanmean(SIZE)), std_size.append(np.nanstd(SIZE)), median_size.append(np.nanmedian(SIZE)), err_size.append(np.nansum(np.abs(SIZE-np.nanmean(SIZE))))
            mean_length.append(np.nanmean(LENGTH)), std_length.append(np.nanstd(LENGTH)), median_length.append(np.nanmedian(LENGTH)), err_length.append(np.nansum(np.abs(LENGTH-np.nanmean(LENGTH))))
            mean_rate.append(np.nanmean(RATE)), std_rate.append(np.nanstd(RATE)), median_rate.append(np.nanmedian(RATE)), err_rate.append(np.nansum(np.abs(RATE-np.nanmean(RATE))))
            mean_position.append(np.nanmean(POSITION)), std_position.append(np.nanstd(POSITION)), median_position.append(np.nanmedian(POSITION)), err_position.append(np.nansum(np.abs(POSITION-np.nanmean(POSITION))))
            mean_interdistance.append(np.nanmean(interdistance)), std_interdistance.append(np.nanstd(interdistance)), median_interdistance.append(np.nanmedian(interdistance)), err_interdistance.append(np.nansum(np.abs(interdistance-np.nanmean(interdistance))))

    mean_fsc, std_fsc, median_fsc, err_fsc = np.array(mean_fsc, np.float64), np.array(std_fsc, np.float64), np.array(median_fsc, np.float64), np.array(err_fsc, np.float64)
    mean_oec, std_oec, median_oec, err_oec = np.array(mean_oec, np.float64), np.array(std_oec, np.float64), np.array(median_oec, np.float64), np.array(err_oec, np.float64)
    mean_size, std_size, median_size, err_size = np.array(mean_size, np.float64), np.array(std_size, np.float64), np.array(median_size, np.float64), np.array(err_size, np.float64)
    mean_length, std_length, median_length, err_length = np.array(mean_length, np.float64), np.array(std_length, np.float64), np.array(median_length, np.float64), np.array(err_length, np.float64)
    mean_rate, std_rate, median_rate, err_rate = np.array(mean_rate, np.float64), np.array(std_rate, np.float64), np.array(median_rate, np.float64), np.array(err_rate, np.float64)
    mean_position, std_position, median_position, err_position = np.array(mean_position, np.float64), np.array(std_position, np.float64), np.array(median_position, np.float64), np.array(err_position, np.float64)
    mean_interdistance, std_interdistance, median_interdistance, err_interdistance = np.array(mean_interdistance, np.float64), np.array(std_interdistance, np.float64), np.array(median_interdistance, np.float64), np.array(err_interdistance, np.float64)
    CellID, FieldNumber = np.array(CellID, np.int64), np.array(FieldNumber, np.int64)
    
    return (mean_fsc, std_fsc, median_fsc, err_fsc,
            mean_oec, std_oec, median_oec, err_oec,
            mean_size, std_size, median_size, err_size,
            mean_length, std_length, median_length, err_length,
            mean_rate, std_rate, median_rate, err_rate,
            mean_position, std_position, median_position, err_position,
            mean_interdistance, std_interdistance, median_interdistance, err_interdistance,
            CellID, FieldNumber
    )

# Fig0060
def PerfectLapIdentify_Interface(
    trace,
    spike_threshold: int | float = 10,
    variable_names: list | None = None
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Lap ID', 'Perfect Lap', 'Distance', 'Navigation Time', 'Average Velocity'])
    
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    beg, end = LapSplit(trace, trace['paradigm'])
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    behav_pos = cp.deepcopy(trace['correct_pos'])
    D = GetDMatrices(trace['maze_type'], 12)
    CP = cp.deepcopy(correct_paths[trace['maze_type']])
    
    is_perfect = np.ones(beg.shape[0], np.int64)
    distance = np.zeros(beg.shape[0], np.float64)
    navigation_time = np.zeros(beg.shape[0], np.float64)
    average_velocity = np.zeros(beg.shape[0], np.float64)
    
    for i in range(beg.shape[0]):
        dx = np.ediff1d(behav_pos[beg[i]:end[i], 0])
        dy = np.ediff1d(behav_pos[beg[i]:end[i], 1])
        distance[i] = np.sum(np.sqrt(dx**2 + dy**2))/10
        
        navigation_time[i] = (trace['correct_time'][end[i]] - trace['correct_time'][beg[i]])/1000
        average_velocity[i] = distance[i]/navigation_time[i]
        
        for j in range(beg[i]+1, end[i]):
            if behav_nodes[j] not in CP:
                is_perfect[i] = 0
                break
            
            if int(D[behav_nodes[j]-1, 0]*100) < int(D[behav_nodes[j-1]-1, 0]*100):
                is_perfect[i] = 0
                break
    
    return np.arange(1, is_perfect.shape[0]+1, dtype=np.int64), is_perfect, distance, navigation_time, average_velocity

def PerfectLapPercentage_Interface(
    trace,
    spike_threshold: int | float = 10,
    variable_names: list | None = None
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Perfect Lap Percentage'])
    
    if trace['maze_type'] == 0:
        return np.array([])
    
    beg, end = LapSplit(trace, trace['paradigm'])
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], nx=12)
    D = GetDMatrices(trace['maze_type'], 12)
    CP = cp.deepcopy(correct_paths[trace['maze_type']])
    
    is_perfect = np.ones(beg.shape[0], np.int64)
    
    for i in range(beg.shape[0]):
        for j in range(beg[i]+1, end[i]):
            if behav_nodes[j] not in CP:
                is_perfect[i] = 0
                print(behav_nodes[j], 'incorrect track')
                break
            
            if int(D[behav_nodes[j]-1, 0]*100) < int(D[behav_nodes[j-1]-1, 0]*100):
                is_perfect[i] = 0
                print(behav_nodes[j], 'turn around', D[behav_nodes[j-1]-1, 0], '>', D[behav_nodes[j]-1, 0])
                break
    
    return np.array([np.nanmean(is_perfect)])

def PlaceCellPercentage_ReverseInterface(
    trace,
    spike_threshold: int | float = 10,
    variable_names: list | None = None
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Percentage', 'Direction', 'Cell Number', 'Place Cell Number'])
    return (np.array([np.mean(trace['cis']['is_placecell']), np.mean(trace['trs']['is_placecell'])], np.float64), 
            np.array(['cis', 'trs']),
            np.array([trace['cis']['is_placecell'].shape[0], trace['trs']['is_placecell'].shape[0]], np.int64),
            np.array([np.sum(trace['cis']['is_placecell']), np.sum(trace['trs']['is_placecell'])], np.float64),
    )
    

#Fig0402 Linearized Position
def ErrorTimesAndFieldFraction_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Error Num', 'Pass Number', 'Error Rate', 'Decision Point'])
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    pass_num, err_num = lapwise_behavioral_score(trace)
    return err_num, pass_num, err_num/pass_num, np.arange(1, pass_num.shape[0]+1, dtype=np.int64)

# Fig0402 Linearized Position (Over-representation and occupation time)
def OccupationTimeAndFieldFraction_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Occupation Time', 'Decision Point'])
    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    DP = cp.deepcopy(DPs[trace['maze_type']])
    F2S = cp.deepcopy(Father2SonGraph)
    occu_time = np.zeros_like(DP)
    
    for i in range(DP.shape[0]):
        occu_time[i] = np.nansum(trace['occu_time_spf'][np.array(F2S[DP[i]])-1])
    
    return occu_time, np.arange(1, occu_time.shape[0]+1, dtype=np.int64)

def PlaceFieldCoveredDensity_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Field Num', 'Position'])
    if 'field_reg' not in trace.keys():
        return np.array([]), np.array([])
    
    count_map = np.zeros(2304)
    D = GetDMatrices(trace['maze_type'], 48)
    CP = cp.deepcopy(correct_paths[(trace['maze_type'], 48)])
    
    for i in range(trace['field_reg'].shape[0]):
        j, k = trace['field_reg'][i, 0], trace['field_reg'][i, 2]
        count_map[trace['place_field_all'][j][k]-1] += 1

    return count_map[CP-1], D[CP-1, 0]
        

def FieldCountPerSession_Interface(
    trace,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Path Type', 'Threshold', 'Field Count', 'Field Number'])
    
    place_field_cp10, place_field_cp5 = cp.deepcopy(trace['place_field_all']), cp.deepcopy(trace['place_field_all5'])
    field_numbercp10 = np.array([len(i.keys()) for i in place_field_cp10], np.float64)
    field_numbercp5 = np.array([len(i.keys()) for i in place_field_cp5], np.float64)
    idx = np.where(trace['is_placecell'] == 1)[0]
    rescp10 = plt.hist(field_numbercp10[idx], range=(0.5, 50.5), bins=50)[0]
    rescp5 = plt.hist(field_numbercp5[idx], range=(0.5, 50.5), bins=50)[0]
    
    if trace['maze_type'] in [1, 2]:
        place_field_all10, place_field_all5 = cp.deepcopy(trace['LA']['place_field_all']), cp.deepcopy(trace['LA']['place_field_all5'])
        field_numberall10, field_numberall5 = np.array([len(i.keys()) for i in place_field_all10], np.float64), np.array([len(i.keys()) for i in place_field_all5], np.float64)
        idx = np.where(trace['LA']['is_placecell'] == 1)[0]
        resall10 = plt.hist(field_numberall10[idx], range=(0.5, 50.5), bins=50)[0]
        resall5 = plt.hist(field_numberall5[idx], range=(0.5, 50.5), bins=50)[0]
        plt.close()
        return np.concatenate([np.repeat("CP", 50), np.repeat("All", 50), np.repeat("CP", 50), np.repeat("All", 50)]), np.concatenate([np.repeat(10, 100), np.repeat(5, 100)]), np.concatenate([rescp10, resall10, rescp5, resall5]), np.concatenate([np.arange(1,51), np.arange(1, 51), np.arange(1,51), np.arange(1,51)])
    else:
        plt.close()
        return np.repeat("OP", 100), np.concatenate([np.repeat(10, 50), np.repeat(5, 50)]), np.concatenate([rescp10, rescp5]), np.concatenate([np.arange(1,51), np.arange(1,51)])
    
from mylib.decoder.NaiveBayesianDecoder import NaiveBayesDecoder 
def CrossMazeDecoder_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
    save_folder: str = r'E:\Data\Simulation_pc\cross_maze_decode_res',
    
):
    if os.path.exists(save_folder) == False:
        mkdir(save_folder)
        
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['MSE', 'RMSE', 'MAE', 'Std. MAE'])
    
    print(f"Mice  {trace['MiceID']}  Date  {trace['date']}, Session  {trace['session']}, Maze  {trace['maze_type']}------------------------------")
    print("  A. Initial parameters and split training and testing set.")
    if trace['maze_type'] == 0:
        T = trace['Spikes'].shape[1]
        RawSpikes = np.zeros_like(trace['RawTraces'], dtype=np.float64)
        
        RawStd = np.std(trace['RawTraces'], axis=1)
        for i in range(RawSpikes.shape[0]):
            RawSpikes[i, :] = np.where(trace['RawTraces'][i, :] >= 3*RawStd[i], 1, 0)
        
        RawSpikesProcessed = np.zeros_like(trace['Spikes'], dtype=np.float64)
        
        for i in range(T):
            t = np.where(trace['ms_time'] == trace['ms_time_behav'][i])[0][0]
            RawSpikesProcessed[:, i] = RawSpikes[:, t]
        
        beg, end = LapSplit(trace, trace['paradigm'])
        
        print("")
    
    decoder = NaiveBayesDecoder(trace, spike_threshold, save_folder)
    decoder.run()
    print("Decoding finished.", end='\n\n')
    return decoder

from mylib.stats.indeptest import indeptest
def IndeptTestForPositionAndFieldLength_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Statistic', 'P-Value', 
                                             'Pearson r', 'Pearson P-Value', 
                                             'Spearman r', 'Spearman P-Value'])
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    n_neuron = trace['Spikes'].shape[0]
    D = GetDMatrices(trace['maze_type'], 48)
    LENGTH = []
    POSITION = []
    
    for i in range(n_neuron):
        if trace['is_placecell'][i] == 1:
            for k in trace['place_field_all'][i].keys():
                LENGTH.append(np.max(D[0, trace['place_field_all'][i][k]-1]) - np.min(D[0, trace['place_field_all'][i][k]-1]))
                POSITION.append(D[k-1, 0])
    
    LENGTH = np.array(LENGTH)
    POSITION = np.array(POSITION)
        
    r, pearsonp = scipy.stats.pearsonr(LENGTH, POSITION)
    print("Pearson Correlation: ", r, pearsonp)
    s, spearmanp = scipy.stats.spearmanr(LENGTH, POSITION)
    print("Spearman Correlation: ", s, spearmanp)
    statistics, pvalue = indeptest(LENGTH, POSITION, shuffle_method='permutation', simu_times=1000)
    print("Independency test:", statistics, pvalue)
    
    return np.array([statistics]), np.array([pvalue]), np.array([r]), np.array([pearsonp]), np.array([s]), np.array([spearmanp])


def IndeptTestForPositionAndStability_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Statistic', 'P-Value', 
                                             'Pearson r', 'Pearson P-Value', 
                                             'Spearman r', 'Spearman P-Value'])
    FSCList = within_field_half_half_correlation(
        trace['smooth_map_fir'],
        trace['smooth_map_sec'],
        trace['place_field_all']
    )
    
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    n_neuron = trace['Spikes'].shape[0]
    D = GetDMatrices(trace['maze_type'], 48)
    FSC = []
    POSITION = []
    
    for i in range(n_neuron):
        if trace['is_placecell'][i] == 1:
            for k in trace['place_field_all'][i].keys():
                FSC.append(FSCList[i][k])
                POSITION.append(D[k-1, 0])
    
    FSC = np.array(FSC)
    POSITION = np.array(POSITION)
    
    idx = np.where((np.isnan(FSC)==False)&(np.isnan(POSITION)==False))[0]
    FSC, POSITION = FSC[idx], POSITION[idx]
        
    r, pearsonp = scipy.stats.pearsonr(POSITION, FSC)
    print("Pearson Correlation: ", r, pearsonp)
    s, spearmanp = scipy.stats.spearmanr(POSITION, FSC)
    print("Spearman Correlation: ", s, spearmanp)
    statistics, pvalue = indeptest(POSITION, FSC, shuffle_method='permutation', simu_times=1000)
    print("Independency test:", statistics, pvalue)
    
    return np.array([statistics]), np.array([pvalue]), np.array([r]), np.array([pearsonp]), np.array([s]), np.array([spearmanp])

def IndeptTestForPositionAndRate_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Statistic', 'P-Value', 
                                             'Pearson r', 'Pearson P-Value', 
                                             'Spearman r', 'Spearman P-Value'])
    
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    n_neuron = trace['Spikes'].shape[0]
    D = GetDMatrices(trace['maze_type'], 48)
    RATE = []
    POSITION = []
    
    for i in range(n_neuron):
        if trace['is_placecell'][i] == 1:
            for k in trace['place_field_all'][i].keys():
                RATE.append(trace['smooth_map_all'][i, k-1])
                POSITION.append(D[k-1, 0])
    
    RATE = np.array(RATE)
    POSITION = np.array(POSITION)
    
    idx = np.where((np.isnan(RATE)==False)&(np.isnan(POSITION)==False))[0]
    RATE, POSITION = RATE[idx], POSITION[idx]
        
    r, pearsonp = scipy.stats.pearsonr(POSITION, RATE)
    print("Pearson Correlation: ", r, pearsonp)
    s, spearmanp = scipy.stats.spearmanr(POSITION, RATE)
    print("Spearman Correlation: ", s, spearmanp)
    statistics, pvalue = indeptest(POSITION, RATE, shuffle_method='permutation', simu_times=1000)
    print("Independency test:", statistics, pvalue)
    
    return np.array([statistics]), np.array([pvalue]), np.array([r]), np.array([pearsonp]), np.array([s]), np.array([spearmanp])

# Fig0408 Statistic independence test for sibling fields' property.
def FieldPropertyIndependence_Chi2_MI_DoubleCheck_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Chi2 Statistic', 'Mutual Information', 'Field Pair Type', 'Variable', 'Pair Num'])
    stab_sib, stab_non = [], []
    size_sib, size_non = [], []
    rate_sib, rate_non = [], []
    
    field_reg = trace['field_reg']
    for i in range(field_reg.shape[0]):
        for j in range(field_reg.shape[0]):
            if i == j:
                continue
            
            if field_reg[i, 0] == field_reg[j, 0]:
                stab_sib.append([field_reg[i, 5], field_reg[j, 5]])
                size_sib.append([field_reg[i, 3], field_reg[j, 3]])
                rate_sib.append([field_reg[i, 4], field_reg[j, 4]])
            else:
                stab_non.append([field_reg[i, 5], field_reg[j, 5]])
                size_non.append([field_reg[i, 3], field_reg[j, 3]])
                rate_non.append([field_reg[i, 4], field_reg[j, 4]])
                
    stab_sib = np.array(stab_sib, np.float64)
    stab_non = np.array(stab_non, np.float64)[np.random.randint(0, len(stab_non), len(stab_sib)), :]
    size_sib = np.array(size_sib, np.int64)
    size_non = np.array(size_non, np.int64)[np.random.randint(0, len(size_non), len(size_sib)), :]
    rate_sib = np.array(rate_sib, np.float64)
    rate_non = np.array(rate_non, np.float64)[np.random.randint(0, len(rate_non), len(rate_sib)), :]
    
    stat_stab_sib, stat_stab_non, len_stab_sib, len_stab_non = indept_field_properties(
        real_distribution=field_reg[:, 5],
        X_pairs=stab_sib,
        Y_pairs=stab_non,
        n_bin=40
    )
    mi_stab_sib, mi_stab_non, _, _ = indept_field_properties_mutual_info(stab_sib, stab_non)
    
    stat_size_sib, stat_size_non, len_size_sib, len_size_non = indept_field_properties(
        real_distribution=field_reg[:, 3],
        X_pairs=size_sib,
        Y_pairs=size_non,
        n_bin=40
    )
    mi_size_sib, mi_size_non, _, _ = indept_field_properties_mutual_info(size_sib, size_non)
    
    stat_rate_sib, stat_rate_non, len_rate_sib, len_rate_non = indept_field_properties(
        real_distribution=field_reg[:, 4],
        X_pairs=rate_sib,
        Y_pairs=rate_non,
        n_bin=40
    )
    mi_rate_sib, mi_rate_non, _, _ = indept_field_properties_mutual_info(rate_sib, rate_non)
    
    print("  Stabilty: ", stat_stab_sib, stat_stab_non, "   MI: ", mi_stab_sib, mi_stab_non)
    print("  Size: ", stat_size_sib, stat_size_non, "   MI: ", mi_size_sib, mi_size_non)
    print("  Rate: ", stat_rate_sib, stat_rate_non, "   MI: ", mi_rate_sib, mi_rate_non)
    
    
    return ([stat_stab_sib, stat_stab_non, stat_size_sib, stat_size_non, stat_rate_sib, stat_rate_non],
            [mi_stab_sib, mi_stab_non, mi_size_sib, mi_size_non, mi_rate_sib, mi_rate_non],
            ['Sibling', 'Non-sibling', 'Sibling', 'Non-sibling', 'Sibling', 'Non-sibling'],
            ['Stability', 'Stability', 'Size', 'Size', 'Rate', 'Rate'],
            [len_stab_sib, len_stab_non, len_size_sib, len_size_non, len_rate_sib, len_rate_non])

# Fig0064
def PlaceFieldNum_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names,
                             check_variable=['Field Number', 'Direction'])
    return [np.nanmean(trace['cis']['place_field_num']), np.nanmean(trace['trs']['place_field_num'])], ['Cis', 'Trs']

def LapwiseDistance_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names,
                             check_variable=['Lap-wise Distance', 'LapID'])
    beg, end = LapSplit(trace, trace['paradigm'])
    dist = np.zeros(beg.shape[0], np.float64)
    for i in range(beg.shape[0]):
        x, y = trace['correct_pos'][beg[i]:end[i]+1, 0], trace['correct_pos'][beg[i]:end[i]+1, 1]
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dist[i] = np.nansum(np.sqrt(dx**2, dy**2))
    return dist, np.arange(1, beg.shape[0]+1)

def PlaceFieldOverlapProportion_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names,
                             check_variable=['Start Session', 'Interval', 'Cell Pair Number',
                                             'Turn-On Proportion', 'Turn-Off Proportion', 
                                             'Kept-On Proportion', 'Kept-Off Proportion', 
                                             'Prev Field Number', 'Next Field Number', 
                                             'Field-On Proportion', 'Field-Off Proportion', 
                                             'Field-Kept Proportion', 'Data Type'])
    try:
        return (trace['res']['Start Session'], trace['res']['Interval'], trace['res']['Cell Pair Number'],
            trace['res']['Turn-On Proportion'], trace['res']['Turn-Off Proportion'],
            trace['res']['Kept-On Proportion'], trace['res']['Kept-Off Proportion'],
            trace['res']['Prev Field Number'], trace['res']['Next Field Number'],
            trace['res']['Field-On Proportion'], trace['res']['Field-Off Proportion'],
            trace['res']['Field-Kept Proportion'], trace['res']['Data Type'])
    except:
        return (trace['Start Session'], trace['Interval'], trace['Cell Pair Number'],
            trace['Turn-On Proportion'], trace['Turn-Off Proportion'],
            trace['Kept-On Proportion'], trace['Kept-Off Proportion'],
            trace['Prev Field Number'], trace['Next Field Number'],
            trace['Field-On Proportion'], trace['Field-Off Proportion'],
            trace['Field-Kept Proportion'], trace['Data Type'])
        
# Fig0066 
def PlacecellOverlap_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Cis Percentage', 'Trs Percentage', 'Overlap Percentage'])
    
    overlap = np.zeros(trace['cis']['is_placecell'].shape[0])
    overlap[np.where((trace['cis']['is_placecell'] == 1) & (trace['trs']['is_placecell'] == 1))[0]] = 1
    
    return [np.mean(trace['cis']['is_placecell'])], [np.mean(trace['trs']['is_placecell'])], [np.mean(overlap)]

# Fig0067
def PlaceFieldNumberPerDirection_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Field Number', 'Direction'])
    
    cis_num, trs_num = trace['cis']['place_field_num'], trace['trs']['place_field_num']
    idx = np.where((cis_num>0)&(trs_num>0))[0]
    cis_num, trs_num = cis_num[idx], trs_num[idx]
    
    return np.concatenate([cis_num, trs_num]), np.array(['Cis']*len(cis_num) + ['Trs']*len(trs_num))

# Fig0067
def PlaceFieldNumberPerDirectionCorr_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Corr', 'Shuffle'])

    cis_num, trs_num = trace['cis']['place_field_num'], trace['trs']['place_field_num']
    idx = np.where((np.isnan(cis_num)==False)&(np.isnan(trs_num)==False)&(cis_num!=0)&(trs_num!=0))[0]
    cis_num, trs_num = cis_num[idx], trs_num[idx]
    
    corr, _ = spearmanr(cis_num, trs_num)
    np.random.shuffle(cis_num)
    np.random.shuffle(trs_num)
    shuffle, _ = spearmanr(cis_num, trs_num)
    
    return [corr], [shuffle]

# Fig0069 - model
from mylib.stats.gamma_poisson import gamma_poisson_pdf
def ModelPlaceFieldNumberPerDirection_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Corr', 'Data Type'])

    cis_num, trs_num = trace['cis']['place_field_num'], trace['trs']['place_field_num']
    idx = np.where((np.isnan(cis_num)==False)&(np.isnan(trs_num)==False)&(cis_num!=0)&(trs_num!=0))[0]
    field_num_cis, field_num_trs = cis_num[idx], trs_num[idx]
    
    corr, _ = pearsonr(field_num_cis, field_num_trs)

    field_num_cis = cis_num[np.where((cis_num > 0))[0]]
    field_num_trs = trs_num[np.where((trs_num > 0))[0]]

    if len(field_num_cis) < 100 and len(field_num_trs) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    xmax = int(np.max(field_num_cis))
    # Fit Poisson Distribution
    a = np.histogram(field_num_cis, bins=xmax, range=(0.5, xmax+0.5))[0]
    prob = a / np.nansum(a)
    lam_cis = EqualPoissonFit(np.arange(1,xmax+1), prob)
    r_cis, p_cis = NegativeBinomialFit(np.arange(1,xmax+1), prob)
    
    xmax = int(np.max(field_num_trs))
    a = np.histogram(field_num_trs, bins=xmax, range=(0.5, xmax+0.5))[0]
    prob = a / np.nansum(a)
    lam_trs = EqualPoissonFit(np.arange(1,xmax+1), prob)
    r_trs, p_trs = NegativeBinomialFit(np.arange(1,xmax+1), prob)
    
    
    simulated_num = np.max([len(field_num_cis), len(field_num_trs)])

    lams_cis = gamma.rvs(r_cis, scale = (1-p_cis)/p_cis, size = simulated_num)
    cdf_cis = gamma_poisson_pdf(r_cis, p_cis/(1-p_cis), max_lam=20, output='cdf', steps=20000)[1]
    cdf_trs = gamma_poisson_pdf(r_trs, p_trs/(1-p_trs), max_lam=20, output='cdf', steps=20000)[1]
    
    num_a, num_b = np.zeros(simulated_num), np.zeros(simulated_num)
    lams_trs = np.zeros(simulated_num)
    for i in range(simulated_num):
        num_a[i] = poisson.rvs(lams_cis[i], size = 1)
        lam = int(lams_cis[i]*1000)-1
        try:
            frac = np.where(cdf_trs >= cdf_cis[lam])[0][0]
        except:
            frac = len(cdf_trs)
        
        #print(lams_cis[i], frac/1000)
        num_b[i] = poisson.rvs(frac/1000, size = 1)
        lams_trs[i] = frac/1000
        
    corr_shuf = pearsonr(num_a, num_b)[0]#spearmanr(lams_cis, lams_trs)[0]#

    return [corr, corr_shuf], ['Data', 'Shuffle']

# Fig0068
def PoissonTest_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['lam', 'KS Statistics', 'KS P-Value',
                                                                            'r', 'p', 'KS Gamma', 'KS Gamma P-value',
                                                                            'Direction'])

    field_num_cis, field_num_trs = trace['cis']['place_field_num'], trace['trs']['place_field_num']
    field_num_cis = field_num_cis[np.where((np.isnan(field_num_cis) == False)&(field_num_cis != 0))[0]]
    field_num_trs = field_num_trs[np.where((np.isnan(field_num_trs) == False)&(field_num_trs != 0))[0]]

    if len(field_num_cis) < 100 and len(field_num_trs) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    xmax = int(np.max(field_num_cis))
    # Fit Poisson Distribution
    a = np.histogram(field_num_cis, bins=xmax, range=(0.5, xmax+0.5))[0]
    prob = a / np.nansum(a)
    lam_cis = EqualPoissonFit(np.arange(1,xmax+1), prob)
    r_cis, p_cis = NegativeBinomialFit(np.arange(1,xmax+1), prob)
    
    print(trace['MiceID'], trace['date'], ' Cis')
    # Kolmogorov-Smirnov Test for three alternative test.
    print(f"Test for Poisson Distribution: lam_cis = {lam_cis}")
    poisson_ks_cis, poison_pvalue_cis = poisson_kstest(field_num_cis)
    print(poisson_ks_cis, poison_pvalue_cis)
    print(f"Test for Negative Binomial Distribution: r_cis = {r_cis}, p_cis = {p_cis}")
    nbinom_ks_cis, nbinom_pvalue_cis = nbinom_kstest(field_num_cis, monte_carlo_times=1000)
    print(nbinom_ks_cis, nbinom_pvalue_cis)
    print()
    xmax = int(np.max(field_num_trs))
    # Fit Poisson Distribution
    a = np.histogram(field_num_trs, bins=xmax, range=(0.5, xmax+0.5))[0]
    prob = a / np.nansum(a)
    lam_trs = EqualPoissonFit(np.arange(1,xmax+1), prob)
    r_trs, p_trs = NegativeBinomialFit(np.arange(1,xmax+1), prob)
    
    print(trace['MiceID'], trace['date'], ' Trs')
    # Kolmogorov-Smirnov Test for three alternative test.
    print(f"Test for Poisson Distribution: lam_trs = {lam_trs}")
    poisson_ks_trs, poison_pvalue_trs = poisson_kstest(field_num_trs)
    print(poisson_ks_trs, poison_pvalue_trs)
    print(f"Test for Negative Binomial Distribution: r_trs = {r_trs}, p_trs = {p_trs}")
    nbinom_ks_trs, nbinom_pvalue_trs = nbinom_kstest(field_num_trs, monte_carlo_times=1000)
    print(nbinom_ks_trs, nbinom_pvalue_trs)
    
    return ([lam_cis, lam_trs], [poisson_ks_cis, poisson_ks_trs], [poison_pvalue_cis, poison_pvalue_trs], 
            [r_cis, r_trs], [p_cis, p_trs], [nbinom_ks_cis, nbinom_ks_trs], [nbinom_pvalue_cis, nbinom_pvalue_trs], ['cis', 'trs'])

# Fig0069 
def PlaceFieldOverlap_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
    overlap_thre: float = 0.6
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Cis Number', 'Trs Number', 'Overlap', 'Data Type'])
    
    fields_cis, fields_trs = trace['cis']['place_field_all'], trace['trs']['place_field_all']
    n_neuron = trace['cis']['n_neuron']
    maze_type = trace['maze_type']
    
    overlap_num = 0
    overlap_num_shuf = 0
    cis_num, trs_num = 0, 0
    
    idx = np.zeros((10, n_neuron), np.int64)
    for i in range(10):
        idx[i, :] = np.arange(n_neuron)
        np.random.shuffle(idx[i, :])
    
    for i in range(n_neuron):
        cis_num += len(fields_cis[i].keys())
        trs_num += len(fields_trs[i].keys())
        
        for k1 in fields_cis[i].keys():
            is_matched = False
            field1 = fields_cis[i][k1]
            for k2 in fields_trs[i].keys():
                field2 = fields_trs[i][k2]
                overlap = np.intersect1d(field1, field2)
                
                if len(overlap)/len(field1) > overlap_thre or len(overlap)/len(field2) > overlap_thre:
                    is_matched = True
                    break
            
            if is_matched:
                overlap_num += 1
        
        shuf_num = 0
        for n in range(10):
            fields1 = fields_cis[i]
            fields2 = fields_trs[idx[n, i]]

            for k1 in fields1.keys():
                is_matched = False
                field1 = fields1[k1]
                for k2 in fields2.keys():
                    field2 = fields2[k2]
                    overlap = np.intersect1d(field1, field2)
                    if len(overlap)/len(field1) > overlap_thre or len(overlap)/len(field2) > overlap_thre:
                        is_matched = True
                        break
            
                if is_matched:
                    shuf_num += 1
        
        overlap_num_shuf += shuf_num/10
        
    return [cis_num, cis_num], [trs_num, trs_num], [(overlap_num/cis_num + overlap_num/trs_num)/2, (overlap_num_shuf/cis_num + overlap_num_shuf/trs_num)/2], ['Data', 'Shuffle']

# Fig0070
def CellNum_Reverse_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Cell Number', 'Place Cell Number', 'Direction'])

    return [trace['n_neuron'], trace['n_neuron']], [np.sum(trace['cis']['is_placecell']), np.sum(trace['trs']['is_placecell'])], ['cis', 'trs']

# Fig0070
def ActivationRateSpatialPosition_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    if trace['maze_type'] == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=['Position', 'Bin', 'Activation Rate', 'Is Perfect'])
    D = GetDMatrices(trace['maze_type'], 48)
    CP = cp.deepcopy(correct_paths[trace['maze_type']])
    
    is_perfect = trace['is_perfect']
    
    perfect_idx = np.where(is_perfect == 1)[0]
    error_idx = np.where(is_perfect == 0)[0]
    
    field_reg = trace['field_reg']
    dis = D[0, field_reg[:, 2]-1]
    fatherbins = S2F[field_reg[:, 2]-1]
    orderedbins = np.zeros_like(fatherbins)
    
    for i in range(orderedbins.shape[0]):
        orderedbins[i] = np.where(CP == fatherbins[i])[0][0]
    
    act_rate_per = np.mean(trace['activation_map'][perfect_idx], axis=0)
    act_rate_err = np.mean(trace['activation_map'][error_idx], axis=0)
    
    if len(perfect_idx) <= 5:
        act_rate_per = act_rate_per*np.nan
    if len(error_idx) <= 5:
        act_rate_err = act_rate_err*np.nan
    
    return (
        np.concatenate([dis, dis]),
        np.concatenate([orderedbins, orderedbins]),
        np.concatenate([act_rate_per, act_rate_err]),
        np.concatenate([np.repeat(1, act_rate_per.shape[0]), np.repeat(0, act_rate_err.shape[0])])
    )
    
from mylib.field.field_tracker import RegisteredField
# Fig0313
def ConditionalProb_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=[
        'Duration', 'Conditional Prob.', 'Conditional Recover Prob.',
        'Global Recover Prob.', 'Cumulative Prob.', 
        'Paradigm', 'On-Next Num', 'Off-Next Num'])

    
    if trace['paradigm'] == 'CrossMaze':
        retained_dur, prob, recover_prob, global_recover_prob, on_next_num, off_next_num = RegisteredField.conditional_prob(
            field_reg=trace['field_reg'],
            thre=4
        )
    
        res = np.nancumprod(prob)*100
        res[np.isnan(prob)] = np.nan
        res[0] = 100
    
        return (retained_dur, prob*100, recover_prob*100, global_recover_prob*100, 
                res, np.repeat(trace['paradigm'], res.shape[0]), 
                on_next_num, off_next_num)
    else:
        retained_dur_cis, prob_cis, recover_prob_cis, global_recover_prob_cis, on_next_num_cis, off_next_num_cis = RegisteredField.conditional_prob(
            field_reg=trace['cis']['field_reg'],
            thre=4
        )
        retained_dur_trs, prob_trs, recover_prob_trs, global_recover_prob_trs, on_next_num_trs, off_next_num_trs = RegisteredField.conditional_prob(
            field_reg=trace['trs']['field_reg'],
            thre=4
        )
    
        res_cis = np.nancumprod(prob_cis)*100
        res_cis[np.isnan(prob_cis)] = np.nan
        res_cis[0] = 100
        res_trs = np.nancumprod(prob_trs)*100
        res_trs[np.isnan(prob_trs)] = np.nan
        res_trs[0] = 100
    
        return (np.concatenate([retained_dur_cis, retained_dur_trs]), 
                np.concatenate([prob_cis*100, prob_trs*100]), 
                np.concatenate([recover_prob_cis*100, recover_prob_trs*100]), 
                np.concatenate([global_recover_prob_cis*100, global_recover_prob_trs*100]),
                np.concatenate([res_cis, res_trs]),
                np.concatenate([np.repeat(trace['paradigm'] + ' cis', res_cis.shape[0]), np.repeat(trace['paradigm'] + ' trs', res_trs.shape[0])]),
                np.concatenate([on_next_num_cis, on_next_num_trs]),
                np.concatenate([off_next_num_cis, off_next_num_trs]))


# Fig0321
from mylib.field.counter import calculate_superstable_fraction, calculate_survival_fraction
def Superstable_Fraction_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=[
        'Duration', 'Superstable Frac.', 'Threshold'])
    superstable_thre = np.arange(3, 51)
    
    dur = np.concatenate([trace['days'] for i in range(superstable_thre.shape[0])])
    superstable_frac = calculate_superstable_fraction(trace['field_reg'].T, thres=np.arange(3, 51)).flatten()
    #superstable_frac = np.concatenate([trace['Superstable Num'][i, :] for i in range(superstable_thre.shape[0])])
    thre = np.repeat(np.arange(3, 51), trace['days'].shape[0])
    
    return dur, superstable_frac, thre

def Superstable_Fraction_Data_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=[
        'Duration', 'Superstable Frac.', 'Threshold', 'Paradigm'])
    
    if trace['paradigm'] == 'CrossMaze':
        field_reg = trace['field_reg']
        num = np.where(np.isnan(field_reg), 0, 1)
        count = np.sum(num, axis=0)
        field_reg = field_reg[:, np.where(count == field_reg.shape[0])[0]]
        superstable_thre = np.arange(3, field_reg.shape[0]+1)
    
        dur = np.concatenate([np.arange(1, field_reg.shape[0]+1) for i in range(superstable_thre.shape[0])])
        superstable_frac = calculate_superstable_fraction(field_reg, thres=superstable_thre).flatten()
        #superstable_frac = np.concatenate([trace['Superstable Num'][i, :] for i in range(superstable_thre.shape[0])])
        thre = np.repeat(superstable_thre, field_reg.shape[0])
    
        return dur, superstable_frac, thre, np.repeat(trace['paradigm'], dur.shape[0])
    else:
        field_reg = trace['cis']['field_reg']
        num = np.where(np.isnan(field_reg), 0, 1)
        count = np.sum(num, axis=0)
        field_reg = field_reg[:, np.where(count == field_reg.shape[0])[0]]
        superstable_thre = np.arange(3, field_reg.shape[0]+1)
        
        dur = np.concatenate([np.arange(1, field_reg.shape[0]+1) for i in range(superstable_thre.shape[0])])
        superstable_frac_cis = calculate_superstable_fraction(trace['cis']['field_reg'], thres=superstable_thre).flatten()
        superstable_frac_trs = calculate_superstable_fraction(trace['trs']['field_reg'], thres=superstable_thre).flatten()
        thre = np.repeat(superstable_thre, field_reg.shape[0])
        
        return (
            np.concatenate([dur, dur]), 
            np.concatenate([superstable_frac_cis, superstable_frac_trs]), 
            np.concatenate([thre, thre]), 
            np.concatenate([np.repeat(trace['paradigm']+' cis', dur.shape[0]), 
                            np.repeat(trace['paradigm']+' trs', dur.shape[0])])
        )
        
        

# Fig0322
def SurvivalField_Fraction_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, 
                             check_variable=['Session Interval', 'Start Session', 
                                             'Survival Frac.'])
    
    survival_frac, start_sessions, training_day = calculate_survival_fraction(trace['field_reg'].T)
    survival_frac = survival_frac.flatten()
    start_sessions = start_sessions.flatten()
    training_day = training_day.flatten()
    
    # Remove nan
    idx = np.where((np.isnan(survival_frac) == False)&(np.isnan(start_sessions) == False)&(np.isnan(training_day) == False))[0]
    survival_frac = survival_frac[idx]
    start_sessions = start_sessions[idx]
    training_day = training_day[idx]
    
    return training_day, start_sessions, survival_frac

def SurvivalField_Fraction_Data_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
):
    VariablesInputErrorCheck(input_variable=variable_names, check_variable=[
        'Session Interval', 'Start Session', 'Survival Frac.', 'Paradigm'])
    
    if trace['paradigm'] == 'CrossMaze':
        field_reg = trace['field_reg']
        num = np.where(np.isnan(field_reg), 0, 1)
        count = np.sum(num, axis=0)
        field_reg = field_reg[:, np.where(count >= field_reg.shape[0]-4)[0]]

        survival_frac, start_sessions, training_day = calculate_survival_fraction(field_reg)
        survival_frac = survival_frac.flatten()
        start_sessions = start_sessions.flatten()
        training_day = training_day.flatten()
    
        # Remove nan
        idx = np.where((np.isnan(survival_frac) == False)&(np.isnan(start_sessions) == False)&(np.isnan(training_day) == False))[0]
        survival_frac = survival_frac[idx]
        start_sessions = start_sessions[idx]
        training_day = training_day[idx]
    
        return training_day, start_sessions, survival_frac, np.repeat(trace['paradigm'], survival_frac.shape[0])
    else:
        field_reg = trace['cis']['field_reg']
        num = np.where(np.isnan(field_reg), 0, 1)
        count = np.sum(num, axis=0)
        field_reg = field_reg[:, np.where(count == field_reg.shape[0])[0]]
        
        survival_frac_cis, start_sessions_cis, training_day_cis = calculate_survival_fraction(field_reg)
        survival_frac_cis = survival_frac_cis.flatten()
        start_sessions_cis = start_sessions_cis.flatten()
        training_day_cis = training_day_cis.flatten()
        
        # Remove nan
        idx = np.where((np.isnan(survival_frac_cis) == False)&
                       (np.isnan(start_sessions_cis) == False)&
                       (np.isnan(training_day_cis) == False))[0]
        survival_frac_cis = survival_frac_cis[idx]
        start_sessions_cis = start_sessions_cis[idx]
        training_day_cis = training_day_cis[idx]
        
        field_reg = trace['trs']['field_reg']
        num = np.where(np.isnan(field_reg), 0, 1)
        count = np.sum(num, axis=0)
        field_reg = field_reg[:, np.where(count == field_reg.shape[0])[0]]
        
        survival_frac_trs, start_sessions_trs, training_day_trs = calculate_survival_fraction(field_reg)
        survival_frac_trs = survival_frac_trs.flatten()
        start_sessions_trs = start_sessions_trs.flatten()
        training_day_trs = training_day_trs.flatten()
        
        # Remove nan
        idx = np.where((np.isnan(survival_frac_trs) == False)&
                       (np.isnan(start_sessions_trs) == False)&
                       (np.isnan(training_day_trs) == False))[0]
        survival_frac_trs = survival_frac_trs[idx]
        start_sessions_trs = start_sessions_trs[idx]
        training_day_trs = training_day_trs[idx]
        
        return (
            np.concatenate([training_day_cis, training_day_trs]),
            np.concatenate([start_sessions_cis, start_sessions_trs]),
            np.concatenate([survival_frac_cis, survival_frac_trs]),
            np.concatenate([np.repeat(trace['paradigm']+' cis', survival_frac_cis.shape[0]),
                            np.repeat(trace['paradigm']+' trs', survival_frac_trs.shape[0])])
        )

# Fig0318
from mylib.field.field_tracker import indept_test_for_evolution_events
def IndependentEvolution_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
    N = None
):
    VariablesInputErrorCheck(
        input_variable=variable_names,
        check_variable=['Training Session', 'Chi-Square Statistic', 'MI', 'Dimension', 'Pair Type', 'Pair Num', 'Paradigm'])
    
    if trace['paradigm'] == 'CrossMaze':
        if isinstance(N, dict):
            start_session, chi_stat, mi, pair_type, pair_num, dim = indept_test_for_evolution_events(
                trace['field_reg'],
                trace['field_ids'],
                N=N[('CrossMaze', trace['is_shuffle'], trace['maze_type'])]
            )
        else:
            start_session, chi_stat, mi, pair_type, pair_num, dim = indept_test_for_evolution_events(
                trace['field_reg'],
                trace['field_ids'],
                N=N
            )
        return (start_session, chi_stat, mi, dim, pair_type, pair_num, np.repeat(trace['paradigm'], start_session.shape[0]))
        
    else:
        if isinstance(N, dict):
            start_session_cis, chi_stat_cis, mi_cis, pair_type_cis, pair_num_cis, dim_cis = indept_test_for_evolution_events(
                trace['cis']['field_reg'],
                trace['cis']['field_ids'],
                N=N[(trace['paradigm']+' cis', trace['is_shuffle'], trace['maze_type'])]
            )
            start_session_trs, chi_stat_trs, mi_trs, pair_type_trs, pair_num_trs, dim_trs = indept_test_for_evolution_events(
                trace['trs']['field_reg'],
                trace['trs']['field_ids'],
                N=N[(trace['paradigm']+' trs', trace['is_shuffle'], trace['maze_type'])]
            )
        else:
            start_session_cis, chi_stat_cis, mi_cis, pair_type_cis, pair_num_cis, dim_cis = indept_test_for_evolution_events(
                trace['cis']['field_reg'],
                trace['cis']['field_ids'],
                N=N
            )
            start_session_trs, chi_stat_trs, mi_trs, pair_type_trs, pair_num_trs, dim_trs = indept_test_for_evolution_events(
                trace['trs']['field_reg'],
                trace['trs']['field_ids'],
                N=N
            )
    
    
        return (np.concatenate([start_session_cis, start_session_trs]),
                np.concatenate([chi_stat_cis, chi_stat_trs]),
                np.concatenate([mi_cis, mi_trs]),
                np.concatenate([dim_cis, dim_trs]),
                np.concatenate([pair_type_cis, pair_type_trs]),
                np.concatenate([pair_num_cis, pair_num_trs]),
                np.concatenate([np.repeat(trace['paradigm']+' cis', start_session_cis.shape[0]), 
                                np.repeat(trace['paradigm']+' trs', start_session_trs.shape[0])])
                )

from mylib.field.field_tracker import compute_joint_probability_matrix
def CoordinatedDrift_Interface(
    trace: dict,
    variable_names: list | None = None,
    spike_threshold: int | float = 10,
    return_item: str = "sib"
):
    VariablesInputErrorCheck(
        input_variable=variable_names,
        check_variable=['Training Session', 'delta-P', 'Dimension', 'Axis', 'Pair Type',
                        'Paradigm', 'X'])
    
    start_session = np.array([])
    dP = np.array([])
    dims = np.array([])
    axis = np.array([])
    xs = np.array([])
    pair_types = np.array([])
    direction = np.array([])
    
    if trace['paradigm'] == 'CrossMaze':
        for dim in range(2, 6):
            
            session, mat = compute_joint_probability_matrix(
                trace['field_reg'],
                trace['field_ids'],
                dim=dim,
                return_item='sib'
            )
            size = mat.shape[1]
            
            sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size)])
            detP = np.concatenate([mat[:, i, i] for i in range(size)])
            dimen = np.repeat(dim, sessions.shape[0])
            ax = np.repeat("IP axis", sessions.shape[0])
            x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size)])
            
            start_session = np.concatenate([start_session, sessions])
            dP = np.concatenate([dP, detP])
            dims = np.concatenate([dims, dimen])
            xs = np.concatenate([xs, x])
            axis = np.concatenate([axis, ax])
            pair_types = np.concatenate([pair_types, np.repeat('Sibling', sessions.shape[0])])
            
            sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size-1)])
            detP = np.concatenate([mat[:, i, size-i-2] for i in range(size-1)])
            dimen = np.repeat(dim, sessions.shape[0])
            ax = np.repeat("CP axis", sessions.shape[0])
            x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size-1)])
            
            start_session = np.concatenate([start_session, sessions])
            dP = np.concatenate([dP, detP])
            dims = np.concatenate([dims, dimen])
            xs = np.concatenate([xs, x])
            axis = np.concatenate([axis, ax])
            pair_types = np.concatenate([pair_types, np.repeat('Sibling', sessions.shape[0])])
            
            session, mat = compute_joint_probability_matrix(
                trace['field_reg'],
                trace['field_ids'],
                dim=dim,
                return_item='non'
            )
            
            sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size)])
            detP = np.concatenate([mat[:, i, i] for i in range(size)])
            dimen = np.repeat(dim, sessions.shape[0])
            ax = np.repeat("IP axis", sessions.shape[0])
            x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size)])
            
            start_session = np.concatenate([start_session, sessions])
            dP = np.concatenate([dP, detP])
            dims = np.concatenate([dims, dimen])
            xs = np.concatenate([xs, x])
            axis = np.concatenate([axis, ax])
            pair_types = np.concatenate([pair_types, np.repeat('Non-sibling', sessions.shape[0])])
            
            sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size-1)])
            detP = np.concatenate([mat[:, i, size-i-2] for i in range(size-1)])
            dimen = np.repeat(dim, sessions.shape[0])
            ax = np.repeat("CP axis", sessions.shape[0])
            x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size-1)])
            
            start_session = np.concatenate([start_session, sessions])
            dP = np.concatenate([dP, detP])
            dims = np.concatenate([dims, dimen])
            xs = np.concatenate([xs, x])
            axis = np.concatenate([axis, ax])
            pair_types = np.concatenate([pair_types, np.repeat('Non-sibling', sessions.shape[0])])
    
        return start_session, dP, dims, axis, pair_types, np.repeat("CrossMaze", xs.shape[0]), xs
    else:
        for k in ['cis', 'trs']:
            for dim in range(2, 6):
            
                session, mat = compute_joint_probability_matrix(
                    trace[k]['cis']['field_reg'],
                    trace[k]['cis']['field_ids'],
                    dim=dim,
                    return_item='sib'
                )
                size = mat.shape[1]
            
                sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size)])
                detP = np.concatenate([mat[:, i, i] for i in range(size)])
                dimen = np.repeat(dim, sessions.shape[0])
                ax = np.repeat("IP axis", sessions.shape[0])
                x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size)])
            
                start_session = np.concatenate([start_session, sessions])
                dP = np.concatenate([dP, detP])
                dims = np.concatenate([dims, dimen])
                xs = np.concatenate([xs, x])
                axis = np.concatenate([axis, ax])
                pair_types = np.concatenate([pair_types, np.repeat('Sibling', sessions.shape[0])])
                direction = np.concatenate([direction, np.repeat(trace['paradigm']+' '+k, sessions.shape[0])])
            
                sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size-1)])
                detP = np.concatenate([mat[:, i, size-i-2] for i in range(size-1)])
                dimen = np.repeat(dim, sessions.shape[0])
                ax = np.repeat("CP axis", sessions.shape[0])
                x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size-1)])
            
                start_session = np.concatenate([start_session, sessions])
                dP = np.concatenate([dP, detP])
                dims = np.concatenate([dims, dimen])
                xs = np.concatenate([xs, x])
                axis = np.concatenate([axis, ax])
                pair_types = np.concatenate([pair_types, np.repeat('Sibling', sessions.shape[0])])
                direction = np.concatenate([direction, np.repeat(trace['paradigm']+' '+k, sessions.shape[0])])
            
                session, mat = compute_joint_probability_matrix(
                    trace['trs']['field_reg'],
                    trace['trs']['field_ids'],
                    dim=dim,
                    return_item='non'
                )
            
                sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size)])
                detP = np.concatenate([mat[:, i, i] for i in range(size)])
                dimen = np.repeat(dim, sessions.shape[0])
                ax = np.repeat("IP axis", sessions.shape[0])
                x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size)])
            
                start_session = np.concatenate([start_session, sessions])
                dP = np.concatenate([dP, detP])
                dims = np.concatenate([dims, dimen])
                xs = np.concatenate([xs, x])
                axis = np.concatenate([axis, ax])
                pair_types = np.concatenate([pair_types, np.repeat('Non-sibling', sessions.shape[0])])
                direction = np.concatenate([direction, np.repeat(trace['paradigm']+' '+k, sessions.shape[0])])
            
                sessions = np.concatenate([np.arange(1, session.shape[0]+1) for i in range(size-1)])
                detP = np.concatenate([mat[:, i, size-i-2] for i in range(size-1)])
                dimen = np.repeat(dim, sessions.shape[0])
                ax = np.repeat("CP axis", sessions.shape[0])
                x = np.concatenate([np.repeat(i+1, session.shape[0]) for i in range(size-1)])
            
                start_session = np.concatenate([start_session, sessions])
                dP = np.concatenate([dP, detP])
                dims = np.concatenate([dims, dimen])
                xs = np.concatenate([xs, x])
                axis = np.concatenate([axis, ax])
                pair_types = np.concatenate([pair_types, np.repeat('Non-sibling', sessions.shape[0])])
                direction = np.concatenate([direction, np.repeat(trace['paradigm']+' '+k, sessions.shape[0])])
            
        return start_session, dP, dims, axis, pair_types, direction, xs
            