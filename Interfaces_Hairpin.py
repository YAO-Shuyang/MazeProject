from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *
from scipy.stats import poisson, norm, nbinom, kstest, ks_2samp, anderson, anderson_ksamp 
from scipy.stats import linregress, pearsonr, chisquare, ttest_1samp, ttest_ind
from scipy.stats import ttest_rel, levene

# Fig 0028 Field Distribution Statistic Test Against Poisson Distribution
def FieldDistributionStatistics_HairpinInterface(
    trace: dict,
    spike_threshold: int = 10,
    variable_names: list = None,
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['lambda', 'residual', 'ChiS Statistics', 'ChiS P-Value',
                                                                                'KS Statistics', 'KS P-Value', 'KS 2sample Statistics', 'KS 2sample P-Value',
                                                                                'AD Statistics 2sample Statistics', 'AD 2sample P-Value', 'Is Rejected', 'Direction'])
    # cis ======================================================================
    idx_cis = np.where(trace['cis']['is_placecell'] == 1)[0] if trace['cis']['maze_type'] == 0 else np.where(trace['cis']['is_placecell'] == 1)[0]
    place_field_all_cis = trace['cis']['place_field_all'] if trace['cis']['maze_type'] == 0 else trace['cis']['place_field_all']

    if len(idx_cis) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_cis = np.zeros(len(idx_cis), dtype = np.int64)
    for i, k in enumerate(idx_cis):
        field_number_cis[i] = len(place_field_all_cis[k].keys())
        
    xmax_cis = np.max(field_number_cis)
    # Fit Poisson Distribution
    a = plt.hist(field_number_cis, bins=xmax_cis, range=(0.5, xmax_cis+0.5))[0]
    plt.close()
    prob_cis = a / np.nansum(a)
    lam_cis = EqualPoissonFit(np.arange(1, xmax_cis+1), prob_cis)
    y_cis = EqualPoisson(np.arange(1, xmax_cis+1), l = lam_cis)
    y_cis = y_cis / np.sum(y_cis) * np.sum(a)
    abs_res_cis = np.mean(np.abs(a-y_cis))
    
    num_cis = len(field_number_cis)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for equal-rate Poisson Distribution (Cis): λ = {lam_cis}, residual = {abs_res_cis}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_cis, pvalue_cis = scipy.stats.kstest(field_number_cis, poisson.rvs(lam_cis, size=num_cis), alternative='two-sided')
    print("KS Test:", ks_sta_cis, pvalue_cis)
    ks_sta2_cis, pvalue2_cis = scipy.stats.ks_2samp(field_number_cis, poisson.rvs(lam_cis, size=num_cis), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_cis, pvalue2_cis)
    chista_cis, chip_cis = scipy.stats.chisquare(a, y_cis)
    print("Chi-Square Test:", chista_cis, chip_cis)
    adsta_cis, c_cis, adp_cis = scipy.stats.anderson_ksamp([field_number_cis, poisson.rvs(lam_cis, size=num_cis)])
    print("Anderson-Darling Test 2 sample:", adsta_cis, c_cis, adp_cis, end='\n\n')
    
    is_rejected_cis = pvalue_cis < 0.05 or pvalue2_cis < 0.05
    
    # Trans =================================================================================================================
    idx_trs = np.where(trace['trs']['is_placecell'] == 1)[0] if trace['trs']['maze_type'] == 0 else np.where(trace['trs']['is_placecell'] == 1)[0]
    place_field_all_trs = trace['trs']['place_field_all'] if trace['trs']['maze_type'] == 0 else trace['trs']['place_field_all']

    if len(idx_trs) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_trs = np.zeros(len(idx_trs), dtype = np.int64)
    for i, k in enumerate(idx_trs):
        field_number_trs[i] = len(place_field_all_trs[k].keys())
        
    xmax_trs = np.max(field_number_trs)
    # Fit Poisson Distribution
    a = plt.hist(field_number_trs, bins=xmax_trs, range=(0.5, xmax_trs+0.5))[0]
    plt.close()
    prob_trs = a / np.nansum(a)
    lam_trs = EqualPoissonFit(np.arange(1, xmax_trs+1), prob_trs)
    y_trs = EqualPoisson(np.arange(1, xmax_trs+1), l = lam_trs)
    y_trs = y_trs / np.sum(y_trs) * np.sum(a)
    abs_res_trs = np.mean(np.abs(a-y_trs))
    
    num_trs = len(field_number_trs)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for equal-rate Poisson Distribution (Trans): λ = {lam_trs}, residual = {abs_res_trs}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_trs, pvalue_trs = scipy.stats.kstest(field_number_trs, poisson.rvs(lam_trs, size=num_trs), alternative='two-sided')
    print("KS Test:", ks_sta_trs, pvalue_trs)
    ks_sta2_trs, pvalue2_trs = scipy.stats.ks_2samp(field_number_trs, poisson.rvs(lam_trs, size=num_trs), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_trs, pvalue2_trs)
    chista_trs, chip_trs = scipy.stats.chisquare(a, y_trs)
    print("Chi-Square Test:", chista_trs, chip_trs)
    adsta_trs, c_trs, adp_trs = scipy.stats.anderson_ksamp([field_number_trs, poisson.rvs(lam_trs, size=num_trs)])
    print("Anderson-Darling Test 2 sample:", adsta_trs, c_trs, adp_trs, end='\n\n')
    
    is_rejected_trs = pvalue_trs < 0.05 or pvalue2_trs < 0.05
    
    return (np.array([lam_cis, lam_trs]), np.array([abs_res_cis, abs_res_trs]), np.array([chista_cis, chista_trs]), np.array([chip_cis, chip_trs]), np.array([ks_sta_cis, ks_sta_trs]), np.array([pvalue_cis, pvalue_trs]), 
            np.array([ks_sta2_cis, ks_sta2_trs]), np.array([pvalue2_cis, pvalue2_trs]), np.array([adsta_cis, adsta_trs]), np.array([adp_cis, adp_trs]), np.array([is_rejected_cis, is_rejected_trs]), np.array(['cis', 'trs']))
    
# Fig 0028 Field Distribution Statistic Test Against Normal Distribution
def FieldDistributionStatistics_AgainstNormal_HairpinInterface(
    trace: dict,
    spike_threshold: int = 10,
    variable_names: list = None,
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Mean', 'Sigma', 'residual', 'ChiS Statistics', 'ChiS P-Value',
                                                                                'KS Statistics', 'KS P-Value', 'KS 2sample Statistics', 'KS 2sample P-Value',
                                                                                'AD 2sample Statistics', 'AD 2sample P-Value', 'Is Rejected', 'Direction'])
    # cis ======================================================================
    idx_cis = np.where(trace['cis']['is_placecell'] == 1)[0] if trace['cis']['maze_type'] == 0 else np.where(trace['cis']['is_placecell'] == 1)[0]
    place_field_all_cis = trace['cis']['place_field_all'] if trace['cis']['maze_type'] == 0 else trace['cis']['place_field_all']

    if len(idx_cis) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_cis = np.zeros(len(idx_cis), dtype = np.int64)
    for i, k in enumerate(idx_cis):
        field_number_cis[i] = len(place_field_all_cis[k].keys())
        
    xmax_cis = np.max(field_number_cis)
    # Fit Poisson Distribution
    a = plt.hist(field_number_cis, bins=xmax_cis, range=(0.5, xmax_cis+0.5))[0]
    plt.close()
    prob_cis = a / np.nansum(a)
    sigma_cis, miu_cis = NormalFit(np.arange(1,xmax_cis+1), prob_cis)
    y_cis = Normal(np.arange(1,xmax_cis+1), sigma_cis, miu_cis)    
    y_cis = y_cis / np.sum(y_cis) * np.sum(a)
    abs_res_cis = np.mean(np.abs(a-y_cis))
    
    num_cis = len(field_number_cis)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for Normal Distribution (cis): μ = {miu_cis}, sigma = {sigma_cis}, residual = {abs_res_cis}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_cis, pvalue_cis = scipy.stats.kstest(field_number_cis, norm.rvs(loc=miu_cis, scale=sigma_cis, size=num_cis), alternative='two-sided')
    print("KS Test:", ks_sta_cis, pvalue_cis)
    ks_sta2_cis, pvalue2_cis = scipy.stats.ks_2samp(field_number_cis, norm.rvs(loc=miu_cis, scale=sigma_cis, size=num_cis), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_cis, pvalue2_cis)
    chista_cis, chip_cis = scipy.stats.chisquare(a, y_cis)
    print("Chi-Square Test:", chista_cis, chip_cis)
    adsta_cis, c_cis, adp_cis = scipy.stats.anderson_ksamp([field_number_cis, norm.rvs(loc=miu_cis, scale=sigma_cis, size=num_cis)])
    print("Anderson-Darling Test 2 sample:", adsta_cis, c_cis, adp_cis, end='\n\n')
    
    is_rejected_cis = pvalue_cis < 0.05 or pvalue2_cis < 0.05
    
    # Trans =================================================================================================================
    idx_trs = np.where(trace['trs']['is_placecell'] == 1)[0] if trace['trs']['maze_type'] == 0 else np.where(trace['trs']['is_placecell'] == 1)[0]
    place_field_all_trs = trace['trs']['place_field_all'] if trace['trs']['maze_type'] == 0 else trace['trs']['place_field_all']

    if len(idx_trs) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_trs = np.zeros(len(idx_trs), dtype = np.int64)
    for i, k in enumerate(idx_trs):
        field_number_trs[i] = len(place_field_all_trs[k].keys())
        
    xmax_trs = np.max(field_number_trs)
    # Fit Poisson Distribution
    a = plt.hist(field_number_trs, bins=xmax_trs, range=(0.5, xmax_trs+0.5))[0]
    plt.close()
    prob_trs = a / np.nansum(a)
    sigma_trs, miu_trs = NormalFit(np.arange(1,xmax_trs+1), prob_trs)
    y_trs = Normal(np.arange(1,xmax_trs+1), sigma_trs, miu_trs)    
    y_trs = y_trs / np.sum(y_trs) * np.sum(a)
    abs_res_trs = np.mean(np.abs(a-y_trs))
    
    num_trs = len(field_number_trs)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for Normal Distribution (trs): μ = {miu_trs}, sigma = {sigma_trs}, residual = {abs_res_trs}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_trs, pvalue_trs = scipy.stats.kstest(field_number_trs, norm.rvs(loc=miu_trs, scale=sigma_trs, size=num_trs), alternative='two-sided')
    print("KS Test:", ks_sta_trs, pvalue_trs)
    ks_sta2_trs, pvalue2_trs = scipy.stats.ks_2samp(field_number_trs, norm.rvs(loc=miu_trs, scale=sigma_trs, size=num_trs), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_trs, pvalue2_trs)
    chista_trs, chip_trs = scipy.stats.chisquare(a, y_trs)
    print("Chi-Square Test:", chista_trs, chip_trs)
    adsta_trs, c_trs, adp_trs = scipy.stats.anderson_ksamp([field_number_trs, norm.rvs(loc=miu_trs, scale=sigma_trs, size=num_trs)])
    print("Anderson-Darling Test 2 sample:", adsta_trs, c_trs, adp_trs, end='\n\n')
    
    is_rejected_trs = pvalue_trs < 0.05 or pvalue2_trs < 0.05
    
    return (np.array([miu_cis, miu_trs]), np.array([sigma_cis, sigma_trs]), np.array([abs_res_cis, abs_res_trs]), np.array([chista_cis, chista_trs]), np.array([chip_cis, chip_trs]), np.array([ks_sta_cis, ks_sta_trs]), np.array([pvalue_cis, pvalue_trs]), 
            np.array([ks_sta2_cis, ks_sta2_trs]), np.array([pvalue2_cis, pvalue2_trs]), np.array([adsta_cis, adsta_trs]), np.array([adp_cis, adp_trs]), np.array([is_rejected_cis, is_rejected_trs]), np.array(['cis', 'trs']))

# Fig 0028 Field Distribution Statistic Test Against Negative Binomial Distribution
def FieldDistributionStatistics_AgainstNegativeBinomial_HairpinInterface(
    trace: dict,
    spike_threshold: int = 10,
    variable_names: list = None,
    is_placecell: bool = True
):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['r', 'p', 'residual', 'ChiS Statistics', 'ChiS P-Value',
                                                                                'KS Statistics', 'KS P-Value', 'KS 2sample Statistics', 'KS 2sample P-Value',
                                                                                'AD 2sample Statistics', 'AD 2sample P-Value', 'Is Rejected', 'Direction'])
    
    # cis ======================================================================
    idx_cis = np.where(trace['cis']['is_placecell'] == 1)[0] if trace['cis']['maze_type'] == 0 else np.where(trace['cis']['is_placecell'] == 1)[0]
    place_field_all_cis = trace['cis']['place_field_all'] if trace['cis']['maze_type'] == 0 else trace['cis']['place_field_all']

    if len(idx_cis) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_cis = np.zeros(len(idx_cis), dtype = np.int64)
    for i, k in enumerate(idx_cis):
        field_number_cis[i] = len(place_field_all_cis[k].keys())
        
    xmax_cis = np.max(field_number_cis)
    # Fit Poisson Distribution
    a = plt.hist(field_number_cis, bins=xmax_cis, range=(0.5, xmax_cis+0.5))[0]
    plt.close()
    prob_cis = a / np.nansum(a)
    r_cis, p_cis = NegativeBinomialFit(np.arange(1,xmax_cis+1), prob_cis)
    y_cis = NegativeBinomial(np.arange(1,xmax_cis+1), r_cis, p_cis)    
    y_cis = y_cis / np.sum(y_cis) * np.sum(a)
    abs_res_cis = np.mean(np.abs(a-y_cis))
    
    num_cis = len(field_number_cis)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for Negative Binomial Distribution (cis): p = {p_cis}, r = {r_cis}, residual = {abs_res_cis}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_cis, pvalue_cis = scipy.stats.kstest(field_number_cis, nbinom.rvs(n=r_cis, p=p_cis, size=num_cis), alternative='two-sided')
    print("KS Test:", ks_sta_cis, pvalue_cis)
    ks_sta2_cis, pvalue2_cis = scipy.stats.ks_2samp(field_number_cis, nbinom.rvs(n=r_cis, p=p_cis, size=num_cis), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_cis, pvalue2_cis)
    chista_cis, chip_cis = scipy.stats.chisquare(a, y_cis)
    print("Chi-Square Test:", chista_cis, chip_cis)
    adsta_cis, c_cis, adp_cis = scipy.stats.anderson_ksamp([field_number_cis, nbinom.rvs(n=r_cis, p=p_cis, size=num_cis)])
    print("Anderson-Darling Test 2 sample:", adsta_cis, c_cis, adp_cis, end='\n\n')
    
    is_rejected_cis = pvalue_cis < 0.05 or pvalue2_cis < 0.05
    
    # Trans =================================================================================================================
    idx_trs = np.where(trace['trs']['is_placecell'] == 1)[0] if trace['trs']['maze_type'] == 0 else np.where(trace['trs']['is_placecell'] == 1)[0]
    place_field_all_trs = trace['trs']['place_field_all'] if trace['trs']['maze_type'] == 0 else trace['trs']['place_field_all']

    if len(idx_trs) < 100:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    field_number_trs = np.zeros(len(idx_trs), dtype = np.int64)
    for i, k in enumerate(idx_trs):
        field_number_trs[i] = len(place_field_all_trs[k].keys())
        
    xmax_trs = np.max(field_number_trs)
    # Fit Poisson Distribution
    a = plt.hist(field_number_trs, bins=xmax_trs, range=(0.5, xmax_trs+0.5))[0]
    plt.close()
    prob_trs = a / np.nansum(a)
    r_trs, p_trs = NegativeBinomialFit(np.arange(1,xmax_trs+1), prob_trs)
    y_trs = NegativeBinomial(np.arange(1,xmax_trs+1), r_trs, p_trs)    
    y_trs = y_trs / np.sum(y_trs) * np.sum(a)
    abs_res_trs = np.mean(np.abs(a-y_trs))
    
    num_trs = len(field_number_trs)
    
    # Kolmogorov-Smirnov Test for Poisson Distribution
    print(f"\nTest for Negative Binomial Distribution (trs): p = {p_trs}, r = {r_trs}, residual = {abs_res_trs}")
    print(trace['MiceID'], trace['date'], 'Maze ', trace['maze_type'])
    ks_sta_trs, pvalue_trs = scipy.stats.kstest(field_number_trs, nbinom.rvs(n=r_trs, p=p_trs, size=num_trs), alternative='two-sided')
    print("KS Test:", ks_sta_trs, pvalue_trs)
    ks_sta2_trs, pvalue2_trs = scipy.stats.ks_2samp(field_number_trs, nbinom.rvs(n=r_trs, p=p_trs, size=num_trs), alternative='two-sided')
    print("KS Test 2 sample:", ks_sta2_trs, pvalue2_trs)
    chista_trs, chip_trs = scipy.stats.chisquare(a, y_trs)
    print("Chi-Square Test:", chista_trs, chip_trs)
    adsta_trs, c_trs, adp_trs = scipy.stats.anderson_ksamp([field_number_trs, nbinom.rvs(n=r_trs, p=p_trs, size=num_trs)])
    print("Anderson-Darling Test 2 sample:", adsta_trs, c_trs, adp_trs, end='\n\n')
    
    is_rejected_trs = pvalue_trs < 0.05 or pvalue2_trs < 0.05
    
    return (np.array([r_cis, r_trs]), np.array([p_cis, p_trs]), np.array([abs_res_cis, abs_res_trs]), np.array([chista_cis, chista_trs]), np.array([chip_cis, chip_trs]), np.array([ks_sta_cis, ks_sta_trs]), np.array([pvalue_cis, pvalue_trs]), 
            np.array([ks_sta2_cis, ks_sta2_trs]), np.array([pvalue2_cis, pvalue2_trs]), np.array([adsta_cis, adsta_trs]), np.array([adp_cis, adp_trs]), np.array([is_rejected_cis, is_rejected_trs]), np.array(['cis', 'trs']))