import numpy as np
from numba import jit
import copy as cp
from scipy.stats import ttest_1samp
from scipy.stats import binned_statistic_2d, binned_statistic, chi2_contingency
from sklearn.metrics import mutual_info_score
#
def bi_ecdf(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    t2: float
) -> float:
    """bi_ecdf: Bivariate Empirical Cumulative Distribution Function

    Parameters
    ----------
    x : np.ndarray
        Dimension 1 of the observation.
    y : np.ndarray
        Dimension 2 of the observation.
    t1 : float
        Threshold for the dimension 1.
    t2 : float
        Threshold for the dimension 2.

    Returns
    -------
    float
        Empirical cumulative distribution function at (t1, t2).
        
    Notes
    -----
    Fn(t1, t2) = sum Indicator(xi <= t1, yi <= t2)/n, while n is the total number of observation
    """
    return np.where((x<=t1) & (y<=t2))[0].shape[0]/x.shape[0]

def si_ecdf(
    x: np.ndarray,
    t: float
) -> float:
    """si_ecdf: Empirical Cumulative Distribution Function

    Parameters
    ----------
    x : np.ndarray
        Observation.
    t : float
        Threshold.

    Returns
    -------
    float
        Empirical cumulative distribution function at t.
        
    Notes
    -----
    F(t) = sum Indicator(xi <= t)/n, while n is the total number of data
    """
    return np.where(x<=t)[0].shape[0]/x.shape[0]

def supremum(
    diff: np.ndarray
) -> float:
    """supremum: the definition of the test statistic.

    Parameters
    ----------
    diff : np.ndarray
        The difference between Fn(t1, t2) and Fx(t1)*Fy(t2)

    Returns
    -------
    float
        The supremum of the |Fn(t1, t2) - Fx(t1)*Fy(t2)|
    """
    return np.max(np.abs(diff))

@jit(nopython=True)
def test_stats(
    x: np.ndarray,
    y: np.ndarray,
    x_bin_num: int = 1000,
    y_bin_num: int = 1000
):
    """
    xbins = np.linspace(np.min(x), np.max(x), x_bin_num+1)
    ybins = np.linspace(np.min(y), np.max(y), y_bin_num+1)
        
    diff = np.zeros((xbins.shape[0], ybins.shape[0]), np.float64)
    for j in range(xbins.shape[0]):
        for k in range(ybins.shape[0]):
            diff[j, k] = np.where((x<=xbins[j]) & (y<=ybins[k]))[0].shape[0]/x.shape[0] - np.where(x<=xbins[j])[0].shape[0]/x.shape[0]*np.where(y<=ybins[k])[0].shape[0]/y.shape[0]
            
    return np.max(np.abs(diff))
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    xbins = np.linspace(x_min, x_max, x_bin_num + 1)
    ybins = np.linspace(y_min, y_max, y_bin_num + 1)
    
    x_indices = np.searchsorted(xbins, x)
    y_indices = np.searchsorted(ybins, y)

    x_total = x.shape[0]
    y_total = y.shape[0]

    diff = np.zeros((x_bin_num + 1, y_bin_num + 1), np.float64)

    for j in range(x_bin_num):
        for k in range(y_bin_num):
            x_indices_j = x_indices[j]
            y_indices_k = y_indices[k]

            count_xy = len(np.where((x_indices_j <= x_indices) & (y_indices_k <= y_indices))[0])
            count_x = len(np.where(x_indices_j <= x_indices)[0])
            count_y = len(np.where(y_indices_k <= y_indices)[0])

            diff[j, k] = count_xy / x_total - (count_x / x_total) * (count_y / y_total)

    return np.max(np.abs(diff))


from tqdm import tqdm
def monte_carlo(
    simu_times: int = 1000,
    simu_len: int = 1000,
    x_bin_num: int = 100,
    y_bin_num: int = 100
):
    simu_statistics = np.zeros(simu_times, np.float64)
    
    for i in tqdm(range(simu_times)):
        x = np.random.rand(simu_len)
        y = np.random.rand(simu_len)
        
        simu_statistics[i] = test_stats(x, y, x_bin_num, y_bin_num)
        
    return simu_statistics


def permutation(
    x: np.ndarray,
    y: np.ndarray,
    x_bin_num: int = 1000,
    y_bin_num: int = 1000,
    simu_times: int = 1000
):
    simu_statistics = np.zeros(simu_times, np.float64)
    
    for i in tqdm(range(simu_times)):
        np.random.shuffle(x)
        
        simu_statistics[i] = test_stats(x, y, x_bin_num, y_bin_num)
        
    return simu_statistics

def indeptest(x, y, simu_times=1000, x_bin_num=100, y_bin_num=100, shuffle_method: str = "monte_carlo"):
    """indeptest
    
    See "Distribution Free Tests of Independence Based on the Sample Distribution Function. J. R. Blum, J. Kiefer and M. Rosenblatt, 1961."
    https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-32/issue-2/Distribution-Free-Tests-of-Independence-Based-on-the-Sample-Distribution/10.1214/aoms/1177705055.full
    
    According to R document of function indeptest https://search.r-project.org/CRAN/refmans/robusTest/html/indeptest.html

    Parameters
    ----------
    x : _type_
        Dimension 1 of observation
    y : _type_
        Dimension 2 of observation
    simu_times : int, optional
        Times of Monte-Carlo simulation, by default 1000
    x_bin_num : int, optional
        The number of bins to divide the range of x observation, by default 1000
    y_bin_num : int, optional
        The number of bins to divide the range of y observation, by default 1000
    shuffle_class : str, optional
        The method of shuffling, by default "monte_carlo".
        Options: "monte_carlo" or "permutation"

    Returns
    -------
    test_statistics: float
        The test statistics of independent task.
    res: scipy.stats._stats_py.TtestResult
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x ({x.shape}) and y ({y.shape}) must have the same length!")
    
    
    test_statistics = test_stats(x, y, x_bin_num, y_bin_num)
    
    if shuffle_method == "monte_carlo":
        print("It needs several minutes to perform monte-carlo simulation.")
        simu_statistics = monte_carlo(simu_times, x.shape[0], 100, 100)
    elif shuffle_method == "permutation":
        print("It needs several minutes to perform permutation.")
        simu_statistics = permutation(cp.deepcopy(x), cp.deepcopy(y), x_bin_num, y_bin_num, simu_times)
    else:
        raise ValueError(f"shuffle_method must be 'monte_carlo' or 'permutation', not {shuffle_method}")
    
    return test_statistics, np.where(simu_statistics >= test_statistics)[0].shape[0]/simu_times


def indept_field_properties(
    real_distribution: np.ndarray,
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray,
    n_bin = 20
) -> tuple:
    """indept_field_properties: Test whether the properties of sibling fields are independent

    Parameters
    ----------
    real_distribution : np.ndarray, (N, )
        The real distribution (estimated) of this field property
    X_pairs : np.ndarray, shape (n1, 2)
        The sample to calculate the join probability of sibling fields' data.
    Y_pairs : np.ndarray, shape (n2, 2)
        The sample to calculate the join probability of  non-sibling fields' data.
    n_bin : int, optional
        The number of bins to divide the range, by default 20
        
    Returns
    -------
    tuple[float, float]
    The first is the chi2 statistics of sibling fields pairs.
    The second is the chi2 statistics of non-sibling fields pairs.
    """
    # remove nan
    real_distribution = real_distribution[np.isnan(real_distribution) == False]
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    
    _range = [np.min(real_distribution), np.max(real_distribution)+0.0001]
    P, _, _ = binned_statistic(
        x=real_distribution,
        values=None,
        statistic="count",
        bins=n_bin,
        range=_range
    )
    P = P / real_distribution.shape[0] # Calculate the probability mass function
    
    observed_joint_freq_X, _, _, binnumber_X = binned_statistic_2d(
        x=X_pairs[:, 0],
        y=X_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    observed_joint_freq_Y, _, _, binnumber_Y = binned_statistic_2d(
        x=Y_pairs[:, 0],
        y=Y_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    joint_p = np.outer(P, P)
    expected_joint_freq_X = joint_p*len(X_pairs) + 1
    expected_joint_freq_Y = joint_p*len(Y_pairs) + 1
    
    observed_joint_freq_X = observed_joint_freq_X + 1
    observed_joint_freq_Y = observed_joint_freq_Y + 1

    return chi2_contingency(observed_joint_freq_X, expected_joint_freq_X)[0], chi2_contingency(observed_joint_freq_Y, expected_joint_freq_Y)[0], len(X_pairs), len(Y_pairs)


def indept_field_properties_mutual_info(
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray,
    n_bin: int = 40
) -> tuple:
    """indept_field_properties: Test whether the properties of sibling fields are independent

    Parameters
    ----------
    X_pairs : np.ndarray, shape (n1, 2)
        The sample to calculate the join probability of sibling fields' data.
    Y_pairs : np.ndarray, shape (n2, 2)
        The sample to calculate the join probability of  non-sibling fields' data.
        
    Returns
    -------
    tuple[float, float]
    The first is the mutual information of sibling fields.
    The second is the mutual information of non-sibling fields.
    """
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    _range = [np.min(X_pairs.flatten()), np.max(Y_pairs.flatten())+0.0001]

    _, _, _, binnumber_X = binned_statistic_2d(
            x=X_pairs[:, 0],
            y=X_pairs[:, 1],
            values=None,
            statistic="count",
            bins=[n_bin, n_bin],
            range=[_range, _range],
            expand_binnumbers=True
    )
    X_pairs = binnumber_X.T
        
    _, _, _, binnumber_Y = binned_statistic_2d(
            x=Y_pairs[:, 0],
            y=Y_pairs[:, 1],
            values=None,
            statistic="count",
            bins=[n_bin, n_bin],
            range=[_range, _range],
            expand_binnumbers=True
    )
    Y_pairs = binnumber_Y.T
    print(X_pairs.shape, Y_pairs.shape, np.min(X_pairs), np.min(Y_pairs), np.max(X_pairs), np.max(Y_pairs))
        
    return mutual_info_score(X_pairs[:, 0], X_pairs[:, 1]), mutual_info_score(Y_pairs[:, 0], Y_pairs[:, 1]), len(X_pairs), len(Y_pairs)
    
def indept_field_evolution_chi2(
    real_distribution: np.ndarray,
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray,
    return_mat: bool = False
):
    _range = [0.5, int(np.max(real_distribution))+0.5]
    n_bin = int(np.max(real_distribution))
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    
    P = np.histogram(real_distribution, bins=n_bin, range=_range)[0] / real_distribution.shape[0]
    
    observed_joint_freq_X, _, _, binnumber_X = binned_statistic_2d(
        x=X_pairs[:, 0],
        y=X_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    observed_joint_freq_Y, _, _, binnumber_Y = binned_statistic_2d(
        x=Y_pairs[:, 0],
        y=Y_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    joint_p = np.outer(P, P)
    expected_joint_freq_X = joint_p*len(X_pairs) + 0.001
    expected_joint_freq_Y = joint_p*len(Y_pairs) + 0.001

    observed_joint_freq_X += 0.001
    observed_joint_freq_Y += 0.001

    if return_mat:
        return observed_joint_freq_X-0.001, observed_joint_freq_Y-0.001, expected_joint_freq_X-0.001
    else:
        return chi2_contingency(observed_joint_freq_X, expected_joint_freq_X)[0], chi2_contingency(observed_joint_freq_Y, expected_joint_freq_Y)[0], len(X_pairs), len(Y_pairs)

def indept_field_evolution_mutual_info(
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray
):
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    return mutual_info_score(X_pairs[:, 0], X_pairs[:, 1]), mutual_info_score(Y_pairs[:, 0], Y_pairs[:, 1]), len(X_pairs), len(Y_pairs)

def frobenius_norm(A: np.ndarray, B: np.ndarray):
    return np.sqrt(np.sum((A-B)**2))

# Coordination Index
def indept_field_evolution_CI(
    real_distribution: np.ndarray,
    X_pairs: np.ndarray,
    Y_pairs: np.ndarray,
    return_mat: bool = False
):
    _range = [0.5, int(np.max(real_distribution))+0.5]
    n_bin = int(np.max(real_distribution))
    X_pairs = X_pairs[np.where((np.isnan(X_pairs[:, 0]) == False) & (np.isnan(X_pairs[:, 1]) == False))[0], :]
    Y_pairs = Y_pairs[np.where((np.isnan(Y_pairs[:, 0]) == False) & (np.isnan(Y_pairs[:, 1]) == False))[0], :]
    
    P = np.histogram(real_distribution, bins=n_bin, range=_range, density=True)[0]
    
    observed_joint_freq_X, _, _, binnumber_X = binned_statistic_2d(
        x=X_pairs[:, 0],
        y=X_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    observed_joint_freq_Y, _, _, binnumber_Y = binned_statistic_2d(
        x=Y_pairs[:, 0],
        y=Y_pairs[:, 1],
        values=None,
        statistic="count",
        bins=[n_bin, n_bin],
        range=[_range, _range],
        expand_binnumbers=True
    )
    
    observed_joint_freq_X += 0.001
    observed_joint_freq_Y += 0.001
    
    p_sib, p_non = observed_joint_freq_X / np.sum(observed_joint_freq_X), observed_joint_freq_Y / np.sum(observed_joint_freq_Y)
    joint_p = np.outer(P, P)
    margi_p = np.zeros_like(joint_p)
    for i in range(len(margi_p)):
        margi_p[i, i] = P[i]
        
    print(frobenius_norm(p_sib, joint_p)/(frobenius_norm(p_sib, joint_p) + frobenius_norm(p_sib, margi_p)),
          frobenius_norm(p_non, joint_p)/(frobenius_norm(p_non, joint_p) + frobenius_norm(p_non, margi_p)))
        
    return (frobenius_norm(p_sib, joint_p)/(frobenius_norm(p_sib, joint_p) + frobenius_norm(p_sib, margi_p)), 
            frobenius_norm(p_non, joint_p)/(frobenius_norm(p_non, joint_p) + frobenius_norm(p_non, margi_p)), 
            len(X_pairs), len(Y_pairs))


if __name__ == "__main__":
    from mylib.statistic_test import mkdir, Clear_Axes, figpath, figdata
    # You can annotate this line if you cannot import mylib.
    import os
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, linregress
    
    # You can manually set a "figpath" to save the figures, and a "figdata" directory to save the results
    # in the forms of both PKL file or EXCEL sheet.
    
    code_id = "0061 - Independent Test (Blum et al., 1961)"
    loc = os.path.join(figpath, code_id)
    mkdir(loc)
    
    test_time = 41
    noise_amp = np.linspace(0.96, 1, test_time)
    pearson_corr = np.zeros(test_time, np.float64)
    pearson_pvalue = np.zeros(test_time, np.float64)
    
    linreg_slope = np.zeros(test_time, np.float64)
    linreg_rvalue = np.zeros(test_time, np.float64)
    linreg_pvalue = np.zeros(test_time, np.float64)
    
    indep_statistics = np.zeros(test_time, np.float64)
    indep_pvalue = np.zeros(test_time, np.float64)
    ttest_statistics = np.zeros(test_time, np.float64)
    
    # Create canvas to avoid too much memory use.
    fig = plt.figure(figsize = (4,4))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    # You can annotate this line if you cannot import mylib. Use ax = plt.axes() as an alternative.
    ax.set_aspect("equal")
    ax.axis([0,1,0,1])
    ax.plot([0,1], [0,1], ':', color='gray', linewidth=0.5)
    
    for i, n in enumerate(noise_amp):
        # Generate data with different degree of noise.
        x = np.random.randn(1000)
        y = x*(1-n) + np.rans
        
        # Comparing the indeptest with pearson correlation and linear regression to test its efficiency.
        pearson_corr[i], pearson_pvalue[i] = pearsonr(x, y)
        linreg_slope[i], intercept, linreg_rvalue[i], linreg_pvalue[i], _ = linregress(x, y)
        indep_statistics[i], tteststruct = indeptest(x, y)
        ttest_statistics[i], indep_pvalue[i] = tteststruct.statistic, tteststruct.pvalue
        
        print("Noise Amplitude: ", n)
        print(f"Pearson: {pearson_corr[i]}, {pearson_pvalue[i]}")
        print(f"Linreg: {linreg_slope[i]}, {linreg_pvalue[i]}")
        print(f"Independent: {indep_statistics[i]}, {indep_pvalue[i]}", end='\n\n')
        
        # Visualizing...
        b = ax.plot([0, 1], [intercept, intercept + linreg_slope[i]], color='orange', linewidth=0.5)
        a = ax.plot(x, y, 'o', markeredgewidth = 0, markersize = 1, color = "black")
        ax.set_title(f"Pearson: {round(pearson_corr[i], 2)}, {round(pearson_pvalue[i], 3)}\n"+
                     f"Linreg: {round(linreg_slope[i], 2)}, {round(linreg_pvalue[i], 3)}\n"+
                     f"Independ: {round(indep_statistics[i], 2)}, {round(indep_pvalue[i], 3)}\n")
        plt.tight_layout()
        plt.savefig(os.path.join(loc, f"Noise Amplitude - {n}.png"), dpi=600)
        plt.savefig(os.path.join(loc, f"Noise Amplitude - {n}.svg"), dpi=600)
        
        for j in a+b:
            j.remove()
    
    # Cleaning data dots and the regression-fitted line while keeping the diagonal gray dotted line.     
    plt.close()
        
    Data = pd.DataFrame(
        {
            "Noise Amplitude": noise_amp,
            "Pearson Correlation": pearson_corr,
            "Pearson P-value": pearson_pvalue,
            "Linear Regression Slope": linreg_slope,
            "Linear Regression P-value": linreg_pvalue,
            "Linear Regression R-value": linreg_rvalue,
            "Independent Test Statistic": indep_statistics,
            "Independent Test P-value": indep_pvalue,
            "1 Sample T-test Statistic": ttest_statistics
        }
    )
    
    # Save the data
    Data.to_excel(os.path.join(figdata, code_id+" [tail].xlsx"), index=False)
    
    with open(os.path.join(figdata, code_id+" [tail].pkl"), 'wb') as handle:
        pickle.dump(Data, handle)