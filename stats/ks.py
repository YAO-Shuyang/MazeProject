from scipy.stats import ks_2samp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mylib.maze_utils3 import EqualPoisson, EqualPoissonFit, Normal, NormalFit, NegativeBinomial, NegativeBinomialFit, Weibull, WeibullFit
from scipy.stats import norm, poisson, weibull_min, lognorm, nbinom, gamma, kstest

# https://medium.com/@pabaldonedo/kolmogorov-smirnov-test-may-not-be-doing-what-you-think-when-parameters-are-estimated-from-the-data-2d5c3303a020

def compute_ks_dn(y, y_pred):
    return np.max(np.abs(y - y_pred))

def poisson_cdf(k, *args, **kwargs):
    return (poisson.cdf(k, *args, **kwargs) - poisson.cdf(0, *args, **kwargs)) / (1 - poisson.cdf(0, *args, **kwargs))

def poisson_pmf(k, *args, **kwargs):
    return poisson.pmf(k, *args, **kwargs) / (1 - poisson.cdf(0, *args, **kwargs))

def nbinom_cdf(k, *args, **kwargs):
    return (nbinom.cdf(k, *args, **kwargs) - nbinom.cdf(0, *args, **kwargs)) / (1 - nbinom.cdf(0, *args, **kwargs))

def nbinom_pmf(k, *args, **kwargs):
    return nbinom.pmf(k, *args, **kwargs) / (1 - nbinom.cdf(0, *args, **kwargs))

def norm_cdf(k, *args, **kwargs):
    return (norm.cdf(k, *args, **kwargs) - norm.cdf(0.5, *args, **kwargs)) / (1 - norm.cdf(0.5, *args, **kwargs))

def norm_pdf(k, *args, **kwargs):
    return norm.pdf(k, *args, **kwargs) / (1 - norm.cdf(0.5, *args, **kwargs))

def poisson_kstest(data, monte_carlo_times=10000, resample_size=1000):
    """
    Iterative KS test for Poisson distribution
    
    Returns
    -------
    D0 : float
        KS statistic of the original data
    pvalue : float
        p-value of the original data
    """
    size = min(resample_size, len(data))
    max_num = int(np.max(data))
    x = np.arange(1, max_num+1)
    prob = np.histogram(data, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]
    # Fit Poisson distribution
    lam = EqualPoissonFit(x, prob)
    P0 = poisson.rvs(lam, size=size)
    P0 = P0[np.where(P0 >= 1)[0]]
    D0 = compute_ks_dn(np.cumsum(prob), poisson_cdf(x, lam))
    
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    for i in tqdm(range(monte_carlo_times)):
        prob = np.histogram(np.random.choice(data, size=size), 
                         bins=max_num, range=(0.5, max_num+0.5), 
                         density=True)[0]
        lam = EqualPoissonFit(x, prob)
        Pt = poisson.rvs(lam, size=size)
        prob = np.histogram(Pt, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]
        Pt = Pt[np.where(Pt >= 1)[0]]
        D[i] = compute_ks_dn(np.cumsum(prob), poisson_cdf(x, lam))
        P = Pt

    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times

def normal_discrete_kstest(data, monte_carlo_times=10000, resample_size = 1000):
    """
    Iterative KS test for Normal distribution
    
    Returns
    -------
    D0 : float
        KS statistic
    pvalue : float
        p-value
    """
    size = min(resample_size, len(data))
    mean, sigma = norm.fit(data)
    max_num = int(np.max(data))
    x = np.arange(1, max_num+1)
    prob = np.histogram(data, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]
    P0 = norm.rvs(loc=mean, scale=sigma, size=size)
    P0 = ((P0+0.5) // 1).astype(np.int64)
    P0 = P0[np.where(P0 >= 1)[0]]
    D0 = compute_ks_dn(np.cumsum(prob), norm_cdf(x, mean, sigma))
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    
    for i in tqdm(range(monte_carlo_times)):
        mean, sigma = norm.fit(np.random.choice(data, size=size))
        Pt = norm.rvs(loc=mean, scale=sigma, size=size)
        prob = np.histogram(Pt, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]
        Pt = ((Pt+0.5) // 1).astype(np.int64)
        Pt = Pt[np.where(Pt >= 1)[0]]
        D[i] = compute_ks_dn(np.cumsum(prob), norm_cdf(x, mean, sigma))
        P = Pt

    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times

def nbinom_kstest(data, monte_carlo_times=10000, resample_size=1000):
    """
    Iterative KS test for Negative Binomial distribution
    
    Returns
    -------
    D0 : float
        KS statistic
    p : float
        p-value
    """
    size = min(resample_size, len(data))
    max_num = int(np.max(data))
    x = np.arange(1, max_num+1)
    prob = np.histogram(data, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]

    r, p = NegativeBinomialFit(x, prob)
    P0 = nbinom.rvs(n=r, p=p, size=size)
    D0 = compute_ks_dn(np.cumsum(prob), nbinom_cdf(np.arange(1, max_num+1), r, p))
        
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    for i in tqdm(range(monte_carlo_times)):
        prob = np.histogram(np.random.choice(data, size=size), bins=max_num, 
                            range=(0.5, max_num+0.5), density=True)[0]
        r, p = NegativeBinomialFit(x, prob)
        Pt = nbinom.rvs(n=r, p=p, size=size)
        prob = np.histogram(Pt, bins=max_num, range=(0.5, max_num+0.5), density=True)[0]
        D[i] = compute_ks_dn(np.cumsum(prob), nbinom_cdf(np.arange(1, max_num+1), r, p))
        P = Pt

    print(np.mean(D), np.std(D))
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times

def lognorm_kstest(data, monte_carlo_times=10000, resample_size=1000):
    """
    Iterative KS test for Log-Normal distribution
    
    Returns
    -------
    D0 : float
        KS statistic
    p : float
        p-value
    """
    size = min(len(data), resample_size)
    shape, locc, scale = lognorm.fit(data, floc = 0)
    P0 = lognorm.rvs(shape, loc=locc, scale=scale, size=size)
    P0 = ((P0+0.5) // 1).astype(np.int64)
    P0 = P0[np.where(P0 >= 1)[0]]
    D0 = kstest(data, lognorm.cdf, args=(shape, locc, scale))[0]
    
    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    for i in tqdm(range(monte_carlo_times)):
        shape, locc, scale = lognorm.fit(np.random.choice(data, size=size), floc = 0)
        Pt = lognorm.rvs(shape, loc=locc, scale=scale, size=size)
        Pt = ((Pt+0.5) // 1).astype(np.int64)
        Pt = Pt[np.where(Pt >= 1)[0]]
        D[i] = kstest(Pt, lognorm.cdf, args=(shape, locc, scale))[0]
        P = Pt
    
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times


def gamma_kstest(data, monte_carlo_times=10000, resample_size=1000, is_floc = False):
    """
    Iterative KS test for Gamma distribution
    
    Returns
    -------
    D0 : float
        KS statistic
    p : float
        p-value
    """
    size = min(len(data), resample_size)
    if is_floc:
        alpha, c, beta = gamma.fit(data, floc = 0)
    else:
        alpha, c, beta = gamma.fit(data)
    P0 = gamma.rvs(alpha, loc = c, scale=beta, size=size)
    P0 = ((P0+0.5) // 1).astype(np.int64)
    P0 = P0[np.where(P0 >= 1)[0]]
    D0 = kstest(data, gamma.cdf, args=(alpha, c, beta))[0]

    D = np.zeros(monte_carlo_times, dtype=np.float64)
    P = P0
    for i in tqdm(range(monte_carlo_times)):
        if is_floc:
            alpha, c, beta = gamma.fit(np.random.choice(data, size=size), floc = 0)
        else:
            alpha, c, beta = gamma.fit(np.random.choice(data, size=size))
        Pt = gamma.rvs(alpha, loc=c, scale=beta, size=size)
        Pt = ((Pt+0.5) // 1).astype(np.int64)
        Pt = Pt[np.where(Pt >= 1)[0]]
        D[i] = kstest(Pt, gamma.cdf, args=(alpha, c, beta))[0]
        P = Pt
        
    return D0, np.where(D0 < D)[0].shape[0] / monte_carlo_times



if __name__ == "__main__":
    import pickle
    
    with open(r"E:\Data\Cross_maze\10227\20230930\session 2\trace.pkl", "rb") as f:
        trace = pickle.load(f)
    
    field = trace['place_field_all']
    idx = np.where(trace['is_placecell'] == 1)[0]
    field_num = np.zeros(idx.shape[0])
    
    for i,index in enumerate(idx):
        field_num[i] = len(field[index].keys())
    
    
    print(poisson_kstest(field_num))