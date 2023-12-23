from scipy.stats import gamma
import numpy as np

import matplotlib.pyplot as plt


def gamma_poisson_pdf(alpha, beta, max_lam: float = 100.0, min_lam: float = 0.001, steps: int = 100000, output: str = 'range'):
    """gamma_poisson_pdf: Compute the pdf of a Gamma-Poisson distribution

    Parameters
    ----------
    alpha : float
        The parameter of the Gamma distribution
    beta : float
        The parameter of the Poisson distribution
    max_lam : float, optional
        The upper bound, by default 100.0
    min_lam : float, optional
        The lower bound, by default 0.001
    steps : int, optional
        The number of steps, by default 100000
    output : str, optional
        The output, by default 'range'. Options are 'range', 'bound', 'pdf' or 'cdf'

    Returns
    -------
    lam : numpy.ndarray
        The range of the output

    Raises
    ------
    ValueError
        If output is not 'range', 'bound', 'pdf' or 'cdf'
    """
    lam = np.linspace(min_lam, max_lam, steps)
    
    pdf = gamma.pdf(lam, a=alpha, scale=1/beta)
    cumulative_sum = np.cumsum(pdf)*(max_lam-min_lam)/(steps-1)
    
    try:
        low_bound = lam[np.where(cumulative_sum < 0.025)[0][-1]]
    except:
        low_bound = lam[0]
        
    try:
        hig_bound = lam[np.where(cumulative_sum > 0.975)[0][0]]
    except:
        hig_bound = lam[-1]

    if output == 'range':
        return hig_bound - low_bound
    elif output == 'bound':
        return low_bound, hig_bound
    elif output == 'pdf':
        return lam, pdf
    elif output == 'cdf':
        return lam, cumulative_sum
    else:
        raise ValueError('output must be "range", "bound", "pdf" or "cdf"')