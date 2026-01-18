import numpy as np

def compute_P(
    Spikes_test: np.ndarray[np.int64],
    pext: np.ndarray[np.float64],
    pext_A: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """
    Spikes_test : (N, T_test) array of 0/1 integers
    pext        : (N, F) array of probabilities
    pext_A      : (F,) array of multipliers
    Returns:
        P : (F, T_test) array
    """
    ...