import numpy as np
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.linalg import pinv


def get_hausdorff_distance(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    X, Y may correlate to a best
    """
    
    T_ls = Y @ pinv(X)
    
    return directed_hausdorff(T_ls @ X, Y)