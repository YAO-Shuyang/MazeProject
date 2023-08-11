import numpy as np
from scipy.interpolate import interp1d

def interpolated_smooth(x: list | np.ndarray, y: list | np.ndarray, kind: str = 'linear', insert_num: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    interpolated_smooth: get a set of interpolated value

    Parameters
    ----------
    x : list | np.ndarray
        x data
    y : list | np.ndarray
        y data
    kind : str, optional
        The method to interpolate, by default 'linear'
    insert_num : int, optional
        The number of values to be interpolated, by default 1000

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The values after interpolating.
    """
    
    new_x_data = np.linspace(min(x), max(x), insert_num)

    if len(x)==1 or len(y)==1:
        return new_x_data, np.zeros(insert_num)

    # Create an interpolating function
    f = interp1d(x, y, kind=kind)

    return new_x_data, f(new_x_data)
