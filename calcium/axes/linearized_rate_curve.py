import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mylib.maze_utils3 import Clear_Axes, Gaussian
from mylib.maze_graph import NRG, correct_paths, incorrect_paths
from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d, gaussian
from mylib.calcium.smooth.interpolation import interpolated_smooth


def LinearizedRateMapAxes(
    ax: Axes,
    content: np.ndarray,
    maze_type: int,
    title: str="",
    is_display_max_only: bool=True,
    smooth_window_length: int = 40,
    sigma: float = 3,
    folder: float = 1,
    M: np.ndarray = None,
    smooth_kwarg: dict = {'kind':'slinear', 'insert_num':1000},
    **kwargs
) -> tuple[Axes, list]:
    """
    LinearizedRateMapAxes: to plot a linearized figure upon the loc-time curve

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to contain the linearized rate map.
    content : np.ndarray
        The content (linearized events rate map) to be plotted.
    maze_type : int
        The maze type.
    title : str, optional
        The title to be put upon this figure, by default ""
    is_display_max_only : bool, optional
        If only show the max value at the y ticks, by default True
    smooth_window_length : int, optional
        The window length to smooth the content, by default 6
    M : np.ndarray, optional
        The smooth matrix, by default None

    Returns
    -------
    tuple[Axes, list]
        The axes and a list contained the lines (and can be removed)
    """
    assert content.shape[0] == 144

    ax = Clear_Axes(ax, close_spines=['top', 'right', 'bottom'], ifyticks=True)
    correct_path = correct_paths[maze_type]
    
    corr_content = content[correct_path-1]

    smooth_x, smooth_content = interpolated_smooth(np.arange(1, corr_content.shape[0]+1), corr_content, **smooth_kwarg)

    if M is None:
        M = gaussian_smooth_matrix1d(smooth_kwarg['insert_num'], window = smooth_window_length, sigma=sigma, folder=folder)

    smooth_content = np.dot(M, smooth_content)

    y_max = (int(np.nanmax(content)*100))/100

    a = ax.plot(smooth_x, smooth_content, color = 'k', **kwargs)
    #a = ax.plot(np.arange(1, corr_content.shape[0]+1), corr_content, color = 'k')

    ax.set_ylabel("Events Rate / Hz")
    ax.axis([0, 145, -y_max/10, y_max])
    if is_display_max_only:
        ax.set_yticks([0, y_max])
    ax.set_title(title)

    return ax, a
    

if __name__ == '__main__':
    import pickle

    with open(r"G:\YSY\Cross_maze\11095\20220828\session 2\trace.pkl", 'rb') as handle:
        trace = pickle.load(handle)
    
    fig = plt.figure(figsize=(4,1.5))
    LinearizedRateMapAxes(
        ax = plt.axes(),
        content = trace['old_map_clear'][1],
        maze_type = trace['maze_type'],
        smooth_window_length = 20,
        folder = 0.1
    )
    plt.tight_layout()
    plt.show()