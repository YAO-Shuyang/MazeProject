from mylib.maze_graph import Father2SonGraph, maze_graphs
from mylib.maze_utils3 import GetDMatrices
from mazepy.behav.graph import Graph
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import copy as cp
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


class FieldShiftMoniter:
    def __init__(self):
        self._pos = self._centers = None
        pass
    
    def _bin_to_xy(self, bin: np.ndarray) -> tuple[int, int]:
        return (bin-1)%48, (bin-1)//48
    
    def _fathers_to_sons(self, father_fields: np.ndarray) -> np.ndarray:
        return np.concatenate([Father2SonGraph[i] for i in father_fields]).astype(np.int64)
    
    def center_pos(self, smooth_maps: np.ndarray, father_fields: np.ndarray) -> np.ndarray:
        son_fields = self._fathers_to_sons(father_fields)
        sub_maps = smooth_maps[:, son_fields-1]
        rate_argmax = np.argmax(sub_maps, axis=1)
        
        pos = np.zeros((smooth_maps.shape[0], 2), dtype=np.float64)
        pos[:, 0], pos[:, 1] = self._bin_to_xy(rate_argmax+1)
        self._pos = pos
        self._centers = rate_argmax+1
        self._num = smooth_maps.shape[0]
        return pos
    
    def calc_shift_distance(self, maze_type: int):
        if self.pos is None:
            raise ValueError("Please call center_pos first.")
        
        Gm = cp.deepcopy(maze_graphs[(int(maze_type), 48)])
        print("    Establish Graph for maze...")
        G = Graph(48, 48, Graph=Gm)
        print("  Done.")
        
        shifts = np.zeros(self.centers.shape[0], dtype=np.float64)
        anchor = G.shortest_distance((self.pos[0, 0], self.pos[0, 1]), (0, 0))
        
        for i in range(self.centers.shape[0]):
            dx = G.shortest_distance((self.pos[i, 0], self.pos[i, 1]), (0, 0)) - anchor
            if dx > 0:
                dirc = 1
            elif dx < 0:
                dirc = -1
            else:
                dirc = 0
                
            shifts[i] = G.shortest_distance((self.pos[i, 0], self.pos[i, 1]), ())*dirc
        
        self._shifts = shifts
    
    @property
    def shifts(self):
        return self._shifts
    
    @property
    def pos(self):
        return self._pos
    
    @property
    def centers(self):
        return self._centers
    
    @staticmethod
    def get_shifts(
        maze_type: int,
        smooth_maps: np.ndarray,
        father_fields: np.ndarray
    ):
        monitor = FieldShiftMoniter()
        monitor.center_pos(smooth_maps=smooth_maps, father_fields=father_fields)
        monitor.calc_shift_distance(maze_type=maze_type)
        return monitor
    
    def add_field_background(self, ax: Axes, father_fields: np.ndarray, background_color: str ='gray', **background_kw) -> Axes:
        for i in father_fields:
            x, y = (i-1)%12*4-0.5, (i-1)//12*4-0.5
            ax.fill_betweenx(y=[y, y+4], x1=x, x2 = x+4, color = background_color, **background_kw)
        
        return ax
    
    @staticmethod
    def add_centers(
        ax: Axes,
        father_fields: np.ndarray,
        markersize: int = 3,
        colors: list = None,
        background_color: str ='gray',
        background_kw: dict = {},
        maze_type: int | None = None,
        smooth_maps: np.ndarray | None = None,
        **kwargs
    ):
        monitor = FieldShiftMoniter()
        monitor.center_pos(smooth_maps=smooth_maps, father_fields=father_fields)
        monitor.calc_shift_distance(maze_type=maze_type)
        
        if colors is None:
            colors = sns.color_palette("rainbow", len(monitor.centers))
        
        for i in range(len(monitor.centers)):
            ax.plot(x = [monitor.pos[i, 0]-0.5], y = [monitor.pos[i, 1]-0.5], marker='o', color = colors[i], markeredgewidth=0, markersize = markersize,**kwargs)
            
        ax = monitor.add_field_background(ax=ax, father_fields=father_fields, background_color=background_color, **background_kw)
        
        return ax
    
    def fit_linear_regression(self, x, y):
        self.slope, self.intercept, r_value, self.p_value, std_err = linregress(x, y)
        return self.slope, self.intercept, self.p_value
    
    def predict(self, x):
        return self.slope*x + self.intercept