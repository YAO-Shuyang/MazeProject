import numpy as np
import matplotlib
from mylib.maze_graph import maze1_graph as MG1
from mylib.maze_graph import maze2_graph as MG2
from mylib.maze_graph import CorrectPath_maze_1 as CP1
from mylib.maze_graph import CorrectPath_maze_2 as CP2
from mylib.maze_graph import IncorrectPath_maze_1 as IP1
from mylib.maze_graph import IncorrectPath_maze_2 as IP2
from mylib.maze_graph import xorder1, xorder2, S2F
from mylib.maze_graph import DecisionPoint1_Linear as DP1
from mylib.maze_graph import DecisionPoint2_Linear as DP2
from mylib.maze_graph import DecisionPoint2WrongGraph1 as WG1
from mylib.maze_graph import DecisionPoint2WrongGraph2 as WG2
from mylib.maze_utils3 import mkdir, DrawMazeTrack, occu_time_transform, Clear_Axes
from mylib.maze_utils3 import ColorBarsTicks, ax_shadow, FastDistance, spike_nodes_transform
from matplotlib.axes import Axes
import matplotlib
import os
import matplotlib.pyplot as plt

def field_arange(trace, is_pc: bool = True):
    num = np.zeros(144, dtype=np.int64)
    n = trace['n_neuron']
    place_field_all = trace['place_field_all']
    
    if np.nansum(trace['is_placecell']) <= 50:
        return num*np.nan
    
    for i in range(n):
        if is_pc == False or trace['is_placecell'][i] == 1:
            fields = np.array([])
            for k in place_field_all[i].keys():
                fields = np.concatenate([fields, place_field_all[i][k]])

            fields = fields.astype(np.int64)
            num[S2F[fields-1]-1] += 1
        
    return num

def add_twiny(data: np.ndarray, ax: Axes, length: int, path_type: str = 'cp', marker: str = 'x',
              ylabel: str = None, markersize: float = 1.5, markerfacecolor: str = 'brown',
              markeredgecolor: str = 'brown', label: str = None, color: str = 'purple', 
              linewidth: float = 0.7, **kwargs):
    if path_type == 'cp':
        ay=Clear_Axes(ax.twinx(), close_spines=['top', 'left'])
        a = ay.plot(np.linspace(1, length, length), data[0:length], marker = marker, 
                    markersize = markersize, color = color,
                    markerfacecolor = markerfacecolor, 
                    markeredgecolor = markeredgecolor, 
                    label = label, linewidth = linewidth, **kwargs)
        ay.set_ylabel(ylabel)
        ay.set_yticks(ColorBarsTicks(peak_rate=round(np.nanmax(data), 1), 
                                     is_auto=True, tick_number=4))
        ay.axis([0,length+1, 0, round(np.nanmax(data), 1)*1.1])
    elif path_type == 'ip':
        ay=Clear_Axes(ax.twinx(), close_spines=['top', 'left'])
        a = ay.plot(np.linspace(length+1, 144, 144 - length), data[length::], 
                    marker = marker, markersize = markersize, color = color,
                    markerfacecolor = markerfacecolor, 
                    markeredgecolor = markeredgecolor, 
                    label = label, linewidth = linewidth, **kwargs)
        ay.set_ylabel(ylabel)
        ay.set_yticks(ColorBarsTicks(peak_rate=round(np.nanmax(data), 1), 
                                     is_auto=True, tick_number=4))
        ay.axis([length+1, 145, 0, round(np.nanmax(data), 1)*1.1])     
    
    return a

def plot_field_arange(trace: dict, save_loc:str = None, file_name: str = 'place_field_arrangement',
                      is_showbehavior = False, is_showevents = True):
    assert is_showbehavior ^ is_showevents == True
    
    num_pc = field_arange(trace, is_pc=True)
    prop_pc = num_pc / np.sum(num_pc) * 100 # proportion
    
    MAX = np.nanmax(prop_pc)
    if np.isnan(MAX):
        return

    mkdir(save_loc)
    if trace['maze_type'] == 0:
        return
    
    length = len(CP1) if trace['maze_type'] == 1 else len(CP2)
    x_order = xorder1-1 if trace['maze_type'] == 1 else xorder2-1
    DP = DP1 if trace['maze_type'] == 1 else DP2 # Decision point
    WG = WG1 if trace['maze_type'] == 1 else WG2
    x_l, x_r, dp_ord = [], [], []
    for p in DP:
        idx = np.where(x_order == p-1)[0][0]
        x_l.append(idx+0.5)
        x_r.append(idx+1.5)
        dp_ord.append(idx+1)
               
    prop_ord_pc = prop_pc[x_order]
    
    # 1. occu rate
    occu_old = occu_time_transform(trace['occu_time'], nx = 12)
    occu_rate = occu_old/np.nansum(occu_old)*100
    occu_rate_ord = occu_rate[x_order]
    
    
    # 2. Calculate behavior events (statistically)
    suc_rate = BehaviorEvents.success_rate(trace['maze_type'], 
                                           spike_nodes_transform(trace['correct_nodes'], 
                                                                 nx = 12) , 
                                           trace['correct_time'])
    suc_rate_ord = suc_rate[x_order]
    
        
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (11,4), 
                             gridspec_kw={'width_ratios': [3, 8]})
    # Fig 1 plot maze profile ====================================================================
    axm = Clear_Axes(axes[0, 0]) 
    axm.set_aspect('equal')
    axm, seg_num = DrawMazeTrack(ax = axm, maze_type=trace['maze_type'], linewidth=1, 
                            text_args={'ha': 'center', 'va':'center', 'fontsize':4},
                            fill_args={'alpha':0.5, 'color': 'gray', 'ec': None},)
    axm.invert_yaxis()
    
    
    # ============================================================================================
    ax = Clear_Axes(axes[0, 1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    # All cells
    """
    ax.plot(np.linspace(1, length, length), prop_ord_ac[0:length], ls='-.', 
            marker = 's', markersize = 2, markerfacecolor = 'brown',markeredgecolor = 'brown', 
            label = 'All cells, c.p.', linewidth = 0.7)
    ax.bar(np.linspace(length+1, 144, 144 - length), prop_ord_ac[length:144], width = 0.8, 
           label = 'All cells, i.p.', alpha = 0.5)
    """
                
    # Place cells
    c = ax.plot(np.linspace(1, length, length), prop_ord_pc[0:length], 
                marker = '^', markersize = 1.5, 
                markerfacecolor = 'black',markeredgecolor = 'black', 
                label = 'Place cells, c.p.', linewidth = 0.7)

    # plot behavior or not
    if is_showbehavior:
        a = add_twiny(data=occu_rate_ord, ax=ax, length=length, 
                      ylabel = 'occu. time. (%)', label = 'occu. time.')
        
        legs = a + c
        labs = [l.get_label() for l in legs]
        ax.legend(legs, labs, facecolor = 'white', edgecolor = 'white', 
                  loc='upper left', bbox_to_anchor=(1.1, 1), 
                  fontsize = 8, title_fontsize = 8, ncol = 1, 
                  title = 'Cell Type')
    elif is_showevents:
        a = add_twiny(data=suc_rate_ord, ax=ax, length=length, 
                      ylabel = 'behav. events(%)', label = 'behav. events')
        legs = a + c
        labs = [l.get_label() for l in legs]
        ax.legend(legs, labs, facecolor = 'white', edgecolor = 'white', 
                  loc='upper left', bbox_to_anchor=(1.1, 1), 
                  fontsize = 8, title_fontsize = 8, ncol = 1, 
                  title = 'Cell Type')
    else:
        ax.legend(facecolor = 'white', edgecolor = 'white', loc='upper left', 
                  bbox_to_anchor=(1, 1), fontsize = 8, title_fontsize = 8, 
                  ncol = 1, title = 'Cell Type')
        
    # Decision area text label
    ax_shadow(ax=ax, x1_list=x_l, x2_list=x_r, y = np.linspace(0, MAX*1.1, 10000), 
              colors = np.repeat('gray', len(DP)), edgecolor = None)
            
    # plot inter decision points area
    k = 0
    cmap = matplotlib.colormaps['rainbow']
    colors = cmap(np.linspace(0, 1, seg_num))
    for i in range(len(x_l)-1):
        if x_l[i+1] == x_r[i]:
            continue
        ax.fill_between(x = np.linspace(x_r[i], x_l[i+1], 2), 
                        y1 = MAX*1.05, y2 = MAX*1.1, ec = None, 
                        color = colors[k])
        k += 1
        
    # Adjust ax object            
    # ax.axvline(length+0.5, ls='--', color = 'black')        
    # ax.set_xlabel(f"Maze ID (Linearized) [Maze {trace['maze_type']} - Mouse {trace['MiceID']}"
    #               +"- Date {trace['date']} - {np.nansum(trace['is_placecell'])} Cells]")
    ax.set_ylabel("Field Proportions / %")
    ax.set_xlabel('Maze ID on correct path')
    ax.set_xticks(dp_ord, labels = DP, fontsize = 6, rotation=90)
        
    ax.set_yticks(ColorBarsTicks(peak_rate=round(MAX, 1)))
    
    # ax.text(length+1, MAX, 'Incorrect\npath', ha = 'left', fontsize = 8, va = 'center')
    # ax.text(length, MAX, 'Correct\npath', ha = 'right', fontsize = 8, va  ='center')
    ax.axis([0,length+1,0, MAX*1.1])
    
    # Maze profile for incorrect path =========================================================
    axm = Clear_Axes(axes[1, 0]) 
    axm.set_aspect('equal')
    axm, seg_num = DrawMazeTrack(ax = axm, maze_type=trace['maze_type'], linewidth=1, 
                            text_args={'ha': 'center', 'va':'center', 'fontsize':4},
                            path_type='ip',
                            fill_args={'alpha':0.5, 'color': 'gray', 'ec': None},)
    axm.invert_yaxis()
    
    
    # Incorrect path figure ======================================================================
    ax = Clear_Axes(axes[1, 1], close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    # All cells
    """
    ax.plot(np.linspace(1, length, length), prop_ord_ac[0:length], ls='-.', 
            marker = 's', markersize = 2, markerfacecolor = 'brown',markeredgecolor = 'brown', 
            label = 'All cells, c.p.', linewidth = 0.7)
    ax.bar(np.linspace(length+1, 144, 144 - length), prop_ord_ac[length:144], width = 0.8, 
           label = 'All cells, i.p.', alpha = 0.5)
    """
                
    # Place cells
    if trace['maze_type'] == 1:
        x = [1, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 33, 34, 35, 39, 43, 47, 
             51, 55, 56, 60, 61, 62, 63, 67, 68, 69, 73, 77, 78, 79, 80, 81]
        xlab = ['2', '25', '7', '8', '16', '28', '42', '41', '36', '35', '96', 
                '106', '93', '43', '53', '52', '40', '98', '86', '121', '135',
                '128', '139', '143', '118', '120']
        xs = [1, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 33, 35, 39, 43, 47, 51, 
              55, 56, 60, 63, 67, 69, 73, 77, 81]
        x_l = [0.5, 4.5, 8.5, 13.5, 18.5, 23.5, 28.5, 32.5, 38.5, 42.5, 46.5, 50.5, 54.5, 59.5, 66.5, 72.5, 76.5]
        x_r = [1.5, 5.5, 10.5, 15.5, 20.5, 25.5, 29.5, 35.5, 39.5, 43.5, 47.5, 51.5, 56.5, 63.5, 69.5, 73.5, 81.5]
        
    else:
        x = [1, 2, 3, 4, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 27, 28, 32, 
             33, 34, 35, 36, 37, 38, 42, 43, 47, 51, 52, 53, 57, 61, 62, 63, 
             64, 65, 66, 67, 71, 75, 79, 83, 87, 91]
        xlab = ['3', '4', '87', '122', '136', '113', '41', '42', '62', '26', 
                '8', '20', '56', '36', '47', '90', '117', '139', '81', '58', 
                '48', '107', '118', '132']
        xs = [1, 4, 8, 12, 16, 23, 27, 28, 32, 38, 42, 43, 47, 51, 53, 57, 61, 
              67, 71, 75, 79, 83, 87, 91]
        x_l = [0.5, 7.5, 11.5, 15.5, 26.5, 31.5, 41.5, 46.5, 50.5, 56.5, 60.5, 70.5, 74.5, 78.5, 82.5, 86.5, 90.5]
        x_r = [4.5, 8.5, 12.5, 23.5, 28.5, 38.5, 43.5, 47.5, 53.5, 57.5, 67.5, 71.5, 75.5, 79.5, 83.5, 87.5, 91.5]
    
    for k in WG.keys():
        if len(WG[k]) == 1:
            m, n = (WG[k][0]-1) % 12, (WG[k][0]-1) // 12
            axm.text(m, n, str(WG[k][0]), ha='center', va = 'center', color = 'k', fontsize = 4)
        else:
            m, n = (WG[k][0]-1) % 12, (WG[k][0]-1) // 12
            axm.text(m, n, str(WG[k][0]), ha='center', va = 'center', color = 'k', fontsize = 4)
            
            m, n = (WG[k][-1]-1) % 12, (WG[k][-1]-1) // 12
            axm.text(m, n, str(WG[k][-1]), ha='center', va = 'center', color = 'k', fontsize = 4)            
    
    assert len(x_l) == len(x_r) and len(x_l) == len(DP)
    
    cmap = matplotlib.colormaps['rainbow']
    colors = cmap(np.linspace(0, 1, len(x_l)))
    
    n = 1
    ks = list(WG.keys())
    for i in range(len(x_l)):
        ls = len(WG[ks[i]])
        ax.bar(np.arange(n, n + ls), prop_pc[np.array(WG[ks[i]])-1], 
               width = 0.8, color = colors[i])
        n = n + ls + 3
        
        ax.text((x_l[i]+x_r[i])/2, MAX*1.06, str(DP[i]), ha = 'center', va = 'center', fontsize = 6)
        ax.fill_between(np.linspace(x_l[i], x_r[i], 2), y1 = MAX*1.02, 
                        y2 = MAX*1.1, ec = None, color = colors[i])

    # plot behavior or not
    if is_showbehavior:
        ay=Clear_Axes(ax.twinx(), close_spines=['top'])
        a = ay.plot(x, occu_rate_ord[length::], marker = 's', 
                    markersize = 1.5, color = 'purple',
                    markerfacecolor = 'brown',markeredgecolor = 'brown', 
                    label = 'occu. prop.', linewidth = 0.7)
        ay.set_ylabel("Occupation\nProportion / %")
        ay.set_yticks(ColorBarsTicks(peak_rate=round(np.nanmax(occu_rate_ord), 1), 
                                     is_auto=True, tick_number=4))
        ay.axis([0,x[-1]+1, 0, round(np.nanmax(occu_rate_ord), 1)*1.1])
    
    # Adjust ax object
    ax.set_xlabel(f"Maze ID on incorrect path")
    ax.set_ylabel("Field Proportions / %")
    ax.set_xticks(xs, labels = xlab, fontsize = 6, rotation=90)
    ax.set_yticks(ColorBarsTicks(peak_rate=round(MAX, 1)))
    ax.axis([0, x[-1]+1, 0, MAX*1.1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, file_name+'.png'), dpi = 1200)
    plt.savefig(os.path.join(save_loc, file_name+'.svg'), dpi = 1200)
    plt.close()


class MazeBase(object):
    def __init__(self, maze_type: int) -> None:
        self._maze = maze_type
        self._mg = MG1 if self._maze == 1 else MG2
        self._dp = DP1 if self._maze == 1 else DP2 # decision point
        self._cp = CP1 if self._maze == 1 else CP2 # correct path
        self._ip = IP1 if self._maze == 1 else IP2 # incorrect path
        self._wg = WG1 if self._maze == 1 else WG2 # wrong path graph
        self._xorder = xorder1-1 if self._maze == 1 else xorder2-1
        self._len = len(CP1) if maze_type == 1 else len(CP2)

class DecisionPointAnalyzer(MazeBase):
    def __init__(self, Data: np.ndarray, maze_type:int) -> None:
        """__init__

        Initialize the basic data structure for the analysis of decision points.

        Parameters
        ----------
        Data : np.ndarray
            shape (day, 144), save the data (spatial distribution of the fields) of all sessions.
        maze_type : int
            maze type
        """
        super().__init__(maze_type=maze_type)
        self._Data = Data
        self._field_prop = (self._Data.T / np.nansum(Data, axis = 1)).T * 100
        
    def _sort_counts(self, obj: np.ndarray):
        return obj[self._xorder]
    
    def _sort_dps(self, obj: np.ndarray):
        dps_ord = np.zeros_like(self._dp)
        
        for i in range(self._dp.shape[0]):
            dps_ord[i] = np.where(self._xorder == self._dp)[0][0]
            
        return dps_ord
    
    @property
    def get_cp_data(self):
        return self._field_prop[:, self._cp-1]
    
    @property
    def get_ip_data(self):
        return self._field_prop[:, self._ip-1]
    
    
class BehaviorEvents(MazeBase):
    def __init__(self, maze_type: int, behav_nodes: np.ndarray, behav_time: np.ndarray) -> None:
        super().__init__(maze_type)
        self.node = behav_nodes
        self.time = behav_time
        self.abbreviate()
        self.get_direc_vec()
        self.get_events()
        
    def abbreviate(self):
        idx = [0]
        
        i = 1
        curr = self.node[0]
        while i < self.node.shape[0]:
            if self.node[i] != curr:
                idx.append(i)
                curr = self.node[i]
                
            i += 1
        
        idx = np.array(idx, np.int64)
        self.abbr_idx = idx
        
        self.abbr_node = self.node[idx]
        self.abbr_time = self.time[idx]
        self.dt = np.append(np.ediff1d(self.abbr_time), 0)
        
    def get_direc_vec(self):
        direc_vec = np.zeros(self.abbr_node.shape[0], dtype=np.float64)*np.nan
        
        for i in range(direc_vec.shape[0]-1):
            a, b = self.abbr_node[i], self.abbr_node[i+1]
            if b not in self._mg[a]: # not next
                continue
            
            if a in self._cp and b in self._cp:
                direc_vec[i] = self.get_status_both_cp(a, b)
            elif a in self._cp and b in self._ip:
                direc_vec[i] = self.get_status_ipcp(a, b)
            elif a in self._ip and b in self._cp:
                direc_vec[i] = self.get_status_ipcp(a, b)
            elif a in self._ip and b in self._ip:
                direc_vec[i] = self.get_status_both_ip(a, b)
            else:
                raise ValueError(f"{a} and {b} are not a valid value pair.")
        
        self.direc_vec = direc_vec
        return direc_vec
        
    def get_status_both_cp(self, p1: int, p2: int):
        assert p1 != p2
        id1 = np.where(self._cp == p1)[0]
        id2 = np.where(self._cp == p2)[0]
        if id1 - id2 < 0:
            return 1
        elif id1 - id2 > 0:
            return -1
        else:
            raise ValueError(f"Input ID {p1} and {p2} are the same.")
    
    def get_status_both_ip(self, p1: int, p2: int):
        assert p1 != p2
        id1 = FastDistance(p1, 144, maze_type = self._maze, nx = 12)
        id2 = FastDistance(p2, 144, maze_type = self._maze, nx = 12)
        
        if id1 - id2 < 0:
            return -2
        elif id1 - id2 > 0:
            return -2
        else:
            raise ValueError(f"Input ID {p1} and {p2} are the same.")
        
    def get_status_ipcp(self, p1: int, p2: int):
        if p1 in self._cp:
            return -2
        else:
            return 2
        
    def get_events(self):
        events_stats = {'turn forward': np.zeros(144, dtype=np.int64),
                        'turn back':    np.zeros(144, dtype=np.int64),
                        'forward stray':np.zeros(144, dtype=np.int64),
                        'back stray':   np.zeros(144, dtype=np.int64),
                        'correct forward': np.zeros(144, dtype=np.int64),
                        'correct back':np.zeros(144, dtype=np.int64),
                        'pass forward': np.zeros(144, dtype=np.int64),
                        'pass back':    np.zeros(144, dtype=np.int64)}
        
        for i in range(self.direc_vec.shape[0]-2):
            a, b = self.direc_vec[i], self.direc_vec[i+1]
            
            if np.isnan(a) or np.isnan(b):
                continue
            
            x, y, z = self.abbr_node[i], self.abbr_node[i+1], self.abbr_node[i+2]
            if a == 1 and b == 1:
                events_stats['pass forward'][y-1] += 1
            elif a == -1 and b == -1:
                events_stats['pass back'][y-1] += 1
            elif a == -1 and b == 1:
                events_stats['turn forward'][y-1] += 1
            elif a == 1 and b == -1:
                events_stats['turn back'][y-1] += 1
            elif a == 1 and b == -2:
                events_stats['forward stray'][y-1] += 1
            elif a == -1 and b == -2:
                events_stats['back stray'][y-1] += 1
            elif a == 2 and b == -1:
                events_stats['correct back'][y-1] += 1
            elif a == 2 and b == 1:
                events_stats['correct forward'][y-1] += 1
                
        self.events_stats = events_stats
        return events_stats
    
    @staticmethod
    def success_rate(maze_type: int, behav_nodes: np.ndarray, 
                     behav_time: np.ndarray):
        BehavObj = BehaviorEvents(maze_type=maze_type, 
                                  behav_nodes=behav_nodes, 
                                  behav_time=behav_time)
        
        total_events = (BehavObj.events_stats['pass forward'] + 
                        BehavObj.events_stats['pass back'] + 
                       BehavObj.events_stats['forward stray'] + 
                       BehavObj.events_stats['back stray'] +
                       BehavObj.events_stats['turn forward'] + 
                       BehavObj.events_stats['turn back'] +
                       BehavObj.events_stats['correct forward'] +
                       BehavObj.events_stats['correct back'])
        success_events = (BehavObj.events_stats['pass forward'] +
                          BehavObj.events_stats['turn forward'] +
                          BehavObj.events_stats['correct forward'])
        return success_events / total_events * 100
        
            
        
if __name__ == '__main__':
    import pickle
    with open(r'E:\Data\Cross_maze\11095\20220828\session 2\trace.pkl', 'rb') as handle:
        trace = pickle.load(handle)
        
    trace['p'] = r"E:\Data\Cross_maze\11095\20220828\session 2"
    
    plot_field_arange(trace, save_loc=os.path.join(trace['p'], 'PeakCurve'), is_showevents=True)
