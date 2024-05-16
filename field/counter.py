import numpy as np
from tqdm import tqdm

class Counter:
    def __init__(self, stat: int) -> None:
        if np.isnan(stat):
            stat = 0
        else:
            stat = int(stat)
            
        self._stat = stat
        self._hist = 1
        
    @property
    def stat(self):
        return self._stat
    
    @property
    def hist(self):
        return self._hist
    
    def update(self, stat: int):
        if np.isnan(stat):
            stat = 0
        else:
            stat = int(stat)
            
        if self._stat == 1:
            if stat == 1:
                self._hist += 1
            else:
                self._hist = 1
                
        elif self._stat == 0:
            if stat == 1:
                self._hist = 1
            else:
                self._hist += 1
        else:
            raise ValueError(f"The stat should be 0 or 1! instead of {stat}")
        
        self._stat = stat

        
def count_superstable_fields(fields: list[Counter], thre = 5):
    n_superstable = 0
    for i, field in enumerate(fields):
        if field.stat == 1 and field.hist >= thre:
            n_superstable += 1
    return n_superstable   
 
def update_counters(fields: list[Counter], index_line):
    for i, field in enumerate(fields):
        field.update(index_line[i])
    return fields

def calculate_superstable_fraction(field_reg: np.ndarray, thres: np.ndarray = np.arange(2, 26, 2)) -> np.ndarray:
    
    superstable_frac = np.zeros((thres.shape[0], field_reg.shape[0]))
    fields = [Counter(i) for i in field_reg[0, :]]
    #n_fields = np.nansum(field_reg, axis=1)
    
    for i in tqdm(range(1, field_reg.shape[0])):
        fields = update_counters(fields, field_reg[i, :])
        for j in range(thres.shape[0]):
            n_fields = np.where((np.isnan(np.sum(field_reg[max(0, i-thres[j]+1):i+1, :], axis=0)) == False)&(field_reg[i, :] == 1))[0].shape[0]
            if n_fields == 0:
                superstable_frac[j, i] = np.nan
            else:
                superstable_frac[j, i] = count_superstable_fields(fields, thres[j]) / n_fields
    return superstable_frac

def calculate_survival_fraction(field_reg: np.ndarray):
    survival_frac = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    start_sessions = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    training_day = np.full((field_reg.shape[0], field_reg.shape[0]), np.nan)
    for i in tqdm(range(field_reg.shape[0])):

        for j in range(i, field_reg.shape[0]):
            # compare session
            # start session
            n_ori = np.where((np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False)&(field_reg[i, :] == 1))[0].shape[0]
            n_retain = np.where(np.nansum(field_reg[i:j+1, :], axis=0) == j-i+1)[0].shape[0]
            survival_frac[i, j] = n_retain / n_ori
            start_sessions[i, j] = i+1
            training_day[i, j] = j-i+1
    return survival_frac, start_sessions, training_day


if __name__ == '__main__':
    import pickle
    import os
    import matplotlib.pyplot as plt
    from mylib.maze_utils3 import Clear_Axes
    #with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\trace_mdays.pkl", 'rb') as handle:
    #with open(r"E:\Data\Cross_maze\10224\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
    #with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_91_equal_rate_0.9_10000fields_50days.pkl", 'rb') as handle:
    #with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_1_converged_10000fields_50days.pkl", 'rb') as handle:
    #with open(r"E:\Data\FigData\PermenantFieldAnalysis\mouse_101_converged_10000fields_50days_poly.pkl", 'rb') as handle:
    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)
        #trace['field_reg'] = trace['field_reg'].T[:, :]
        trace['field_reg'] = trace['field_reg'][:, :]
        
    field_num_mat = np.where(np.isnan(trace['field_reg']), 0, 1)[:, :]
    num = np.count_nonzero(field_num_mat, axis=0)
    #idx = np.where(num >= 26)[0]
    #trace['field_reg'] = trace['field_reg'][:, idx] # [equal rate]
    """    
    with open(r"E:\Data\FigData\PermenantFieldAnalysis\Mouse 1 [equal rate].pkl", 'rb') as handle: # [equal rate]
        trace = pickle.load(handle)
    
    """
    print(trace['field_reg'].shape)
    print(np.where(np.isnan(trace['field_reg']))[0].shape)

    supstable_frac = calculate_superstable_fraction(trace['field_reg'], thres=np.arange(3, 51, 2))
    training_day = np.arange(supstable_frac.shape[1])
    fig = plt.figure(figsize=(4,2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.plot(training_day, supstable_frac.T, linewidth=0.8)
    ax.plot(np.arange(3, 51, 2)-1, np.power(0.9, np.arange(3, 51, 2)-1), color = 'black')
    ax.set_xlabel("Training day")
    ax.set_ylabel("Superstable Fields %")
    # ax.set_ylim(0, 0.15)
    plt.show()