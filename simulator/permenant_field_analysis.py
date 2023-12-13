import numpy as np
from tqdm import tqdm
from mylib.maze_utils3 import Sigmoid

"""
def P1(i: int):
    if i < 0:
        raise ValueError("The index should be larger than 0!")
    elif i >= 1 and i <= 17:
        prob = np.array([60.92719857, 76.71708913, 83.84018737, 87.17430757,
                        88.71243408, 91.61007075, 93.32393228, 94.75129995,
                        95.91532307, 95.3408827, 95.03293152, 97.4463869,
                        99.46091251, 98.60132972, 96.29500129, 97.33285268,
                        100, 100, 100, 100, 100, 100, 100], dtype=np.float64)/100
        return prob[i-1]
    else:
        return 1
"""
def P1(i: int):
    if i < 0:
        raise ValueError("The index should be larger than 0!")
    elif i >= 1 and i <= 18:
        return 1-Sigmoid(i, a=-0.3070071833384061, b=-2.2827038320232362)
    else:
        return 1-Sigmoid(18, a=-0.3070071833384061, b=-2.2827038320232362)

def P2(i: int):
    if i < 0:
        raise ValueError("The index should be larger than 0!")
    elif i >= 1 and i <= 22:
        prob = np.array([0.199851374, 0.06326125, 0.028289579, 0.011983633,
                        0.006273658, 0.004004578, 0.00218729, 0.001579632,
                        0.001209344, 0.000780585, 0.00046724, 0.00026257,
                        0.000227393, 0.000321582, 0.000307891, 0.000361952,
                        0.000144781, 0.000217171, 0.000144781, 0.000217171,
                        0.0000723903, 0.0000723903], dtype=np.float64)
        return prob[i-1]
    else:
        return 0

class Field(object):
    def __init__(self, start_session: int) -> None:
        self._stat = 1
        self._hist = 1
        self._is_permenant = False
        self._active_sessions = 1
        self._start_session = start_session
        
    @property
    def stat(self):
        return self._stat
    
    @property
    def hist(self):
        return self._hist
    
    @property
    def active_sessions(self):
        return self._active_sessions
    
    @property
    def is_permenant(self):
        return self._is_permenant
    
    @property
    def start_session(self):
        return self._start_session
    
    def updata(self):
        
        if self._stat == 1:
            self._stat = np.random.choice([1, 0], p = [P1(self._hist), 1-P1(self._hist)])
            
            if self.stat == 1:
                self._hist += 1
                self._active_sessions += 1
            else:
                self._hist == 1
                self._is_permenant = False
                
        elif self._stat == 0:
            self._stat = np.random.choice([1, 0], p = [P2(self._hist), 1-P2(self._hist)])
            
            if self.stat == 1:
                self._hist = 1
                self._active_sessions += 1
            else:
                self._hist += 1
            
        if self._hist > 17 and self._stat == 1:
            self._is_permenant = True

def update_fields(fields: list[Field]):
    for field in fields:
        field.updata()
        
    return fields
        
# Update -> Count Permanent Fields -> Add New Fields 
def count_permanent_fields(fields: list[Field]):
    count_perm, count_act = 0, 0
    for field in fields:
        if field.is_permenant:
            count_perm += 1
        if field.stat == 1:
            count_act += 1

    return count_perm, count_act

def add_fields(fields: list[Field], session: int, field_num: int):
    count_perm, count_act = count_permanent_fields(fields)
    if count_act < field_num:
        for i in range(field_num-count_act):
            fields.append(Field(session))
            
    return fields

def init_fields(field_num: int):
    fields = []
    for i in range(field_num):
        fields.append(Field(1))
        
    return fields


def count_overlapping_fields(fields: list[Field], simu_fields: int):
    count_overlap = 0
    for i, field in enumerate(fields):
        if i >= simu_fields:
            break
        if field.stat == 1:
            count_overlap += 1
    return count_overlap

def count_active_history(fields: list[Field], days: int):
    active_history = np.zeros(days, np.int64)
    for field in fields:
        active_history[field.active_sessions-1] += 1
    return active_history

def update_field_reg(fields: list[Field], field_reg: np.ndarray):
    add_mat = np.zeros((len(fields) - field_reg.shape[0], field_reg.shape[1]), np.int64)
    field_reg = np.vstack((field_reg, add_mat))
    
    add_col = np.zeros((field_reg.shape[0], 1))
    for i, field in enumerate(fields):
        if field.stat == 1:
            add_col[i, 0] = 1
            
    field_reg = np.concatenate([field_reg, add_col], axis=1)
    return field_reg

def count_overlapping_fields(field_reg: np.ndarray):
    starts = np.arange(int((field_reg.shape[1]-1)/10)+1)*10
    
    overlaps = np.zeros((starts.shape[0], field_reg.shape[1]))*np.nan
    for i, start in enumerate(starts):
        for j in range(start, field_reg.shape[1]):
            overlaps[i, j] = len(np.where((field_reg[:, j] == 1)&(field_reg[:, start] == 1))[0])
        
    return overlaps

def main(MiceID: int, days: int = 100, simu_fields: int = 10000):
    trace = {"Mouse": MiceID, "days": np.arange(1, days+1), "simu_fields": simu_fields}
    
    permanent_field_num = np.zeros(days, np.int64)
    active_field_num = np.zeros(days, np.int64)
    field_num = np.zeros(days, np.int64)
    active_history = np.zeros(days, np.int64)
        
    fields = init_fields(simu_fields)
    active_field_num[0] = simu_fields
    field_num[0] = simu_fields
    field_reg = np.ones((simu_fields, 2), np.int64)
        
    for j in tqdm(range(2, days+1)):
        fields = update_fields(fields)
        fields = add_fields(fields, j, field_num=simu_fields)
        permanent_field_num[j-1], active_field_num[j-1] = count_permanent_fields(fields)
        field_num[j-1] = len(fields)
        field_reg = update_field_reg(fields, field_reg)
    
    field_reg = field_reg[:, 1:]
    active_history[:] = count_active_history(fields, days)
    overlaps = count_overlapping_fields(field_reg)
    
    trace['Permanent Field Num'] = permanent_field_num
    trace['Active Field Num'] = active_field_num
    trace['Cumulative Field Num'] = field_num
    trace['field_reg'] = field_reg
    trace['Active History'] = active_history
    trace['Overlap Field'] = overlaps
            
    return trace

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import os
    from mylib.local_path import figdata
    from mylib.statistic_test import *
    
    loc = os.path.join(figdata, "PermenantFieldAnalysis")
    if os.path.exists(loc)==False:
        os.mkdir(loc)
    

    for i in range(1, 11):
        trace = main(MiceID=i, days=100)
        with open(os.path.join(loc, "Mouse "+str(i)+'.pkl'), 'wb') as handle:
            pickle.dump(trace, handle)
            
        plt.figure(figsize=(4,2))
        ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
        ax.set_aspect("auto")
        ax.plot(trace['days'], trace['Permanent Field Num'], 'k', label='Permanent')
        ax.plot(trace['days'], trace['Active Field Num'], 'r', label='Active')
        ax.plot(trace['days'], trace['Cumulative Field Num'], 'b', label='Cumulative')
        ax.legend()
        plt.show()

    print()