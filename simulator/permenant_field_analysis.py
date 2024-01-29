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
    if i <= 0:
        raise ValueError("The index should be larger than 0!")
    elif i >= 1 and i <= 18:
        return 1-Sigmoid(i, a=-0.307, b=1.5)#-0.3070071833384061, -1.284
    else:
        return P1(18)


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
        self._start_session = start_session
        
    @property
    def stat(self):
        return self._stat
    
    @property
    def hist(self):
        return self._hist
    
    
    @property
    def start_session(self):
        return self._start_session
    
    def update(self, is_equal_rate_drift: bool = False):
            
        if self._stat == 1 and is_equal_rate_drift == False:
            _stat = np.random.choice([1, 0], p = [P1(self._hist), 1-P1(self._hist)])
            
            if _stat == 1:
                self._hist += 1
            else:
                self._hist = 1
        elif self._stat == 1 and is_equal_rate_drift:
            _stat = np.random.choice([1, 0], p = [P1(1), 1-P1(1)])
            
            if _stat == 1:
                self._hist = 1
            else:
                self._hist += 1
                
        elif self._stat == 0:
            _stat = np.random.choice([1, 0], p = [P2(self._hist), 1-P2(self._hist)])
            
            if _stat == 1:
                self._hist = 1
            else:
                self._hist += 1
        else:
            raise ValueError("The stat should be 0 or 1!")
        
        self._stat = _stat

def update_fields(fields: list[Field], is_equal_rate_drift: bool = False):
    new_fields = []
    for field in fields:
        field.update(is_equal_rate_drift=is_equal_rate_drift)
        new_fields.append(field)
        
    return new_fields
    
"""  
# Update -> Count Permanent Fields -> Add New Fields 
def count_permanent_fields(fields: list[Field]):
    count_perm, count_act = 0, 0
    for field in fields:
        if field.is_permenant:
            count_perm += 1
        if field.stat == 1:
            count_act += 1

    return count_perm, count_act
""" 
def count_active_fields(fields: list[Field]):
    count_act = 0
    for field in fields:
        if field.stat == 1:
            count_act += 1
    return count_act

def count_superstable(fields: list[Field]):
    count_stable = 0
    for field in fields:
        if field.stat == 1 and field.hist >= 13:
            count_stable += 1
    return count_stable

def add_fields(fields: list[Field], session: int, field_num: int):
    count_act = count_active_fields(fields)
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

"""
def count_active_history(fields: list[Field], days: int):
    active_history = np.zeros(days, np.int64)
    for field in fields:
        active_history[field.active_sessions-1] += 1
    return active_history
"""
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

def main(MiceID: int, days: int = 100, simu_fields: int = 10000, is_equal_rate_drift: bool = False):
    trace = {"Mouse": MiceID, "days": np.arange(1, days+1), "simu_fields": simu_fields}
    
    # permanent_field_num = np.zeros(days, np.int64)
    active_field_num = np.zeros(days, np.int64)
    field_num = np.zeros(days, np.int64)
    #active_history = np.zeros(days, np.int64)
    superstable_num = np.zeros(days, np.float64)
        
    fields = init_fields(simu_fields)
    active_field_num[0] = simu_fields
    field_num[0] = simu_fields
    field_reg = np.ones((simu_fields, 2), np.int64)
        
    for j in tqdm(range(2, days+1)):
        fields = update_fields(fields, is_equal_rate_drift=is_equal_rate_drift)
        fields = add_fields(fields, j, field_num=simu_fields)
        active_field_num[j-1] = count_active_fields(fields)
        #permanent_field_num[j-1], active_field_num[j-1] = count_permanent_fields(fields)
        field_num[j-1] = len(fields)
        field_reg = update_field_reg(fields, field_reg)
        superstable_num[j-1] = count_superstable(fields) / active_field_num[0]
    
    field_reg = field_reg[:, 1:]
    #active_history[:] = count_active_history(fields, days)
    # overlaps = count_overlapping_fields(field_reg)
    
    #trace['Permanent Field Num'] = permanent_field_num
    trace['Active Field Num'] = active_field_num
    trace['Cumulative Field Num'] = field_num
    trace['field_reg'] = field_reg
    trace['Superstable Num'] = superstable_num
    #trace['Active History'] = active_history
    # trace['Overlap Field'] = overlaps
            
    return trace

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import os
    from mylib.local_path import figdata
    from mylib.statistic_test import *
    from field.field_tracker import conditional_prob
    
    loc = os.path.join(figdata, "PermenantFieldAnalysis")
    if os.path.exists(loc)==False:
        os.mkdir(loc)

    print(P1(1))
    x = np.arange(1, 26)
    y = np.array([P1(i) for i in x])
    plt.plot(x, y)

    Data = {"TestID": np.array([], np.int64), "Maze Type": np.array([]), "Data Type": np.array([]),
            "Duration": np.array([], np.int64), "Conditional Prob.": np.array([], np.float64), "No Detect Prob.": np.array([], np.float64),
            "Recovered Prob.": np.array([], np.float64), "Cumulative Prob.": np.array([], np.float64),
            "Re-detect Active Prob.": np.array([], np.float64), "Re-detect Prob.": np.array([], np.float64)}   
    
    for i in range(1):
        trace = main(10227, 26, 10000, is_equal_rate_drift=False)

        with open(os.path.join(loc, "Mouse "+str(i+1)+'.pkl'), 'wb') as handle:
            pickle.dump(trace, handle)
 
        trace['field_reg'] = trace['field_reg'].T
    
        retained_dur, prob, nodetect_prob, recover_prob, redetect_prob, redetect_frac, on_next_prob1 = conditional_prob(trace)
        Data['TestID'] = np.concatenate([Data['TestID'], np.repeat(int(i+1), prob.shape[0])])
        Data['Data Type'] = np.concatenate([Data['Data Type'], np.repeat("Simu", prob.shape[0])])
        Data['Maze Type'] = np.concatenate([Data['Maze Type'], np.repeat("Maze 1", prob.shape[0])])
        Data['Duration'] = np.concatenate([Data['Duration'], retained_dur])
        Data['Conditional Prob.'] = np.concatenate([Data['Conditional Prob.'], prob*100])
        Data['No Detect Prob.'] = np.concatenate([Data['No Detect Prob.'], nodetect_prob*100])
        Data['Recovered Prob.'] = np.concatenate([Data['Recovered Prob.'], recover_prob*100])
        Data['Re-detect Active Prob.'] = np.concatenate([Data['Re-detect Active Prob.'], redetect_prob*100])
        Data['Re-detect Prob.'] = np.concatenate([Data['Re-detect Prob.'], redetect_frac*100])
        res = np.nancumprod(prob)*100
        res[np.isnan(prob)] = np.nan
        res[0] = 100
        Data['Cumulative Prob.'] = np.concatenate([Data['Cumulative Prob.'], res])
        
    fig = plt.figure(figsize=(4,2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    idx = np.where((Data['Maze Type'] == "Maze 1")&(np.isnan(Data['Conditional Prob.']) == False))[0]
    SubData = SubDict(Data, Data.keys(), idx=idx)
    a1, b1 = SigmoidFit(SubData['Duration'], SubData['Conditional Prob.']/100)
    x1 = np.linspace(1, 23, 26001)
    y1 = Sigmoid(x1, a1, b1)
    print(f"Maze 1: {a1:.3f}, {b1:.3f}")
    ax.plot(x1, y1*100, color=sns.color_palette("rocket", 3)[1], linewidth=0.5)
    sns.stripplot(
        x = 'Duration',
        y = 'Conditional Prob.',
        data=Data,
        hue = "Maze Type",
        palette = [sns.color_palette("Blues", 9)[3], sns.color_palette("flare", 9)[3]],
        edgecolor='black',
        size=3,
        linewidth=0.15,
        ax = ax,
        dodge=True,
        jitter=0.1
    )
    ax.set_ylim(0, 103)
    ax.set_yticks(np.linspace(0, 100, 6))
    plt.show()
    """    """