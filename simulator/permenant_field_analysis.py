import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Exponential parameters estimated by real data.
MEAN_K = 0.3079
STD_K = 0.1299
K_MIN, K_MAX = 0.1597, 0.5414

MEAN_X0 = -4.886
STD_X0 = 2.699
X0_MIN, X0_MAX = -8.608, -1.460

# KWW decay parameters estimated by real data.
MEAN_A = 14.354
STD_A = 12.024
A_MIN, A_MAX = 2.400, 43.665

MEAN_B = 0.07224
STD_B = 0.08641
B_MIN, B_MAX = 0.00874, 0.29436

MEAN_C = 0.4121
STD_C = 0.0886
C_MIN, C_MAX = 0.3174, 0.5893

def kww_decay(x, a, b, c):
    return a*np.exp(-np.power(x/b, c))

def exp_func(x, k, b):
    return 1 - np.exp(-k * (x-b))

def P1(i: int, k: float, b: float):
    if i <= 0:
        raise ValueError("The index should be larger than 0!")
    else:
        return exp_func(i, k, b)-np.random.rand()*0.02 # bottom, level

def P2(i: int, a, b, c):
    if i <= 0:
        raise ValueError("The index should be larger than 0!")
    else:
        return kww_decay(i, a, b, c)

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
    
    def update(self, exp_params: list | float, kww_params: list, drift_rate: str = "converged"):
        assert drift_rate in ["converged", "equal-rate"]
        
        assert drift_rate != "equal-rate" or (exp_params <= 1 and exp_params >= 0)
        
        if self._stat == 1 and drift_rate == "converged":
            P = P1(self._hist, exp_params[0], exp_params[1])
            _stat = np.random.choice([1, 0], p = [P, 1-P])
            if _stat == 1:
                self._hist += 1
            else:
                self._hist = 1
        elif self._stat == 1 and drift_rate:
            _stat = np.random.choice([1, 0], p = [exp_params, 1-exp_params])
            
            if _stat == 1:
                self._hist += 1
            else:
                self._hist = 1
                
        elif self._stat == 0:
            P = P2(self._hist, kww_params[0], kww_params[1], kww_params[2])
            try:
                _stat = np.random.choice([1, 0], p = [P, 1-P])
            except:
                print(P, 1-P, kww_params)
                assert False
                
            if _stat == 1:
                self._hist = 1
            else:
                self._hist += 1
        else:
            raise ValueError("The stat should be 0 or 1!")
        
        self._stat = _stat

def update_fields(fields: list[Field], exp_params: list | float, kww_params: list, drift_rate: str = "converged"):
    new_fields = []
    for field in fields:
        field.update(exp_params, kww_params, drift_rate)
        new_fields.append(field)
        
    return new_fields
    
def count_active_fields(fields: list[Field]):
    count_act = 0
    for field in fields:
        if field.stat == 1:
            count_act += 1
    return count_act

def count_superstable(fields: list[Field], thre: int):
    count_stable = 0
    for field in fields:
        if field.stat == 1 and field.hist >= thre:
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

def main(MiceID: int, days: int = 100, simu_fields: int = 10000, drift_model: str = "converged", drift_rate: float = None):
    if drift_model == 'converged':
        exp_params = [norm.rvs(loc=MEAN_K, scale=STD_K), norm.rvs(loc=MEAN_X0, scale=STD_X0)]
    
        while exp_params[0] < K_MIN or exp_params[0] > K_MAX:
            exp_params[0] = norm.rvs(loc=MEAN_K, scale=STD_K)
        
        while exp_params[1] < X0_MIN or exp_params[1] > X0_MAX:
            exp_params[1] = norm.rvs(loc=MEAN_X0, scale=STD_X0)
            
        x = np.linspace(1, 26, 1000)
        y = exp_func(x, exp_params[0], exp_params[1])
        plt.plot(x, y)
        plt.ylim(0, 1)
        plt.show()
        
    elif drift_model == 'equal-rate':
        exp_params = drift_rate
    else:
        raise ValueError("The drift model should be 'converged' or 'equal-rate'!")
        
    kww_params = [norm.rvs(loc=MEAN_A, scale=STD_A), norm.rvs(loc=MEAN_B, scale=STD_B), norm.rvs(loc=MEAN_C, scale=STD_C)]
        
    while kww_params[1] < B_MIN or kww_params[1] > B_MAX:
        kww_params[1] = norm.rvs(loc=MEAN_B, scale=STD_B)
        
    while kww_params[2] < C_MIN or kww_params[2] > C_MAX:
        kww_params[2] = norm.rvs(loc=MEAN_C, scale=STD_C)
        
    while kww_params[0] < A_MIN or kww_params[0] > A_MAX or kww_params[0] >= np.exp(1/np.power(kww_params[1], kww_params[2])):
        kww_params[0] = norm.rvs(loc=MEAN_A, scale=STD_A)   
    
    y2 = kww_decay(x, kww_params[0], kww_params[1], kww_params[2])
    plt.plot(x, y2)
    plt.ylim(0, 1)
    plt.show()
    trace = {"Mouse": MiceID, "days": np.arange(1, days+1), "simu_fields": simu_fields,
             "exp_params": exp_params, "kww_params": kww_params, "drift_model": drift_model}
    
    print(exp_params, kww_params)
    
    # permanent_field_num = np.zeros(days, np.int64)
    active_field_num = np.zeros(days, np.int64)
    field_num = np.zeros(days, np.int64)
    #active_history = np.zeros(days, np.int64)
    superstable_thre = np.arange(3, 26, 2)
    superstable_num = np.zeros((superstable_thre.shape[0], days), np.float64)
        
    fields = init_fields(simu_fields)
    active_field_num[0] = simu_fields
    field_num[0] = simu_fields
    field_reg = np.ones((simu_fields, 2), np.int64)
        
    for j in tqdm(range(2, days+1)):
        fields = update_fields(fields, exp_params, kww_params, drift_rate=drift_model)
        fields = add_fields(fields, j, field_num=simu_fields)
        active_field_num[j-1] = count_active_fields(fields)
        #permanent_field_num[j-1], active_field_num[j-1] = count_permanent_fields(fields)
        field_num[j-1] = len(fields)
        field_reg = update_field_reg(fields, field_reg)
        for k, thre in enumerate(superstable_thre):
            superstable_num[k, j-1] = count_superstable(fields, thre=thre) / active_field_num[0]
    
    field_reg = field_reg[:, 1:]
    #active_history[:] = count_active_history(fields, days)
    # overlaps = count_overlapping_fields(field_reg)
    
    #trace['Permanent Field Num'] = permanent_field_num
    trace['Superstable Thre'] = superstable_thre
    trace['Active Field Num'] = active_field_num
    trace['Cumulative Field Num'] = field_num
    trace['field_reg'] = field_reg
    trace['Superstable Num'] = superstable_num
    #trace['Active History'] = active_history
    # trace['Overlap Field'] = overlaps
            
    return trace

    
if __name__ == '__main__':
    import pickle
    import os
    
    dir_name = r"E:\Data\FigData\PermenantFieldAnalysis"
    
    
    # Convergent Drift Model: 50 simulated Mice
    for mouse in range(1, 51):
        print(mouse, " Convergent Drift Model ---------------------------------------------------------")
        trace = main(mouse, 50, 10000, drift_model='converged')
        with open(os.path.join(dir_name, f'mouse_{mouse}_converged_10000fields_50days.pkl'), 'wb') as handle:
            pickle.dump(trace, handle)
        print(end='\n\n\n')
    """        
    # Equal Rate Drift Model: 50 simulated Mice
    FIXED_RATE = [0.5, 0.6, 0.7, 0.8, 0.9]
    for mouse in range(51, 101):
        print(mouse, " Equal Rate Drift Model ---------------------------------------------------------")
        trace = main(mouse, 50, 10000, drift_model='equal-rate', drift_rate=FIXED_RATE[int((mouse-51)//10)])
        with open(os.path.join(dir_name, f'mouse_{mouse}_equal_rate_{FIXED_RATE[int((mouse-51)//10)]}_10000fields_50days.pkl'), 'wb') as handle:
            pickle.dump(trace, handle)
        print(end='\n\n\n')
    """        
    
    """
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
    """