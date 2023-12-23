import numpy as np
import os
import pickle
import copy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

from mylib.multiday.field_tracker import conditional_prob
from mylib.maze_utils3 import Sigmoid, SigmoidFit, Exponential, ExponentialFit

class FieldImputation(object):
    def __init__(
        self, 
        field_reg: np.ndarray,
        iter_num: int = 10,
        max_nod_duration: int = 3
    ) -> None:
        """__init__: Initialize FieldImputation

        Parameters
        ----------
        field_reg : np.ndarray
            The field register binarization matrix
        iter_num : int, optional
            The iteration number, by default 50
        max_nod_duration : int, optional
            The maximum not detected duration, by default 3
        """
        self._field_reg = field_reg
        self._iter_num = iter_num
        self._max_nod_duration = max_nod_duration
        
        self._retained_prob_rec = np.zeros((iter_num+1, field_reg.shape[0]-1), np.float64)
        self._recover_prob_rec = np.zeros((iter_num+1, field_reg.shape[0]-1), np.float64)
        
        self._retained_prob_rec[0, :], self._recover_prob_rec[0, :] = self.update_p(field_reg)
        self._retained_prob, self._recover_prob = self._retained_prob_rec[0, :], self._recover_prob_rec[0, :]
    
    @property
    def retained_prob_rec(self):
        return self._retained_prob_rec
    
    @property
    def recover_prob_rec(self):
        return self._recover_prob_rec
    
    @property
    def field_reg(self):
        return self._field_reg
    
    @property
    def iter_num(self):
        return self._iter_num
    
    @property
    def max_nod_duration(self):
        return self._max_nod_duration
    
    def update_p(self, field_reg: np.ndarray):
        _, retained_prob, _, recover_prob, _, _ = conditional_prob(field_reg=field_reg, thre = 5)
        
        idx = np.where(np.isnan(retained_prob))[0]
        if len(idx) > 1:
            retained_prob[idx] = retained_prob[idx[1]-1]
            
        idx = np.where(np.isnan(recover_prob))[0]
        if len(idx) > 1:
            recover_prob[idx] = recover_prob[idx[1]-1]
            
        return retained_prob[1:], recover_prob[1:]
    
    def guess(self):
        guess_reg = cp.deepcopy(self._field_reg)
    
        for j in tqdm(range(guess_reg.shape[1])):
            count_one, count_nil = 0, 0 # turn on and turn off
            is_exceeded = False
            gap_i = 0
            for i in range(guess_reg.shape[0]-1):
                if i < gap_i:
                    continue
                
                if np.isnan(guess_reg[i, j]):
                    
                    if np.isnan(np.nansum(guess_reg[i:, j])):
                        # If the rest values are all NANs.
                        break
                    
                    """
                    if i < guess_reg.shape[0] - self._max_nod_duration:
                        if np.isnan(np.nansum(guess_reg[i:i+self._max_nod_duration+1, j])):
                            # The continuous NAN values exceed the max_nod_duration
                            is_exceeded = True
                            # gap_i = i+self._max_nod_duration+1
                    """
                    if is_exceeded:
                        continue
                    
                    if count_one >= 1:
                        if np.isnan(self._retained_prob[count_one-1]):
                            continue
                        # On state -> not detected
                        guess_reg[i, j] = np.random.choice([0, 1], size = 1, p=[1-self._retained_prob[count_one-1], self._retained_prob[count_one-1]])
                        if guess_reg[i, j] == 1:
                            count_one += 1
                        else:
                            count_one = 0
                            count_nil = 1
                    elif count_nil >= 1:
                        if np.isnan(self._recover_prob[count_nil-1]):
                            continue
                        # Off state -> not detected
                        guess_reg[i, j] = np.random.choice([0, 1], size = 1, p=[1-self._recover_prob[count_nil-1], self._recover_prob[count_nil-1]])
                        if guess_reg[i, j] == 1:
                            count_nil = 0
                            count_one = 1
                        else:
                            count_nil += 1
                            
                elif guess_reg[i, j] == 1:
                    is_exceeded = False
                    if count_one >= 1:
                        count_one += 1
                    else:
                        count_one = 1
                        count_nil = 0
                        
                elif guess_reg[i, j] == 0:
                    is_exceeded = False
                    if count_nil >= 1:
                        count_nil += 1
                    else:
                        count_nil = 1
                        count_one = 0
                        
                else:
                    raise ValueError(f"field reg contains invalid value {guess_reg[i, j]} at row {i} column {j}. ")
                
        return guess_reg
    
    def iteration(self):
        for i in range(self._iter_num):
            print("iteration: ", i)
            guess_reg = self.guess()
            self._retained_prob_rec[i+1, :], self._recover_prob_rec[i+1, :] = self.update_p(guess_reg)
            self._retained_prob, self._recover_prob = self._retained_prob_rec[i+1, :], self._recover_prob_rec[i+1, :]
            
    @staticmethod
    def imputation(field_reg: np.ndarray, iter_num: int, max_nod_duration: int):
        imputer = FieldImputation(field_reg=field_reg, iter_num=iter_num, max_nod_duration=max_nod_duration)
        imputer.iteration()
        return imputer
    

if __name__ == '__main__':
    import pickle
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mylib.maze_utils3 import Clear_Axes
    
    with open(r"E:\Data\Cross_maze\10212\Maze1-2-footprint\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)  
        
    imputer = FieldImputation.imputation(field_reg=trace['field_reg'], iter_num=10, max_nod_duration=3)

    Data = {"Duration": np.array([], dtype=np.int64), "Conditional Prob.": np.array([], dtype=np.float64), "Iter.": np.array([], dtype=np.int64)}
    SubData = {"Duration": np.array([], dtype=np.int64), "Conditional Prob.": np.array([], dtype=np.float64), "Iter.": np.array([], dtype=np.int64)}
    for i in range(imputer.iter_num+1):
        print(np.arange(1, imputer.retained_prob_rec.shape[1]+1), imputer.retained_prob_rec[i, :])
        idx = np.where(np.isnan(imputer.retained_prob_rec[i, :]) == False)[0]
        print(imputer.retained_prob_rec[i, idx])
        a, b = SigmoidFit(np.arange(1, imputer.retained_prob_rec.shape[1]+1)[idx], imputer.retained_prob_rec[i, idx])
        x = np.linspace(1, idx.shape[0], idx.shape[0]*1000+1)
        y = Sigmoid(x, a, b)
        
        Data["Duration"] = np.concatenate([Data["Duration"], x - 1])
        Data["Conditional Prob."] = np.concatenate([Data["Conditional Prob."], y])
        Data['Iter.'] = np.concatenate([Data['Iter.'], np.repeat(i, x.shape[0])])
        
        SubData["Duration"] = np.concatenate([SubData["Duration"], np.arange(1, imputer.retained_prob_rec.shape[1]+1)[idx] - 1])
        SubData["Conditional Prob."] = np.concatenate([SubData["Conditional Prob."], imputer.retained_prob_rec[i, idx]])
        SubData['Iter.'] = np.concatenate([SubData['Iter.'], np.repeat(i, idx.shape[0])])
        
    fig = plt.figure(figsize=(4,2))
    ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    sns.lineplot(
        x = "Duration",
        y = "Conditional Prob.",
        hue = "Iter.",
        palette="rocket",
        data = Data,
        ax = ax,
        linewidth = 0.5
    )
    sns.stripplot(
        x = "Duration",
        y = "Conditional Prob.",
        hue = "Iter.",
        palette="rocket",
        data = SubData,
        ax = ax,
        edgecolor='black',
        size=3,
        linewidth=0.15,
        jitter=0.1,
        alpha = 0.8
    )
    plt.show()