# Continuous hidden state model

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

def reci_func(x, r, L0, b0, L1, b1):
    return np.clip(
        x * (L0 * r + b0) + (1-x) * (L1 * (1-r) + b1), 
        0, 1
    )

def logistic_func(x, r, k0, r01, k1, r02):
    return np.clip(
        x * (1 / (1 + np.exp(-k0 * (r - r01)))) + (1-x) * (1 - 1 / (1 + np.exp(-k1 * (1-r - r02)))),
        0, 1
    )

def poly2_func(x, r, a1, b1, c1, a2, b2, c2):
    return np.clip(
        x * (a1 * r**2 + b1 * r + c1) + (1-x) * (a2 * (1-r)**2 + b2 * (1-r) + c2),
        0, 1
    )
    
def poly3_func(x, r, a1, b1, c1, d1, a2, b2, c2, d2):
    return np.clip(
        x * (a1 * r**3 + b1 * r**2 + c1 * r + d1) + (1-x) * (a2 * (1-r)**3 + b2 * (1-r)**2 + c2 * (1-r) + d2),
        0, 1
    )

class ContinuousHiddenStateModel:
    def __init__(self, func: str, p0: str = 0.6) -> None:
        """
        func: 'reci', 'logistic', 'poly2', 'poly3'
        """
        if func not in ['reci', 'logistic', 'poly2', 'poly3']:
            raise ValueError(
                f"func must be one of ['reci', 'logistic', 'poly2', 'poly3'], "
                f"but {func} is given."
            )
            
        self._params = None
        if func == 'reci':
            self.init_guess = [0.6, 0.4, -0.6, 0.6]
            self._func = reci_func
        elif func == 'logistic':
            self.init_guess = [0.6, 0.01, 0.6, 0.01]
            self._func = logistic_func
        elif func == 'poly2':
            self.init_guess = [0.2, 0.4, 0.4, -0.2, -0.4, 0.6]
            self._func = poly2_func
        elif func == 'poly3':
            self.init_guess = [0.2, 0.3, 0.4, 0.1, -0.2, -0.3, -0.1, 0.6]
            self._func = poly3_func
        
        self.func_name = func
        self.loss_tracker = []
        self._loss = None
        self.predicted_prob = None
        self.p0 = p0
        
    @property
    def params(self):
        return self._params
    
    def compute_loss(self, params, func, sequences: list[np.ndarray]):
        total_log_likelihood = 0
        total_n = np.sum([len(seq) for seq in sequences])
        max_length= max([len(seq) for seq in sequences])
        # Prepare arrays
        num_sequences = len(sequences)
        r = np.full((num_sequences, max_length + 1), self.p0)
        s = np.zeros((num_sequences, max_length), dtype=int)
        mask = np.zeros((num_sequences, max_length), dtype=bool)

        # Populate s and mask arrays
        for idx, seq in enumerate(sequences):
            T = len(seq)
            s[idx, :T] = seq
            mask[idx, :T] = True

        for t in range(max_length):
            s_t = s[:, t]
            r_t = r[:, t]

            # Update r_{t+1}
            r_next = func(s_t, r_t, *params)
            r_next = np.clip(r_next, 0, 1)
            r[:, t + 1] = r_next

            # Compute log-likelihood only where mask is True
            valid = mask[:, t]
            total_log_likelihood += np.sum(
                s_t[valid] * np.log(r_t[valid] + 1e-10) + (1 - s_t[valid]) * np.log(1 - r_t[valid] + 1e-10)
            )

        self.loss_tracker.append(-total_log_likelihood / total_n)
        return -total_log_likelihood  # Negative for minimization  

    
    @property
    def loss(self):
        return self._loss
    
    def fit(self, sequences: list[np.ndarray], **kwargs):
        self.p0 = np.sum([seq[1] for seq in sequences]) / len(sequences)
        self.loss_tracker = []
        result = minimize(self.compute_loss, self.init_guess, args=(self._func, sequences))
        self._params = result.x
        
    def predict(self, x, r):
        if self._params is not None:
            return self._func(x, r, *self._params)
        else:
            raise ValueError("Model is not fitted yet.")
    
    def simulate(self, sequences: list[np.ndarray], is_noise: bool = False) -> list[np.ndarray]:            
        simu_seq = []
        for n, seq in enumerate(sequences):
            if is_noise:
                pe = np.random.normal(0, 0.03, len(seq))
            else:
                pe = np.repeat(0, len(seq))
                
            simu = [1]
            curr_p = [self.p0]
            for i in range(len(seq) - 1):
                p = np.clip(curr_p[-1] + pe[i], 0, 1)
                simu.append(np.random.choice([0, 1], p=[1 - p, p]))
                curr_p.append(self.predict(simu[-1], p))
            simu_seq.append(np.array(simu))
        return simu_seq
        
    def get_predicted_prob(self, sequences: list[np.ndarray], is_noise: bool = False) -> list[np.ndarray]:
        predicted_prob = []
        for seq in sequences:
            if is_noise:
                pe = np.random.normal(0, 0.03, len(seq))
            else:
                pe = np.repeat(0, len(seq))
                
            curr_p = [self.p0]
            for i in range(1, len(seq) - 1):
                p = np.clip(curr_p[-1] + pe[i], 0, 1)
                curr_p.append(self.predict(seq[i], p))
            predicted_prob.append(np.array(curr_p, np.float64))
        self.predicted_prob = predicted_prob
        return predicted_prob
    
    def calc_loss(self, sequences: list[np.ndarray], is_report: bool = False):
        if self.predicted_prob is None:
            self.get_predicted_prob(sequences)
            
        loss = 0
        for i in range(len(self.predicted_prob)):
            if len(self.predicted_prob[i]) > 0:
                loss += np.nansum(sequences[i][1:] * np.log(self.predicted_prob[i] - 1e-10) + (1 - sequences[i][1:]) * np.log(1 - self.predicted_prob[i] + 1e-10))
        
        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        if is_report:
            print(f"Continuous Hidden State Model with {self.func_name}:\n"
                  f"  Loss: {self.loss}\n"
                  f"  Parameters: {self.params}.\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray], is_report: bool = False) -> float:
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        if is_report:
            print(f"Continuous Hidden State Model with {self.func_name}:\n"
                  f"  Loss: {loss}\n")
        return loss

    @property
    def loss(self):
        return self._loss
    
    def _check_permanent_silent(self, field_reg):
        for i in range(field_reg.shape[0]):
            I = 0
            for j in range(field_reg.shape[1]):
                if field_reg[i, j] == 0:
                    I += 1
                else:
                    I = 0
                
                if I >= 9:
                    field_reg[i, j:] = 0
        
        return field_reg
    
    def simulate_across_day(self, n_step: int = 26, n_fields: int = 10000, is_noise: bool = False, is_gated: bool = True):
        field_reg = np.zeros((np.sum(n_fields), n_step), np.float64) * np.nan
        field_reg[:, 0] = 1
        p = np.zeros_like(field_reg, np.float64)*np.nan
        p[:, 0] = np.repeat(self.p0, field_reg.shape[0])
        
        is_permanent_silent = np.zeros(field_reg.shape[0], np.int64)
        for i in tqdm(range(1, n_step)):        
            
            if is_noise:
                p[:, i-1] += np.random.normal(0, 0.01, p.shape[0])
                p[:, i-1] = np.clip(p[:, i-1], 0, 1)
            # Generate State
            for j in range(p.shape[0]):
                if is_permanent_silent[j] < 8 or is_gated == False:
                    field_reg[j, i] = np.random.choice([0, 1], p=[1 - p[j, i-1], p[j, i-1]])
            # Predict p
            p[:, i] = self.predict(field_reg[:, i], p[:, i-1])
            
            # Update Field State
            is_permanent_silent[field_reg[:, i] == 0] += 1
            is_permanent_silent[field_reg[:, i] == 1] = 0
            
            # Append new field
            df = int(np.nansum(field_reg[:, i-1]) - np.nansum(field_reg[:, i]))
            
            if df <= 0:
                continue
            
            append_reg = np.zeros((df, n_step), np.float64) * np.nan
            append_reg[:, i] = 1
            
            field_reg = np.vstack([field_reg, append_reg])
            is_permanent_silent = np.concatenate([is_permanent_silent, np.zeros(df)])
            
            append_p = np.full((df, n_step), np.nan)
            append_p[:, i] = self.p0
            p = np.vstack([p, append_p])

        field_identity = np.ones_like(field_reg, np.float64)
        field_identity[np.isnan(field_reg)] = np.nan
        # Set as permanent silent neurons
        if is_gated == False:
            return field_reg, field_identity, p

        for i in range(field_reg.shape[0]):
            I = 0
            for j in range(field_reg.shape[1]):
                if field_reg[i, j] == 0:
                    I += 1
                else:
                    I = 0
                
                if I >= 9:
                    field_reg[i, j:] = 0
                    field_identity[i, j:] = np.nan
                    break
            
        p[np.isnan(field_identity)] = np.nan
        return field_reg, field_identity, p
            
    
if __name__ == "__main__":
    import pickle
    
    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
        sequences = pickle.load(handle)
        
    model = ContinuousHiddenStateModel(func='reci')
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)
    
    model = ContinuousHiddenStateModel(func='logistic')
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)
    
    model = ContinuousHiddenStateModel(func='poly2')
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)