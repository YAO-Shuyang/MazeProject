# Continuous hidden state model

import numpy as np
from scipy.optimize import minimize

def reci_func(x, r, L0, b0, L1, b1):
    return np.clip(
        x * (1 - L0 / (r + b0 + L0)) + (1-x) * L1 / (1 - r + b1 + L1), 
        0, 
        1
    )

def logistic_func(x, r, k0, r01, k1, r02):
    return np.clip(
        x * (1 / (1 + np.exp(-k0 * (r - r01)))) + (1-x) * (1 - 1 / (1 + np.exp(-k1 * (1-r - r02))))
        0, 1
    )

def poly2_func(x, r, a1, b1, c1, a2, b2, c2):
    return np.clip(
        x * (a1 * r**2 + b1 * r + c1) + (1-x) * (a2 * (1-r)**2 + b2 * (1-r) + c2),
        0, 1
    )

class ContinuousHiddenStateModel:
    def __init__(self, func: str, p0: str = 0.6) -> None:
        if func not in ['reci', 'logistic', 'poly2']:
            raise ValueError(
                f"func must be one of ['reci', 'logistic', 'poly2'], "
                f"but {func} is given."
            )
            
        self._params = None
        if func == 'reci':
            self.init_guess = [0.6, 0.01, 0.6, 0.01]
            self._func = reci_func
        elif func == 'logistic':
            self.init_guess = [0.6, 0.01, 0.6, 0.01]
            self._func = logistic_func
        elif func == 'poly2':
            self.init_guess = [0.3, 0.3, 0.4, -0.3, -0.3, 0.4]
            self._func = poly2_func
        
        self.func_name = func
        self.loss_tracker = []
        self.p0 = p0
        
    @property
    def params(self):
        return self._params
    
    def calc_loss(self, params, func, sequences: list[np.ndarray]):
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
            r_next = np.clip(r_next, 1e-8, 1 - 1e-8)
            r[:, t + 1] = r_next

            # Compute log-likelihood only where mask is True
            valid = mask[:, t]
            total_log_likelihood += np.sum(
                s_t[valid] * np.log(r_t[valid]) + (1 - s_t[valid]) * np.log(1 - r_t[valid])
            )

        self.loss_tracker.append(-total_log_likelihood / total_n)
        return -total_log_likelihood  # Negative for minimization  
    
    @property
    def loss(self):
        return self.loss_tracker[-1]
    
    def fit(self, sequences: list[np.ndarray], **kwargs):
        self.loss_tracker = []
        result = minimize(self.calc_loss, self.init_guess, args=(self._func, sequences))
        print(
            f"Continuous Hidden State Model with {self.func_name}:\n"
            f"  Loss: {self.loss}\n"
            f"  Parameters: {result.x}"
        )
        self._params = result.x
        
    def predict(self, x, r):
        if self._params is not None:
            return self._func(x, r, *self._params)
        else:
            raise ValueError("Model is not fitted yet.")
    
    def generate_single_sequence(self, n: int) -> np.ndarray:
        seq = [1]
        curr_p = [self.p0]
        
        for _ in range(n-1):
            curr_p.append(self.predict(seq[-1], curr_p[-1]))
            seq.append(np.random.choice([0, 1], p=[1 - curr_p[-1], curr_p[-1]]))
        return np.array(seq)
    
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        simu_seq = []
        for seq in sequences:
            simu_seq.append(self.generate_single_sequence(len(seq)))
        return simu_seq
        