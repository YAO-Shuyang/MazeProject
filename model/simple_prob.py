# Only consider retention and recover probability.
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from mylib.field.tracker_v2 import Tracker2d
from tqdm import tqdm

def retention_func(x, L, b):
    return 1 - L / (x + b)

def recovery_func(x, L, b):
    return L / (x + b)

class EqualRateDriftModel:
    """
    This model assumes that the drift rate is equal for retention and recovery.
    """
    def __init__(self):
        self._params = None
        self._loss = None
        self.predicted_prob = None
        
    @property
    def params(self):
        return self._params
    
    @property
    def loss(self):
        return self._loss
    
    def fit(self, sequences):
        n_ones = np.sum([np.sum(seq) for seq in sequences])
        n_all = np.sum([len(seq) for seq in sequences])
        self._params = (n_ones / n_all, )
        
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        simu_seq = []
        for seq in sequences:
            simu_seq.append(
                np.concatenate([[1], np.random.choice(
                    [0, 1], p=[1 - self.params[0], self.params[0]],
                    size=len(seq)-1
                )]
            ))
    
        return simu_seq
    
    def get_predicted_prob(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self._params is None:
            self.fit(sequences)
            
        self.predicted_prob = [np.repeat(self.params[0], len(seq)-1) for seq in sequences]
        return self.predicted_prob
    
    def calc_loss(self, sequences: list[np.ndarray]):
        if self.predicted_prob is None:
            self.get_predicted_prob(sequences)
            
        loss = 0
        for i in range(len(self.predicted_prob)):
            if len(self.predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(self.predicted_prob[i] - 1e-10) + (1 - sequences[i][1:]) * np.log(1 - self.predicted_prob[i] + 1e-10))
        
        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Simple Drift Model:\n"
              f"  Loss: {self.loss}\n"
              f"  Parameters: {self.params}.\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray]):
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        print(f"Simple Drift Model:\n"
              f"  Loss: {loss}\n"
              f"  Parameters: {self.params}.\n")
        return loss

class TwoProbDriftModel:
    """
    This model assumes that the drift rate is equal for retention and recovery.
    """
    def __init__(self):
        self._params = None
        self._loss = None
        self.predicted_prob = None
        
    @property
    def params(self):
        return self._params
    
    @property
    def loss(self):
        return self._loss
    
    def fit(self, sequences):
        X = np.concatenate([seq[:-1] for seq in sequences])
        Y = np.concatenate([seq[1:] for seq in sequences])
        self._params = (
            np.where((Y==1)&(X==1))[0].shape[0] / np.where(X==1)[0].shape[0],
            np.where((Y==1)&(X==0))[0].shape[0] / np.where(X==0)[0].shape[0]
        )
        
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        simu_seq = []
        for seq in sequences:
            simu = [1]
            for i in range(len(seq)-1):
                PA = self.params[0] if simu[-1] == 1 else self.params[1]
                simu_val = np.random.choice([0, 1], p=[1 - PA, PA])
                simu.append(simu_val)
            simu_seq.append(np.array(simu))
    
        return simu_seq
    
    def get_predicted_prob(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self._params is None:
            self.fit(sequences)
        
        ps = np.array([self.params[1], self.params[0]])
        self.predicted_prob = [ps[seq][:-1] for seq in sequences]
        return self.predicted_prob
    
    def calc_loss(self, sequences: list[np.ndarray]):
        if self.predicted_prob is None:
            self.get_predicted_prob(sequences)
            
        loss = 0
        for i in range(len(self.predicted_prob)):
            if len(self.predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(self.predicted_prob[i] - 1e-10) + (1 - sequences[i][1:]) * np.log(1 - self.predicted_prob[i] + 1e-10))
        
        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Two Probability Drift Model:\n"
              f"  Loss: {self.loss}\n"
              f"  Parameters: {self.params}.\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray]):
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        print(f"Two Probability Drift Model:\n"
              f"  Loss: {loss}\n"
              f"  Parameters: {self.params}.\n")
        return loss

def count(P1: np.ndarray, P2: np.ndarray, sequence: np.ndarray):
    sequence = sequence.astype(np.int64)
    A, I = 1, 0
    
    for i in range(len(sequence)-1):
        if A != 0 and I == 0:
            P1[A-1, sequence[i+1]] += 1
        elif A == 0 and I != 0:
            P2[I-1, sequence[i+1]] -= 1

        if sequence[i+1] == 1:
            A += 1
            I = 0
        else:
            I += 1
            A = 0
            
    return P1, P2
        
class TwoProbabilityIndependentModel:
    """
    This model proposes two probabilities are independent.
    """
    def __init__(self):
        self._params_retention = None
        self._params_recovery = None
        self.p0 = None
        self.predicted_prob = None
        self._loss = None
        
    @property
    def params_retention(self):
        return self._params_retention
    
    @property
    def params_recovery(self):
        return self._params_recovery

    def fit(self, sequences):
        max_length = max([len(seq) for seq in sequences])
        P1 = np.zeros((max_length-1, 2), dtype=np.float64)
        P2 = np.zeros((max_length-2, 2), dtype=np.float64)
        
        for seq in tqdm(sequences):
            P1, P2 = count(P1, P2, seq)
            
        P1 = P1[:, 1] / np.sum(P1, axis=1)
        P2 = P2[:, 1] / np.sum(P2, axis=1)
        
        self._params_retention = curve_fit(
            retention_func, np.arange(1, P1.shape[0]+1), P1, p0=[0.5, 0.5]
        )[0]
        
        self._params_recovery = curve_fit(
            recovery_func, np.arange(1, P2.shape[0]+1), P2, p0=[0.5, 0.5]
        )[0]

        self.p0 = retention_func(1, *self.params_retention)
    
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        self.get_predicted_prob(sequences)
        
        simu_seq = []
        for n, seq in enumerate(sequences):
            simu = [1]
            
            for i in range(len(seq)-1):
                PA = self.predicted_prob[n][i]
                simu_val = np.random.choice([0, 1], p=[1 - PA, PA])
                simu.append(simu_val)

            simu_seq.append(np.array(simu))
        
        return simu_seq
    
    def get_predicted_prob(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self._params_retention is None or self._params_recovery is None:
            self.fit(sequences)
            
        prob_seq = []
        
        for seq in sequences:
            prob = []
            A, I = 1, 0
            
            for i in range(len(seq)-1):
                if A != 0 and I == 0:
                    PA = retention_func(A, *self.params_retention)
                elif A == 0 and I != 0:
                    PA = recovery_func(I, *self.params_recovery)
                
                prob.append(PA)
                if seq[i+1] == 1:
                    A = A + 1
                    I = 0
                else:
                    I = I + 1
                    A = 0
            prob_seq.append(np.array(prob))
        
        self.predicted_prob = prob_seq
        return prob_seq
    
    def calc_loss(self, sequences: list[np.ndarray]):
        self.get_predicted_prob(sequences)
            
        loss = 0
        for i in range(len(self.predicted_prob)):
            if len(self.predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(self.predicted_prob[i] - 1e-10) + (1 - sequences[i][1:]) * np.log(1 - self.predicted_prob[i] + 1e-10))
        
        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Retention + Recovery Model:\n"
              f"  Loss: {self.loss}\n"
              f"  Retention Parameters: {self.params_retention}\n"
              f"  Recovery Parameters: {self.params_recovery}.\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray]):
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        print(f"Retention + Recovery Model:\n"
              f"  Loss: {loss}\n"
              f"  Retention Parameters: {self.params_retention}\n"
              f"  Recovery Parameters: {self.params_recovery}.\n")
        return loss
        
    @property
    def loss(self):
        return self._loss

def surface2d(data, b, u, v, w):
    y, x = data
    return (x + b) / (u*y + v*(x-1) + w)

class JointProbabilityModel:
    def __init__(self, p0: float = 0.55):
        """
        Initialize the Joint Probability model.

        Parameters:
            p0 (float): Initial probability of an action.
        """
       
        self.p0 = p0
        self.predicted_prob = None
        self._loss = None
        self._params = None
        
    @property
    def params(self):
        return self._params
    
    def fit(self, sequences):
        joint_prob = Tracker2d.get_joint_prob(sequences)
        
        I, A = np.meshgrid(np.arange(joint_prob.shape[1]), np.arange(1, joint_prob.shape[0]+1))
        
        I, A, P = I.flatten(), A.flatten(), joint_prob.flatten()
        
        mask = np.where(np.isnan(P) == False)[0]
        I, A, P = I[mask], A[mask], P[mask]
        
        def objective_function(params):
            return np.sum((surface2d((I, A), *params) - P) ** 2)
    
        bounds = [(-1 + 1e-10, 100), (1e-10, 100), (1e-10, 100), (1e-10, 100)]
        result = differential_evolution(objective_function, bounds, maxiter=10000)
        self._params = result.x
    
    def simulate(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self.predicted_prob is None:
            self.get_predicted_prob(sequences)
            
        simu_seq = []
        
        for n, seq in enumerate(sequences):
            simu = [1]
            
            for i in range(len(seq)-1):
                P = self.predicted_prob[n][i]
                simu_val = np.random.choice([0, 1], p=[1 - P, P])
                simu.append(simu_val)
            
            simu_seq.append(np.array(simu))
            
        return simu_seq
    
    def get_predicted_prob(self, sequences: list[np.ndarray]) -> list[np.ndarray]:
        if self._params is None:
            self.fit(sequences)
            
        prob_seq = []
        
        for seq in sequences:
            prob = []
            A, I = 1, 0
            
            for i in range(len(seq)-1):
                PA = surface2d((I, A), *self.params)
                
                prob.append(PA)
                if seq[i+1] == 1:
                    A += 1
                else:
                    I += 1
        
            prob_seq.append(np.array(prob))
        
        self.predicted_prob = prob_seq
        return prob_seq
    
    def calc_loss(self, sequences: list[np.ndarray]) -> float:
        self.get_predicted_prob(sequences)
            
        loss = 0
        for i in range(len(self.predicted_prob)):
            if len(self.predicted_prob[i]) > 0:
                loss += np.sum(sequences[i][1:] * np.log(self.predicted_prob[i] - 1e-10) + (1 - sequences[i][1:]) * np.log(1 - self.predicted_prob[i] + 1e-10))
        
        n_total = np.sum([len(seq)-1 for seq in sequences if len(seq) > 1])
        self._loss = -loss / n_total
        print(f"Joint Probability Model:\n"
              f"  Loss: {self.loss}\n"
              f"  Parameters: {self.params}.\n")
        return self._loss

    def calc_loss_along_seq(self, sequences: list[np.ndarray]):
        max_length = max([len(seq) for seq in sequences])
        predicted_p = self.get_predicted_prob(sequences)
        padded_p = np.zeros((len(predicted_p), max_length-1)) * np.nan
        padd_seq = np.zeros((len(predicted_p), max_length-1)) * np.nan
        for i in range(len(predicted_p)):
            padded_p[i, :len(predicted_p[i])] = predicted_p[i]
            padd_seq[i, :len(predicted_p[i])] = sequences[i][1:]
        
        dloss = padd_seq * np.log(padded_p + 1e-10) + (1 - padd_seq) * np.log(1 - padded_p + 1e-10)
        loss = -np.nanmean(dloss, axis=0)
        print(f"Joint Probability Model:\n"
              f"  Loss: {loss}\n"
              f"  Parameters: {self.params}.\n")
        return loss
    
    @property
    def loss(self):
        return self._loss
        

if __name__ == "__main__":
    import pickle
    
    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\test\demo_seq.pkl", 'rb') as handle:
        sequences = pickle.load(handle)
        
    model = EqualRateDriftModel()
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)
    
    model = TwoProbabilityIndependentModel()
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)
    
    model = JointProbabilityModel()
    model.fit(sequences)
    model.get_predicted_prob(sequences)
    model.calc_loss(sequences)
    model.simulate(sequences)