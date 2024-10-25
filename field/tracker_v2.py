import numpy as np
import pickle
import copy as cp
from tqdm import tqdm

class Tracker2d:
    def __init__(self, field_reg: np.ndarray = None):
        if field_reg is not None:
            self._field_reg = cp.deepcopy(field_reg)
            self._refine_reg()
        
            max_disppear = min(self._field_reg.shape[0] - 2, 9)
            max_retained = self._field_reg.shape[0] - 1
        
            self.P1 = np.zeros((max_disppear, max_retained, 2)) # Recover Prob.
        
    def _refine_reg(self):
        for i in range(1, self._field_reg.shape[0]-1): 
            idx = np.where(
                (np.isnan(self._field_reg[i, :])) &
                (self._field_reg[i-1, :] == 1) &
                (self._field_reg[i+1, :] == 1)
            )[0]
            self._field_reg[i, idx] = 1
            
    @property
    def field_reg(self):
        return self._field_reg
    
    def convert_to_sequence(self):
        """
        convert field_reg and related structure to sequences.
        """
        field_reg = self._field_reg
        sequences = []
        for i in range(field_reg.shape[0]-1):
            for j in range(i+1, field_reg.shape[0]):
                if i == 0 and j != field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                    
                elif i != 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                
                elif i == 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                else:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                    
        # Cut the sequence if 0 continuously occurs for 8 times
        for i in range(len(sequences)-1):
            num_zero = 0
            is_split = False
            split_point = []
            for j in range(sequences[i].shape[0]):
                if num_zero == 8:
                    is_split = True
                
                if sequences[i][j] == 0:
                    num_zero += 1
                else:
                    if is_split:
                        split_point.append(j)
                        is_split = False
                    num_zero = 0
            
            if len(split_point) != 0:      
                split_point = [0] + split_point + [sequences[i].shape[0]]
                for j in range(1, len(split_point)-1):
                    sequences.append(sequences[i][split_point[j]:split_point[j+1]])
                        
                sequences[i] = sequences[i][split_point[0]:split_point[1]]
        
        for i in range(len(sequences)-1, -1, -1):
            if np.sum(sequences[i]) == 0:
                sequences.pop(i)
                continue
    
            for j in range(sequences[i].shape[0]):
                if sequences[i][j] == 1:
                    sequences[i] = sequences[i][j:]

                    if sequences[i].shape[0] <= 1:
                        sequences.pop(i)
                    break
        
        return sequences
    
    @staticmethod
    def reconstruct_reg(field_reg, thre: int = 5):
        tracker = Tracker2d(field_reg)
        field_reg = tracker.field_reg
        
        sequences = []
        n_session = []
        for i in range(field_reg.shape[0]-1):
            for j in range(i+1, field_reg.shape[0]):
                if i == 0 and j != field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        n_session.append(i)
                    
                elif i != 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        n_session.append(i)
                
                elif i == 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        n_session.append(i)
                else:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        n_session.append(i)
                    
        # Cut the sequence if 0 continuously occurs for 8 times
        for i in range(len(sequences)-1):
            num_zero = 0
            is_split = False
            split_point = []
            for j in range(sequences[i].shape[0]):
                if num_zero == 8:
                    is_split = True
                
                if sequences[i][j] == 0:
                    num_zero += 1
                else:
                    if is_split:
                        split_point.append(j)
                        is_split = False
                    num_zero = 0
            
            if len(split_point) != 0:      
                split_point = [0] + split_point + [sequences[i].shape[0]]
                for j in range(1, len(split_point)-1):
                    sequences.append(sequences[i][split_point[j]:split_point[j+1]])
                    n_session.append(n_session[i] + split_point[j])
                        
                sequences[i] = sequences[i][split_point[0]:split_point[1]]
        
        for i in range(len(sequences)-1, -1, -1):
            if np.sum(sequences[i]) == 0:
                sequences.pop(i)
                n_session.pop(i)
                continue
    
            for j in range(sequences[i].shape[0]):
                if sequences[i][j] == 1:
                    sequences[i] = sequences[i][j:]
                    n_session[i] = n_session[i]+j

                    if sequences[i].shape[0] <= thre-1:
                        sequences.pop(i)
                        n_session.pop(i)
                    break
        
        # Reconstuct the sequence
        reconstructed_reg = np.zeros((field_reg.shape[0], len(sequences))) * np.nan
        for i, seq in enumerate(sequences):
            reconstructed_reg[n_session[i]:n_session[i]+seq.shape[0], i] = seq
        return reconstructed_reg
    
    @staticmethod
    def convert_for_glm(field_reg, glm_params, least_length=5, is_seq_format=False) -> tuple[np.ndarray, np.ndarray]:
        
        sequences = []
        param_sequences = []
        for i in range(field_reg.shape[0]-1):
            for j in range(i+1, field_reg.shape[0]):
                if i == 0 and j != field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        param_sequences.append(glm_params[i:j+1, k, :])
                    
                elif i != 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        param_sequences.append(glm_params[i:j+1, k, :])
                
                elif i == 0 and j == field_reg.shape[0]-1:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        param_sequences.append(glm_params[i:j+1, k, :])
                else:
                    idx = np.where(
                        (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                        (np.isnan(field_reg[i-1, :]) == True) &
                        (np.isnan(field_reg[j+1, :]) == True)
                    )[0]
                    for k in idx:
                        sequences.append(field_reg[i:j+1, k].astype(np.int64))
                        param_sequences.append(glm_params[i:j+1, k, :])
                    
        # Cut the sequence if 0 continuously occurs for 8 times
        for i in range(len(sequences)-1):
            num_zero = 0
            is_split = False
            split_point = []
            for j in range(sequences[i].shape[0]):
                if num_zero == 8:
                    is_split = True
                
                if sequences[i][j] == 0:
                    num_zero += 1
                else:
                    if is_split:
                        split_point.append(j)
                        is_split = False
                    num_zero = 0
            
            if len(split_point) != 0:      
                split_point = [0] + split_point + [sequences[i].shape[0]]
                for j in range(1, len(split_point)-1):
                    sequences.append(sequences[i][split_point[j]:split_point[j+1]])
                    param_sequences.append(param_sequences[i][split_point[j]:split_point[j+1], :])
                        
                sequences[i] = sequences[i][split_point[0]:split_point[1]]
                param_sequences[i] = param_sequences[i][split_point[0]:split_point[1], :]
        
        for i in range(len(sequences)-1, -1, -1):
            if np.sum(sequences[i]) == 0:
                sequences.pop(i)
                param_sequences.pop(i)
                continue
    
            for j in range(sequences[i].shape[0]):
                if sequences[i][j] == 1:
                    sequences[i] = sequences[i][j:]
                    param_sequences[i] = param_sequences[i][j:, :]

                    if sequences[i].shape[0] <= 1:
                        sequences.pop(i)
                        param_sequences.pop(i)
                    break
                
                
        seq_lengths = np.array([seq.shape[0] for seq in sequences])
        idx = np.where(seq_lengths >= least_length)[0]
        sequences = [sequences[i] for i in idx]
        param_sequences = [param_sequences[i] for i in idx]
        
        if is_seq_format:
            for i in range(len(sequences)):
                param_sequences[i][np.isnan(param_sequences[i])] = 0
            return sequences, param_sequences
        
        for i in range(len(sequences)):
            attached_column = np.arange(sequences[i].shape[0])
            param_sequences[i] = np.concatenate([param_sequences[i], attached_column[:, np.newaxis]], axis=1)

        for i in range(len(sequences)):
            if len(sequences[i]) != param_sequences[i].shape[0]:
                raise ValueError('Length of sequences and param_sequences are not same.')
        
        # Quality Control
        Y = np.concatenate([seq[1:] for seq in sequences])
        X = np.concatenate([param_seq[:-1, :] for param_seq in param_sequences], axis=0)
        
        sum = np.sum(X, axis=1)
        idx = np.where(np.isnan(sum) == False)[0]
    
        return X[idx, :], Y[idx]
    
    def calc_P1(self, sequences: list[np.ndarray] = None) -> np.ndarray:
        if sequences is None:
            sequences = self.convert_to_sequence()

        max_length = max([len(seq) for seq in sequences])

        probs = np.zeros((max_length, max_length, 2))

        for seq in sequences:
            for i in range(seq.shape[0]-1):
                act, inact = np.sum(seq[:i+1]), i+1 - np.sum(seq[:i+1])
                probs[int(act), int(inact), int(seq[i+1])] += 1
        
        probs = probs[1:, :, 1]/np.sum(probs[1:, :, :], axis=2)
        return probs
    
    def calc_P3(self, sequences: list[np.ndarray] = None) -> np.ndarray:
        if sequences is None:
            sequences = self.convert_to_sequence()

        max_length = max([len(seq) for seq in sequences])

        probs = np.zeros((max_length, max_length, 2))
        
        for seq in sequences:
            HA, A, I = 0, 1, 0
            for i in range(seq.shape[0]-1):
                if seq[i] == 1 and seq[i+1] == 0:
                    probs[A-1, I, 0] += 1
                    HA = A
                    A = 0
                    I += 1
                elif seq[i] == 0 and seq[i+1] == 0:
                    probs[HA-1, I, 0] += 1
                    I += 1
                elif seq[i] == 0 and seq[i+1] == 1:
                    probs[HA-1, I, 1] += 1
                    A += 1
                    I = 0
                    HA = 0
                elif seq[i] == 1 and seq[i+1] == 1:
                    probs[A-1, I, 1] += 1
                    A += 1
        
        print(probs)
        idx = np.where(np.sum(probs, axis=2) <= 5)
        probs = probs[:, :, 1]/np.sum(probs[:, :, :], axis=2)
        probs[idx] = np.nan
        return probs
        
    @staticmethod
    def recovery_prob2d(field_reg: np.ndarray) -> np.ndarray:
        tracker = Tracker2d(field_reg)
        return tracker.calc_P1()
    
    @staticmethod
    def get_joint_prob(sequences: list[np.ndarray]) -> np.ndarray:
        tracker = Tracker2d()
        return tracker.calc_P1(sequences)
    
    @staticmethod
    def retained_dur_dependent_prob(sequences: list[np.ndarray]) -> np.ndarray:
        tracker = Tracker2d()
        return tracker.calc_P3(sequences)
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    import pickle
    from tqdm import tqdm
    import copy as cp

    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10224\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs = Tracker2d.recovery_prob2d(field_reg)

    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs2 = Tracker2d.recovery_prob2d(field_reg)

    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10209\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs3 = Tracker2d.recovery_prob2d(field_reg)


    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10212\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs4 = Tracker2d.recovery_prob2d(field_reg)
    
    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10209\Maze1-2-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs5 = Tracker2d.recovery_prob2d(field_reg)


    with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10212\Maze1-2-footprint\trace_mdays_conc.pkl", 'rb') as handle:
        trace = pickle.load(handle)

    field_reg = cp.deepcopy(trace['field_reg'][0:, :])

    probs6 = Tracker2d.recovery_prob2d(field_reg)



    # plot 3d surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.arange(probs.shape[1]), np.arange(probs.shape[0]))
    ax.plot_surface(X, Y, (probs + probs2 + probs3 + probs4)/4, cmap='rainbow', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Inaction')
    ax.set_ylabel('Action')
    ax.view_init(azim=-30, elev=30)
    ax.set_zlim(0, 1)
    plt.show()
