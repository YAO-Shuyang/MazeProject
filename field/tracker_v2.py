import numpy as np
import pickle
import copy as cp
from tqdm import tqdm

class Tracker2d:
    def __init__(self, field_reg: np.ndarray):
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
    
    def convert_to_sequence(self):
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
                    
    def calc_P1(self) -> np.ndarray:
        field_reg = self._field_reg
        sequences = self.convert_to_sequence()

        probs = np.zeros((field_reg.shape[0], field_reg.shape[0], 2))

        for seq in sequences:
            for i in range(seq.shape[0]-1):
                act, inact = np.sum(seq[:i+1]), i+1 - np.sum(seq[:i+1])
                probs[int(act), int(inact), int(seq[i+1])] += 1
        
        probs = probs[1:, :, 1]/np.sum(probs[1:, :, :], axis=2)
        return probs
        
    @staticmethod
    def recovery_prob2d(field_reg: np.ndarray):
        tracker = Tracker2d(field_reg)
        return tracker.calc_P1()
        
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
