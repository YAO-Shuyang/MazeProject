import numpy as np
import matplotlib.pyplot as plt
import torch

import pickle
from tqdm import tqdm
import copy as cp

with open(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227\Maze1-footprint\trace_mdays_conc.pkl", 'rb') as handle:
    trace = pickle.load(handle)

field_reg = cp.deepcopy(trace['field_reg'][0:, :])

sequences = []

for i in range(1, field_reg.shape[0]-1): 
    idx = np.where(
        (np.isnan(field_reg[i, :])) &
        (field_reg[i-1, :] == 1) &
        (field_reg[i+1, :] == 1)
    )[0]
    field_reg[i, idx] = 1

for i in range(field_reg.shape[0]-1):
    for j in range(i+1, field_reg.shape[0]):
        if i == 0 and j != field_reg.shape[0]-1:
            idx = np.where(
                (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                (np.isnan(field_reg[j+1, :]) == True)
            )[0]
            for k in idx:
                sequences.append(field_reg[i:j+1, k])
                    
        elif i != 0 and j == field_reg.shape[0]-1:
            idx = np.where(
                (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                (np.isnan(field_reg[i-1, :]) == True)
            )[0]
            for k in idx:
                sequences.append(field_reg[i:j+1, k])
                
        elif i == 0 and j == field_reg.shape[0]-1:
            idx = np.where(
                (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False)
            )[0]
            for k in idx:
                sequences.append(field_reg[i:j+1, k])
                
        else:
            idx = np.where(
                (np.isnan(np.sum(field_reg[i:j+1, :], axis=0)) == False) &
                (np.isnan(field_reg[i-1, :]) == True) &
                (np.isnan(field_reg[j+1, :]) == True)
            )[0]
            for k in idx:
                sequences.append(field_reg[i:j+1, k])

print(len(sequences))    
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
    
print(len(sequences))

transition_idx = np.array([seq.shape[0] for seq in sequences])
for i in range(len(transition_idx)):
    idx = np.where(np.ediff1d(sequences[i]) != 0)[0]
    if idx.shape[0] > 0:
        transition_idx[i] = idx[0]
        
nums = np.concatenate([[transition_idx.shape[0]], [np.where(transition_idx == i)[0].shape[0] for i in range(14)]])

probs = np.zeros((field_reg.shape[0], field_reg.shape[0], 2))

for seq in sequences:
    for i in range(seq.shape[0]-1):
        act, inact = np.sum(seq[:i+1]), i+1 - np.sum(seq[:i+1])
        probs[int(act), int(inact), int(seq[i+1])] += 1
        
probs = probs[1:, :, 1]/np.sum(probs[1:, :, :], axis=2)

# plot 3d surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(probs.shape[1]), np.arange(probs.shape[0]))
ax.plot_surface(X, Y, probs, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('Inaction')
ax.set_ylabel('Action')
ax.view_init(azim=-30, elev=30)
ax.set_zlim(0, 1)
plt.show()