import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mylib.stats.indeptest import indept_field_evolution_chi2, indept_field_evolution_mutual_info, indept_field_evolution_CI
from mylib.maze_utils3 import GetDMatrices
import copy as cp

class RegisteredField(object):
    def __init__(self, field_reg: np.ndarray) -> None:
        self._content = field_reg
        self._session = field_reg.shape[0]
                
    def report(self, thre: int = 4):
        field_reg = self._content
        duration = np.arange(field_reg.shape[0])  # Retained duration, silent duration, not detected duration
        on_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State ON with duration t -> State ON/OFF/NOT DETECTED on the next sessions
        off_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State OFF on Session t -> State ON DETECTED on Session t+1
        silent_num = 0

        for i in range(field_reg.shape[0]-1):
            for j in range(i+1, field_reg.shape[0]):
                if i == 0:

                    base_num = np.where((np.sum(field_reg[i:j, :], axis=0)==j-i)&(field_reg[j, :] == 0))[0]
                    active_num = np.where(np.sum(field_reg[i:j+1, :], axis=0)==j-i+1)[0]
                    on_next_prob[j-i, 0] += active_num.shape[0]
                    on_next_prob[j-i, 1] += base_num.shape[0]
                    
                elif i == 1:
                    base_num = np.where((np.sum(field_reg[i:j, :], axis=0)==j-i)&(field_reg[j, :] == 0)&(field_reg[i-1, :] != 1))[0]
                    active_num = np.where((np.sum(field_reg[i:j+1, :], axis=0)==j-i+1)&(field_reg[i-1, :] != 1))[0]
                    on_next_prob[j-i, 0] += active_num.shape[0]
                    on_next_prob[j-i, 1] += base_num.shape[0]
                
                else:
                    base_num = np.where((np.sum(field_reg[i:j, :], axis=0)==j-i)&(field_reg[j, :] == 0)&
                                        ((field_reg[i-1, :] == 0)|((np.isnan(field_reg[i-1, :]))&
                                                                     (field_reg[i-2, :] != 1))))[0]
                    active_num = np.where((np.sum(field_reg[i:j+1, :], axis=0)==j-i+1)&
                                          ((field_reg[i-1, :] == 0)|((np.isnan(field_reg[i-1, :]))&
                                                                     (field_reg[i-2, :] != 1))))[0]
                    on_next_prob[j-i, 0] += active_num.shape[0]
                    on_next_prob[j-i, 1] += base_num.shape[0]
                
                if j == i + 1:
                    continue
                     
                recover_num = np.where((np.sum(field_reg[i+1:j, :], axis=0)==0)&(field_reg[i, :] == 1)&(field_reg[j, :] == 1))[0]
                non_recover_num = np.where((np.nansum(field_reg[i+1:j, :], axis=0)==0)&(field_reg[i, :] == 1)&(field_reg[j, :] == 0))[0]
                off_next_prob[j-i-1, 0] += recover_num.shape[0]
                off_next_prob[j-i-1, 1] += non_recover_num.shape[0]
                

            idx = np.where((field_reg[i, :] == 1)&(field_reg[i+1, :] == 0))[0]
            silent_num += idx.shape[0]
        
        retained_prob = on_next_prob[:, 0] / np.sum(on_next_prob, axis=1)

        global_recover_prob = off_next_prob[:, 0] / silent_num
        conditional_recover_prob = off_next_prob[:, 0] / np.sum(off_next_prob, axis=1)
    
        retained_prob[np.where(np.sum(on_next_prob, axis=1) <= thre)[0]] = np.nan
        global_recover_prob[np.where(off_next_prob[:, 0] <= thre)[0]] = np.nan
        conditional_recover_prob[np.where(np.sum(off_next_prob, axis=1) <= thre)[0]] = np.nan

        return duration, retained_prob, conditional_recover_prob, global_recover_prob, np.sum(on_next_prob, axis=1), np.sum(off_next_prob, axis=1)
    
    @staticmethod
    def conditional_prob(field_reg, thre: int = 4):
        Reg = RegisteredField(field_reg)
        return Reg.report(thre=thre)
        
'''    
def conditional_prob(trace: dict = None, field_reg: np.ndarray = None, thre: int = 5):
    """
    if trace['Stage'] in ['Stage 1', 'Stage 1+2'] or trace['maze_type'] == 2:
        field_reg = cp.deepcopy(trace['field_reg'])[2:, :]
    else:
    """
    if field_reg is None:
        field_reg = cp.deepcopy(trace['field_reg'])

    duration = np.arange(field_reg.shape[0])  # Retained duration, silent duration, not detected duration
    on_next_prob = np.zeros((field_reg.shape[0], 4), dtype=np.int64) # State ON with duration t -> State ON/OFF/NOT DETECTED on the next sessions
    off_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State OFF on Session t -> State ON DETECTED on Session t+1
    nod_next_prob = np.zeros((field_reg.shape[0], 2), dtype=np.int64) # State NOT DETECTED on Session t -> State ON DETECTED on Session t+1
    silent_num = 0
    no_detect_num = 0
    
    for j in tqdm(range(field_reg.shape[1])):
        count_one, count_nil = 0, 0 # turn on and turn off
        count_nod = 0 # no detect
        is_disappear = False
        for i in range(field_reg.shape[0]):
            if np.isnan(field_reg[i, j]):
                if count_one >= 1:
                    on_next_prob[count_one, 2] += 1
                    count_nod += 1
                    no_detect_num += 1
                elif count_one == 0 and count_nil == 0 and count_nod >= 1:
                    count_nod += 1
                elif count_nil >= 1:
                    no_detect_num += 1
                
                count_one = 0
                count_nil = 0
                is_disappear = False
            elif field_reg[i, j] == 1:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 1] += 1
                    count_nod = 0
                if count_one >= 1:
                    on_next_prob[count_one, 1] += 1
                elif count_one == 0:
                    if count_nil >= 1 and is_disappear:
                        off_next_prob[count_nil, 0] += 1
                        count_nil = 0
                
                is_disappear = True
                count_one += 1
            elif field_reg[i, j] == 0:
                if count_nod >= 1:
                    nod_next_prob[count_nod, 0] += 1
                    count_nod = 0
                if count_nil == 0:
                    if count_one >= 1:
                        on_next_prob[count_one, 0] += 1
                        count_one = 0
                        silent_num += 1
                else:
                    off_next_prob[count_nil, 1] += 1

                count_nil += 1
            else:
                raise ValueError(f"field reg contains invalid value {field_reg[i, j]} at row {i} column {j}. ")

        if count_one >= 1:
            on_next_prob[count_one-1, 3] += 1
        
    retained_prob = on_next_prob[:, 1] / np.sum(on_next_prob[:, :2], axis=1)
    nodetect_prob = on_next_prob[:, 2] / (np.sum(on_next_prob[:, :3], axis=1) + np.concatenate([[0], on_next_prob[:-1, 3]]))

    global_recover_prob = off_next_prob[:, 0] / silent_num
    conditional_recover_prob = off_next_prob[:, 0] / np.sum(off_next_prob, axis=1)
    redetect_prob = nod_next_prob[:, 1] / (nod_next_prob[:, 0] + nod_next_prob[:, 1])
    redetect_frac = np.sum(nod_next_prob, axis=1) / no_detect_num
    
    retained_prob[np.where(np.sum(on_next_prob[:, :2], axis=1) <= thre)[0]] = np.nan
    nodetect_prob[np.where(np.sum(on_next_prob[:, :3], axis=1) <= thre)[0]] = np.nan
    global_recover_prob[np.where(off_next_prob[:, 0] <= thre)[0]] = np.nan
    conditional_recover_prob[np.where(np.sum(off_next_prob, axis=1) <= thre)[0]] = np.nan
    redetect_prob[np.where(nod_next_prob[:, 1] <= thre)[0]] = np.nan

    #return duration, retained_prob, nodetect_prob, conditional_recover_prob, global_recover_prob, redetect_prob, redetect_frac, np.sum(on_next_prob[:, :2], axis=1), np.sum(off_next_prob, axis=1)
    return duration, retained_prob, conditional_recover_prob, global_recover_prob, np.sum(on_next_prob[:, :2], axis=1), np.sum(off_next_prob, axis=1)
'''

from numba import jit

@jit(nopython=True)
def get_evolve_event_label(evolve_event: np.ndarray):
    n = evolve_event.shape[0]
    # Create an array of powers of 2, reversed, to match the binary representation
    powers_of_two = 2 ** np.arange(n)
    
    # Perform dot product
    return np.dot(powers_of_two.astype(np.float64), evolve_event)

@jit(nopython=True)
def get_evolve_event_pairs(
    i: int,
    j: int,
    field_reg: np.ndarray,
    sib_field_pairs: np.ndarray,
    non_field_pairs: np.ndarray,
):
    sib_num, non_num = sib_field_pairs.shape[0], non_field_pairs.shape[0]
    
    evol_event_sib = np.reshape(np.concatenate((get_evolve_event_label(field_reg[i:j, sib_field_pairs[:, 0]]), 
                                                get_evolve_event_label(field_reg[i:j, sib_field_pairs[:, 1]]))), (2, sib_num)).T

    evol_event_sib = evol_event_sib[np.where((np.isnan(evol_event_sib[:, 0]) == False)&
                                             (np.isnan(evol_event_sib[:, 1]) == False)&
                                             (evol_event_sib[:, 0] != 0)&
                                             (evol_event_sib[:, 1] != 0))[0], :]
    
    
    evol_event_non = np.reshape(np.concatenate((get_evolve_event_label(field_reg[i:j, non_field_pairs[:, 0]]), 
                                                get_evolve_event_label(field_reg[i:j, non_field_pairs[:, 1]]))), (2, non_num)).T

    evol_event_non = evol_event_non[np.where((np.isnan(evol_event_non[:, 0]) == False)&
                                             (np.isnan(evol_event_non[:, 1]) == False)&
                                             (evol_event_non[:, 0] != 0)&
                                             (evol_event_non[:, 1] != 0))[0], :]
    evol_event_non = evol_event_non[np.random.choice(np.arange(evol_event_non.shape[0]), size=evol_event_sib.shape[0], replace=False), :]  
    return evol_event_sib, evol_event_non  

def indept_test_for_evolution_events(
    field_reg: np.ndarray, 
    field_ids: np.ndarray, 
    maze_type: int = 1,
    N: None|int = None,
    field_centers: None|np.ndarray = None,
    if_consider_distance: bool = False,
    dis_thre: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions, chi2_stat, MI, pair_type, pair_num, dimension = [], [], [], [], [], []
    
    D = GetDMatrices(maze_type, 48)
    thre = dis_thre if maze_type != 0 else 0.4
    print(field_reg.shape, field_ids.shape)
    sib_field_pairs, non_field_pairs = [], []
    for i in range(len(field_ids)-1):
        for j in range(i+1, len(field_ids)):
            if field_ids[i] == field_ids[j]:
                if len(sib_field_pairs) >= 3000000:
                    continue
                
                if if_consider_distance and D[int(field_centers[i])-1, int(field_centers[j])-1] <= thre*100:
                    continue
                
                sib_field_pairs.append([i, j])
            else:
                if len(non_field_pairs) >= 200000000:
                    continue
                
                non_field_pairs.append([i, j])
                
    sib_field_pairs = np.array(sib_field_pairs)
    non_field_pairs = np.array(non_field_pairs)
    print(" init size ", sib_field_pairs.shape, non_field_pairs.shape)
    sib_num, non_num = sib_field_pairs.shape[0], non_field_pairs.shape[0]
    
    for dt in np.arange(2, 6):
        for i in range(field_reg.shape[0]-dt+1):
            idx = np.where(np.isnan(np.sum(field_reg[i:i+dt, :], axis=0)) == False)[0]
                
            if idx.shape[0] == 0:
                continue

            real_distribution = get_evolve_event_label(field_reg[i:i+dt, idx])
            real_distribution = real_distribution[np.where(real_distribution != 0)[0]]
            
            evol_event_sib, evol_event_non = get_evolve_event_pairs(
                i=i,
                j=i+dt,
                field_reg=field_reg,
                sib_field_pairs=sib_field_pairs,
                non_field_pairs=non_field_pairs
            ) 
            print("    ", evol_event_sib.shape, evol_event_non.shape)
      
            chi_stat_sib, chi_stat_non, n_pair_stat, _ = indept_field_evolution_chi2(
                real_distribution=real_distribution,
                X_pairs=evol_event_sib,
                Y_pairs=evol_event_non
            )
            mi_sib, mi_non, _, _ = indept_field_evolution_mutual_info(
                X_pairs=evol_event_sib,
                Y_pairs=evol_event_non
            )

            sessions = sessions + [i+1, i+1]
            chi2_stat = chi2_stat + [chi_stat_sib, chi_stat_non]
            MI = MI + [mi_sib, mi_non]
            dimension = dimension + [dt, dt]
            pair_type = pair_type + ['Sibling', 'Non-sibling']
            pair_num = pair_num + [evol_event_sib.shape[0], evol_event_non.shape[0]]
        
    return np.array(sessions), np.array(chi2_stat), np.array(MI), np.array(pair_type), np.array(pair_num), np.array(dimension)


# Return Coordination Matrix
def coordination_index_for_evolution_events(
    field_reg: np.ndarray, 
    field_ids: np.ndarray, 
    maze_type: int = 1,
    N: None|int = None,
    field_centers: None|np.ndarray = None,
    if_consider_distance: bool = False,
    dis_thre: float = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions, CI, pair_type, pair_num, dimension = [], [], [], [], []
    
    D = GetDMatrices(maze_type, 48)
    thre = dis_thre if maze_type != 0 else 0.4
    print(field_reg.shape, field_ids.shape)
    sib_field_pairs, non_field_pairs = [], []
    for i in range(len(field_ids)-1):
        for j in range(i+1, len(field_ids)):
            if field_ids[i] == field_ids[j]:
                if len(sib_field_pairs) >= 3000000:
                    continue
                
                if if_consider_distance and D[int(field_centers[i])-1, int(field_centers[j])-1] <= thre*100:
                    continue
                
                sib_field_pairs.append([i, j])
            else:
                if len(non_field_pairs) >= 200000000:
                    continue
                
                non_field_pairs.append([i, j])
                
    sib_field_pairs = np.array(sib_field_pairs)
    non_field_pairs = np.array(non_field_pairs)
    print(" init size ", sib_field_pairs.shape, non_field_pairs.shape)
    sib_num, non_num = sib_field_pairs.shape[0], non_field_pairs.shape[0]
    
    for dt in np.arange(2, 6):
        for i in range(field_reg.shape[0]-dt+1):
            idx = np.where(np.isnan(np.sum(field_reg[i:i+dt, :], axis=0)) == False)[0]
                
            if idx.shape[0] == 0:
                continue

            real_distribution = get_evolve_event_label(field_reg[i:i+dt, idx])
            real_distribution = real_distribution[np.where(real_distribution != 0)[0]]
            
            evol_event_sib, evol_event_non = get_evolve_event_pairs(
                i=i,
                j=i+dt,
                field_reg=field_reg,
                sib_field_pairs=sib_field_pairs,
                non_field_pairs=non_field_pairs
            ) 
            print("    ", evol_event_sib.shape, evol_event_non.shape)
      
            CI_stat_sib, CI_stat_non, n_pair_stat, _ = indept_field_evolution_CI(
                real_distribution=real_distribution,
                X_pairs=evol_event_sib,
                Y_pairs=evol_event_non
            )

            sessions = sessions + [i+1, i+1]
            CI = CI + [CI_stat_sib, CI_stat_non]
            dimension = dimension + [dt, dt]
            pair_type = pair_type + ['Sibling', 'Non-sibling']
            pair_num = pair_num + [evol_event_sib.shape[0], evol_event_non.shape[0]]
        
    return np.array(sessions), np.array(CI), np.array(pair_type), np.array(pair_num), np.array(dimension)

def compute_joint_probability_matrix(
    field_reg: np.ndarray, 
    field_ids: np.ndarray, 
    N: None|int = None,
    dim: int = 2,
    return_item: str = "sib", 
    sib_field_pairs: None|np.ndarray = None,
    non_field_pairs: None|np.ndarray = None
):
    if return_item not in ['sib', 'non']:
        raise ValueError(f"return_item must be 'sib' or 'non', rather than {return_item}")

    sessions, mat = [], []
    if sib_field_pairs is None or non_field_pairs is None:
        sib_field_pairs, non_field_pairs = [], []
        for i in range(len(field_ids)-1):
            for j in range(i+1, len(field_ids)):
                if field_ids[i] == field_ids[j]:
                    if len(sib_field_pairs) >= 3000000:
                        continue
                    sib_field_pairs.append([i, j])
                    sib_field_pairs.append([j, i])
                else:
                    if len(non_field_pairs) >= 200000000:
                        continue
                    non_field_pairs.append([i, j])
                    non_field_pairs.append([j, i])
                
        sib_field_pairs = np.array(sib_field_pairs)
        non_field_pairs = np.array(non_field_pairs)
        
    print(" init size ", sib_field_pairs.shape, non_field_pairs.shape)
    sib_num, non_num = sib_field_pairs.shape[0], non_field_pairs.shape[0]
    
    dt = dim
    for i in range(field_reg.shape[0]-dt+1):
        idx = np.where(np.isnan(np.sum(field_reg[i:i+dt, :], axis=0)) == False)[0]
                
        if idx.shape[0] == 0:
            continue

        real_distribution = get_evolve_event_label(field_reg[i:i+dt, idx])
        real_distribution = real_distribution[np.where(real_distribution != 0)[0]]
            
        evol_event_sib, evol_event_non = get_evolve_event_pairs(
            i=i,
            j=i+dt,
            field_reg=field_reg,
            sib_field_pairs=sib_field_pairs,
            non_field_pairs=non_field_pairs
        )
        print(" after processed size ", evol_event_sib.shape, evol_event_non.shape)
        
        sib_mat, non_sib, joint_mat = indept_field_evolution_chi2(
            real_distribution=real_distribution,
            X_pairs=evol_event_sib,
            Y_pairs=evol_event_non,
            return_mat=True
        )

        sessions.append(i+1)
        if return_item == 'sib':
            mat.append(sib_mat/np.sum(sib_mat)-joint_mat/np.sum(joint_mat))
        else:
            mat.append(non_sib/np.sum(non_sib)-joint_mat/np.sum(joint_mat))
            
    return np.array(sessions), np.stack(mat)



if __name__ == '__main__':
    """
    import pickle
    import pandas as pd
    with open(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\trace_mdays.pkl", 'rb') as handle:
        trace = pickle.load(handle)    
    
    indept_test_for_evolution_events(trace['field_reg'], trace['field_info'])
    
    REG = {}
    INFO = {}
    for i, day in enumerate([ 20230806, 20230808, 20230810, 20230812, 
          20230814, 20230816, 20230818, 20230820, 
          20230822, 20230824, 20230827, 20230829,
          20230901,
          20230906, 20230908, 20230910, 20230912, 
          20230914, 20230916, 20230918, 20230920, 
          20230922, 20230924, 20230926, 20230928, 
          20230930]):
        REG[day] = trace['field_reg'][i, :]
        INFO[day] = trace['field_info'][i, :, 0]
    
    REG = pd.DataFrame(REG)
    INFO = pd.DataFrame(INFO)
    
    REG.to_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227superlong-field_reg_output.xlsx", index=False)
    INFO.to_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\10227superlong-field_info_output.xlsx", index=False)
    """
        
            
