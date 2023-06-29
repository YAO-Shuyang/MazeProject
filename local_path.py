import os
import pandas as pd
import numpy as np

figpath = r'F:\YSY\FinalResults'
figdata = r'F:\YSY\FigData'

CM_path = r'G:\YSY\Cross_maze'
SM_path = r'G:\YSY\previous data\Simple_maze'
f1 = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'calcium')
f1_behav = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'behavior')
f2 = pd.read_excel(os.path.join(SM_path,'simple_maze_paradigm.xlsx'), sheet_name = 'calcium')

f_CellReg_opm1 = pd.read_excel(os.path.join(CM_path, 'cell_reg_path.xlsx'), sheet_name = 'op-m1')
f_CellReg_opm2 = pd.read_excel(os.path.join(CM_path, 'cell_reg_path.xlsx'), sheet_name = 'op-m2')
f_CellReg_m1m2 = pd.read_excel(os.path.join(CM_path, 'cell_reg_path.xlsx'), sheet_name = 'm1-m2')
f_CellReg_opop = pd.read_excel(os.path.join(CM_path, 'cell_reg_path.xlsx'), sheet_name = 'op-op')

cellReg_95_maze1 = r'G:\YSY\Cross_maze\11095\Maze1-footprint\Cell_reg\cellRegistered.mat'
cellReg_95_maze2 = r'G:\YSY\Cross_maze\11095\Maze2-footprint\Cell_reg\cellRegistered.mat'
order_95_maze1 = np.array([], dtype = np.int64)

recording_sessions_95 = pd.read_excel(os.path.join(CM_path,'recording_sessions.xlsx'), sheet_name = '11095')
recording_sessions_92 = pd.read_excel(os.path.join(CM_path,'recording_sessions.xlsx'), sheet_name = '11092')
recording_sessions_09 = pd.read_excel(os.path.join(CM_path,'recording_sessions.xlsx'), sheet_name = '12009')
recording_sessions_12 = pd.read_excel(os.path.join(CM_path,'recording_sessions.xlsx'), sheet_name = '12012')


familiar_dates = [20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
familiar_dates_str = [str(i) for i in familiar_dates]

stage2_dates = [20220813, 20220815, 20220718, 20220820, 20220822, 20220824, 20220826, 20220828, 20220830]
stage2_dates_str = [str(i) for i in stage2_dates]