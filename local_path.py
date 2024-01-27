import os
import pandas as pd
import numpy as np

figpath = r'E:\Data\FinalResults'
figdata = r'E:\Data\FigData'

DMatrixPath = r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\dismat"# r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\decoder_DMatrix.pkl"
CellregDate = r"E:\Anaconda\envs\maze\Lib\site-packages\mylib\cellreg"

CM_path = r'E:\Data\Cross_maze'
DSP_path = r'E:\Data\Dsp_maze'
RM_path = r'E:\Data\Reverse_maze'
HP_path = r'E:\Data\Hairpin_maze'

f1 = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'calcium')
f1_behav = pd.read_excel(os.path.join(CM_path,'cross_maze_paradigm.xlsx'), sheet_name = 'behavior')
f_pure_behav = pd.read_excel(os.path.join(CM_path,'behavior_only.xlsx'), sheet_name = 'behavior')
f_pure_old = pd.read_excel(os.path.join(CM_path,'behavior_only.xlsx'), sheet_name = 'prev')

f2 = pd.read_excel(os.path.join(DSP_path,'dsp_maze_paradigm.xlsx'), sheet_name = 'calcium')
f2_behav = pd.read_excel(os.path.join(DSP_path,'dsp_maze_paradigm.xlsx'), sheet_name = 'behavior')

f3 = pd.read_excel(os.path.join(RM_path,'Reverse_maze_paradigm.xlsx'), sheet_name = 'calcium')
f3_behav = pd.read_excel(os.path.join(RM_path,'Reverse_maze_paradigm.xlsx'), sheet_name = 'behavior')

f4 = pd.read_excel(os.path.join(HP_path,'Hairpin_maze_paradigm.xlsx'), sheet_name = 'calcium')
f4_behav = pd.read_excel(os.path.join(HP_path,'Hairpin_maze_paradigm.xlsx'), sheet_name = 'behavior')

f_CellReg_env = pd.read_excel(r"E:\Data\cellregs.xlsx", sheet_name='env')
f_CellReg_day = pd.read_excel(r"E:\Data\cellregs.xlsx", sheet_name='day')
f_CellReg_reverse = pd.read_excel(r"E:\Data\cellregs.xlsx", sheet_name='reverse')
f_CellReg_day_hairpin = pd.read_excel(r"E:\Data\cellregs.xlsx", sheet_name='day_hairpin')

f_decode = pd.read_excel(r"E:\Data\Simulation_pc\cross_maze_decode.xlsx", sheet_name='Lap-wise')
f_decode_shuffle = pd.read_excel(r"E:\Data\Simulation_pc\cross_maze_decode.xlsx", sheet_name='Shuffle')

