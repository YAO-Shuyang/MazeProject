from mylib.statistic_test import *

with open(r'E:\Data\Cross_maze\11095\20220830\session 3\trace.pkl', 'rb') as handle:
    trace = pickle.load(handle)

print(np.ediff1d(trace['ms_time_behav']))