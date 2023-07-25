from mylib.statistic_test import *

a = np.zeros((6, 2))
b = np.zeros((12, 2))

D = GetDMatrices(1, 12)
print(np.where(D < 0))
