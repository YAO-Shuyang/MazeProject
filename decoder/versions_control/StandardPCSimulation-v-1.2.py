from mylib.maze_utils3 import *

class StandardPCSimulation():
    '''
    version: 1.2
    Author: YAO Shuyang
    Date: August 30th, 2022

    What is a standard place cell? We regard place cell that follows theses 3 criteria as a standard (or classical) place cell:
        a.	Has only 1 place field.
        b.	Firing rate distribution strictly obeys Gaussian distribution.
        c.	neuron populations are independently firing at each spatial bins. 

    Log(-v1.1): 
        The previous version of StandardPCSimulation does not concern that different neuron have different center peak rate.
        I fix this point by adding a random parameter peak_rate which determines the center rate of each cells, ranging from 0.5 to 1.
    
    Log(-v1.2):
        1. The previous version set the sigma value as a constant value. To test the influence to decoding rate when modulating the 
        sigmavalue, we made some improvement to take the sigma value undercontrol.
        2. The previous version set the peak rate value as a constant value. To test the influence to decoding rate when modulating 
        the peak rate value, we made some improvement to take the peak rate value undercontrol.
    '''


    def __init__(self, n = 100, sigma_low = 3, sigma_high = 5, peak_high = 0.8, peak_low = 0.6, nx = 2304, maze_type = 1, _version = 1.2):
        '''
        n: int, numbers of neuron to generate.

        nx: int, how many spatial bins can be selected as place field center.

        sigma(_low, _high): float, Gaussian parameter, determines the field size of place cell.
                the value is randomly generated from range [sigma_low,sigma_high)
        '''
        self.n = n
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.nx = nx
        self.maze_type = maze_type
        self.is_cease = False
        self.peak_high = peak_high
        self.peak_low = peak_low
        self.version = 'StandardPCSimulation v '+str(_version)

    def _ReadInDMatrix(self):
        with open('decoder_DMatrix.pkl', 'rb') as handle:
            D_Matrice = pickle.load(handle)
        if self.nx == 12**2:
            D = D_Matrice[6 + self.maze_type] #/ self.nx * 12
        elif self.nx == 24**2:
            D = D_Matrice[3 + self.maze_type] #/ self.nx * 12
        elif self.nx == 2304:
            D = D_Matrice[self.maze_type] #/ self.nx * 12
        else:
            self.is_cease = True
            print("self.res value error! Report by self._dis")
            return
        self.D = D
        return D

    def _Gaussian(self, x, sigma = 3, peak_rate = 0.8):
        return peak_rate * np.exp(- x * x / (sigma * sigma * 2))

    def _PossibilityMatrix(self):
        # PM: ndarray with shape of (n,nx). Possiblity matrix for neuron populations
        # FieldCenter: ndarray with shape of (n,). Randomly generate field center vector consists of values selected in range(1,nx+1), 
        #       and the value represents the field center.
        # FieldSize: ndarray with shape of (n,). We determines field size by the value of sigma. The bigger sigma is, the larger field
        #       size is.
        # PeakRate: ndarray with shape of (n,). New added parameter in version 1.1 and later versions. Random value ranges from 0.5 
        #       to 1.
        PM = np.zeros((self.n, self.nx), dtype = np.float64)
        FieldCenter = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = self.n)
        FieldSize = np.random.random(size = self.n) * (self.sigma_high-self.sigma_low) + self.sigma_low
        PeakRate = np.random.random(size = self.n) * (self.peak_high-self.peak_low) + self.peak_low

        # Generate D_Mtrix: ndarray with size (nx,nx) which contains distance value between any 2 bins.
        D = self._ReadInDMatrix()

        # To initiate PM.
        for k in tqdm(range(self.n)):
            for j in range(self.nx):
                x = D[FieldCenter[k]-1,j]
                PM[k,j] = self._Gaussian(x, sigma = FieldSize[k], peak_rate = PeakRate[k])
        # PM = sklearn.preprocessing.normalize(PM, norm = 'l1')
        
        self.FieldCenter = FieldCenter
        self.FieldSize = FieldSize
        self.PeakRate = PeakRate
        self.PM = PM
        return PM

    def Generate_TrainingSet(self, T = 20000):
        '''
        T: int, numbers of frames.
        '''
        MazeID_sequence = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = T)
        Spikes_sequence = np.zeros((self.n, T), dtype = np.float64)
        print("     Generate possibility matrix...")
        PM = self._PossibilityMatrix()
        print("     Generate simulated spike sequence...")
        # To randomly generate a Spikes_sequence according to the possibility matrix PM.
        for t in tqdm(range(T)):
            for n in range(self.n):
                Spikes_sequence[n,t] = np.random.choice([0, 1], p = [1-PM[n,MazeID_sequence[t]-1], PM[n,MazeID_sequence[t]-1]])

        self.T = T
        self.MazeID_sequence = MazeID_sequence
        self.Spikes_sequence = Spikes_sequence
        print("  Simulation finish.")
        
        return MazeID_sequence, Spikes_sequence

'''
PCPopulation = PCSimulation(n = 1000, maze_type = 0)
spike_nodes, Spikes = PCPopulation.Generate_TrainingSet(T = 20000)

with open('PCSimulation.pkl','wb') as f:
    pickle.dump(PCPopulation,f)

def Gaussian(x, sigma = 3, peak_rate = 0.8):
    x = x
    return peak_rate * np.exp(- x**2 / (sigma**2 * 2))

import matplotlib.pyplot as plt
x = np.linspace(0,68,68001)
y = Gaussian(x)
im = plt.imshow(np.reshape(PCPopulation.PM[0],[48,48]))
plt.colorbar(im)
plt.show()
'''

    






