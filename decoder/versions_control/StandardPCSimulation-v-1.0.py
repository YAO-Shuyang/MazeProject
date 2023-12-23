from mylib.maze_utils3 import *

class StandardPCSimulation():
    '''
    version： 1.0
    Author: YAO Shuyang
    Date: August 29th, 2022

    What is a standard place cell? We regard place cell that follows theses 3 criteria as a standard (or classical) place cell:
        a.	Has only 1 place field.
        b.	Firing rate distribution strictly obeys Gaussian distribution.
        c.	neuron populations are independently firing at each spatial bins. 
        
    '''


    def __init__(self, n = 100, sigma = 5, nx = 2304, maze_type = 1, _version = 1.0):
        '''
        n: int, numbers of neuron to generate.
        nx: int, how many spatial bins can be selected as place field center.
        sigma: float, Gaussian parameter, determines the field size of place cell.
                the value is randomly generated from range (1,sigma)
        '''
        self.n = n
        self.sigma = sigma
        self.nx = nx
        self.maze_type = maze_type
        self.is_cease = False
        self.version = 'StandardPCSimulation v '+str(_version)

    def _ReadInDMatrix(self):
        with open(r'G:\YSY\maze_learning\decoder_DMatrix.pkl', 'rb') as handle:
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
        PM = np.zeros((self.n, self.nx), dtype = np.float64)
        FieldCenter = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = self.n)
        FieldSize = (np.random.random(size = self.n)+1)*self.sigma

        # Generate D_Mtrix: ndarray with size (nx,nx) which contains distance value between any 2 bins.
        D = self._ReadInDMatrix()

        # To initiate PM.
        for k in range(self.n):
            for j in range(self.nx):
                x = D[FieldCenter[k]-1,j]
                PM[k,j] = self._Gaussian(x, sigma = FieldSize[k])
        # PM = sklearn.preprocessing.normalize(PM, norm = 'l1')
        
        self.PM = PM
        return PM

    def Generate_TrainingSet(self, T = 20000):
        '''
        T: int, numbers of frames.
        '''
        MazeID_sequence = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = T)
        Spikes_sequence = np.zeros((self.n, T), dtype = np.float64)
        print("    Generate PM...")
        PM = self._PossibilityMatrix()
        print("    Generate Spikes...")
        # To randomly generate a Spikes_sequence according to the possibility matrix PM.
        for t in range(T):
            for n in range(self.n):
                Spikes_sequence[n,t] = np.random.choice([0, 1], p = [1-PM[n,MazeID_sequence[t]-1], PM[n,MazeID_sequence[t]-1]])

        self.T = T
        self.MazeID_sequence = MazeID_sequence
        self.Spikes_sequence = Spikes_sequence
        print("Done.",end='\n\n\n')
        
        return MazeID_sequence, Spikes_sequence