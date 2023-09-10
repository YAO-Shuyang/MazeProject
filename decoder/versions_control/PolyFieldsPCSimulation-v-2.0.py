from mylib.maze_utils3 import *

class PolyFieldsPCSimulator():
    '''
    version: 2.0
    Author: YAO Shuyang
    Date: August 30th, 2022
    ----------------------------------

    What is a standard place cell? I regard place cell that follows these 3 criteria as a standard (or classical) place cell:
        a.	Has only 1 place field.
        b.	Firing rate distribution strictly obeys Gaussian distribution.
        c.	neuron populations are independently firing at each spatial bins. 
    However, the cells we recorded is not always standard. It may has more than 1 field, or the fields is not a standard circle.
    To futher simulate this kind of place cells, I define PolyFieldsPCSimulation.
    
    '''

    def __init__(self, n = 100, sigma_low = 3, sigma_high = 5, peak_high = 0.8, peak_low = 0.6, 
                    nx = 2304, maze_type = 1, _version = 2.0, possibility = []):
        '''
        n: int, numbers of neuron to generate.

        nx: int, how many spatial bins can be selected as place field center.

        sigma(_low, _high): float, Gaussian parameter, determines the field size of place cell.
                the value is randomly generated from range [sigma_low,sigma_high)

        peak(_high,low): float, ranging from [0,1), the firing possibility of main field center locates in this area.

        possibility: field number possiblity. for randomly select cells
        '''
        self.n = n
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.nx = nx
        self.maze_type = maze_type
        self.is_cease = False
        self.peak_high = peak_high
        self.peak_low = peak_low
        self.p = possibility
        self.version = 'PolyFieldsPCSimulation v '+str(_version)

    def _ReadInDMatrix(self):
        with open('decoder_DMatrix.pkl', 'rb') as handle:
            D_Matrice = pickle.load(handle)
        if self.nx == 12**2:
            D = D_Matrice[6 + self.maze_type] #/ self.nx * 12
        elif self.nx == 24**2:
            D = D_Matrice[3 + self.maze_type] #/ self.nx * 12
        elif self.nx == 48**2:
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
        MAX_FIELDNUMBER = len(self.p)
        FieldCenter = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = (self.n,MAX_FIELDNUMBER))
        FieldSize = np.random.random(size = (self.n, MAX_FIELDNUMBER)) * (self.sigma_high-self.sigma_low) + self.sigma_low
        FieldNumber = np.random.choice(a = np.array(range(1,MAX_FIELDNUMBER+1)), size = self.n, p = self.p, replace = True)
        
        # Generate field center
        for i in range(self.n):
            for j in range(FieldNumber[i],MAX_FIELDNUMBER):
                FieldCenter[i,j] =  0
                FieldSize[i,j] = 0

        # Generate field center rate:
        CenterRate = np.zeros_like(FieldCenter, dtype = np.float64)
        CenterRate[:,0] = np.random.random(size = self.n) * (self.peak_high-self.peak_low) + self.peak_low
        for i in range(self.n):
            CenterRate[i,1:FieldNumber[i]] = np.random.random(size = FieldNumber[i]-1) * 0.5 * CenterRate[i,0] + 0.5 * CenterRate[i,0]     

        # Generate D_Mtrix: ndarray with size (nx,nx) which contains distance value between any 2 bins.
        D = self._ReadInDMatrix()

        # To initiate PM.
        
        for k in tqdm(range(self.n)):
            for j in range(self.nx):
                for n in range(FieldNumber[k]):
                    x = D[FieldCenter[k,n]-1,j]
                    PM[k,j] += self._Gaussian(x, sigma = FieldSize[k,n], peak_rate = CenterRate[k,n])
        # PM = sklearn.preprocessing.normalize(PM, norm = 'l1')
        # to make sure the element in PM will never be bigger than 1.
        for k in range(self.n):
            PM[k,:] = PM[k,:] / np.max(PM[k,:]) * CenterRate[k,0]

        self.FieldCenter = FieldCenter
        self.FieldSize = FieldSize
        self.PeakRate = CenterRate
        self.FieldNumber = FieldNumber
        self.MAX_FIELDNUMBER = MAX_FIELDNUMBER
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

