from mylib.maze_utils3 import clear_NAN, SmoothMatrix
import numpy as np
import pickle
import tqdm
import scipy

class PolyFieldsPCSimulator():
    '''
    version: 2.1
    Author: YAO Shuyang
    Date: August 31st, 2022 to September 7th, 2022
    ----------------------------------

    What is a standard place cell? I regard place cell that follows these 3 criteria as a standard (or classical) place cell:
        a.	Has only 1 place field.
        b.	Firing rate distribution strictly obeys Gaussian distribution.
        c.	neuron populations are independently firing at each spatial bins. 
    However, the cells we recorded is not always standard. It may has more than 1 field, or the fields is not a standard circle.
    To futher simulate this kind of place cells, I define PolyFieldsPCSimulation.
    
    Log(-v2.1):
        1. We think the firing rate of a certain cell in a particular site does not simply obey a independent and identical distribution
        (IN Chinese, '独立同分布'). So in version 2.1, we try to introduce a factor to depict the the relationship between velocity and 
        firing rate. We assum that there exist a funciton f between the distribution v and the distribution P(Spike = 1|pos). We do not 
        know the exactly details about this funciton, however, we only faintly know that firing rate will more posibly raise when speeding
        up. So we add two functions: self._linear_velocity(v) and self._piecewise_velocity(v) to give a estimated value for the 
        relationship between firing rate and velocity.

        2. Provide a new funciton to generate rate maps from the simulated spike sequence.
    '''

    def __init__(self, n = 100, sigma_low = 3, sigma_high = 5, peak_high = 0.8, peak_low = 0.6, 
                    nx = 2304, maze_type = 1, _version = 2.1, possibility = []):
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

    def _linear_velocity(self,v, max_velocity = 62.5, min_velocity = 2.5, min_possible = 0.1):
        if v >= max_velocity:
            return 1
        elif v <= min_velocity:
            return min_possible
        else:
            return (1 - min_possible) / (max_velocity - min_velocity) * (v - min_velocity) + min_possible

    def _piecewise_velocity(self,v, max_velocity = 62.5, min_velocity = 2.5, min_possible = 0.1):
        if v <= 2.5:
            return 0.1
        elif v >2.5 and v<=10:
            return 0.5
        elif v > 10 and v <= 30:
            return 0.8
        else:
            return 1

    def _velocity_generator(self):
        # velocity distribution is according to a recording data and can vary from mouse to mouse and from session to session obviously
        # So we only take a session for example to generate a random velocity sequence.
        with open('speed_distribution.pkl','rb') as handle:
            speed_dis = pickle.load(handle)
        
        v_level = np.zeros(self.T, dtype = np.float64)
        for t in range(self.T):
            v_level[t] = np.random.choice(np.array(range(0,25)), p = speed_dis)
        v_vec = np.random.random(size = self.T) * 2.5 + v_level*2.5

        self.v_level = v_level
        self.speed_dis = speed_dis
        self.V = v_vec
        return v_vec


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
        self.T = T
        self._velocity_generator()
        V = self.V
        MazeID_sequence = np.random.randint(low = 1, high = self.nx+1, dtype = np.int64, size = T)
        Spikes_sequence = np.zeros((self.n, T), dtype = np.float64)
        print("     Generate possibility matrix...")
        PM = self._PossibilityMatrix()

        print("     Generate simulated spike sequence...")
        # To randomly generate a Spikes_sequence according to the possibility matrix PM.
        for t in tqdm(range(T)):
            for n in range(self.n):
                # pv = self._linear_velocity(V[t])
                # pv = self._piecewise_velocity(V[t])
                pv = 1
                Spikes_sequence[n,t] = np.random.choice([0, 1], p = [1-PM[n,MazeID_sequence[t]-1]*pv, PM[n,MazeID_sequence[t]-1]*pv])

        
        self.MazeID_sequence = MazeID_sequence
        self.Spikes_sequence = Spikes_sequence
        print("  Simulation finish.")
        
        return MazeID_sequence, Spikes_sequence

    def Simulate_RateMap(self):
        _nbins = self.nx
        _coords_range = [0, _nbins +0.0001 ]
        n_neuron = self.Spikes_sequence.shape[0]
        
        spike_freq_all = np.zeros([n_neuron,_nbins], dtype = np.float64)
        count_freq = np.zeros(_nbins, dtype = np.float64)
        for i in range(n_neuron):
            spike_freq_all[i,] ,_ ,_= scipy.stats.binned_statistic(
                self.MazeID_sequence,
                self.Spikes_sequence[i,:],
                bins=_nbins,
                statistic="sum",
                range=_coords_range)
            
        count_freq ,_ ,_= scipy.stats.binned_statistic(
                self.MazeID_sequence,
                self.Spikes_sequence[i,:],
                bins=_nbins,
                statistic="count",
                range=_coords_range)

        rate_map_simulated = spike_freq_all / count_freq * 30
        clear_map_simulated = clear_NAN(rate_map_simulated)[0]
        ms = SmoothMatrix(maze_type = self.maze_type, sigma = 2, _range = 7, nx = int(np.sqrt(self.nx))).T
        smooth_map_simulated = np.dot(clear_map_simulated,ms)

        self.rate_map_simulated = rate_map_simulated
        self.clear_map_simulated = clear_map_simulated
        self.smooth_map_simulated = smooth_map_simulated
        print("    Rate map has been generated from simulated spike sequence.")


# Usage
if __name__ == '__main__':
    # from mylib.PolyFieldsPCSimulation import PolyFieldsPCSimulator

    # Read the file:
    '''
    with open('...', 'rb) as handle:
        PCPopulation = pickle.load(handle)
    '''
    # Or generate a new group of cells and their Spikes.
    PCPopulation = PolyFieldsPCSimulator(n = 400) # Number of cells. There's no problem to keep other parameters as default.
    PCPopulation.Generate_TrainingSet(T = 20000) # Number of frames

    # Use for decoding.
    Spikes = PCPopulation.Spikes_sequence
    spike_nodes = PCPopulation.MazeID_sequence
    # -> further processing

    # Save the simulator object
    with open('Simulator-default.pkl', 'wb') as f: # The directory can be altered.
        pickle.dump(PCPopulation, f)

    