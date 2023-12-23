from mylib.maze_utils3 import *

class NaiveBayesDecoder(object):
    '''
    version: 1.0
    Author: YAO Shuyang
    Date: August 29th, 2022
    A naive Bayesian decoder based on new priciples which aims at decoding neural signals.
    This decoder is written for improving the decoding correction rate for some neural recording in specific environment, like maze.
    This decoder can be run significantly faster than KordingLab's and has a reletively equivalent outcome.
    '''

    def __init__(self, maze_type = 1, res=12, l = 0.01):
        '''
        This decoder is compatible with open field and different maze type. Some parameters is set to modulate the decoding efficiency.
        ---------
        Parameters:
        
        maze type: int, different maze type
        
        res: int, the size of spatial bin. Default = 12, which is equal to a bin size of 8 cm.
        l: int, Laplacian parameter, to smooth spike sequence.
        '''
        self.res=res
        self.is_cease = False
        self.maze_type = maze_type
        self.l = l
        return
    
    # broadth first search (BFS) to calculating the distance of 2 maze bin
    # A Dijkstra strategy
    # for NaiveBayesDecoder_ForMaze class to generate distance matrix D during model fitting.
    # Generate distance matrix D
    def _Generate_D_Matrix(self):
        print("Generate D matrix")
        maze_type = self.maze_type
        nx = self.res
        with open('decoder_DMatrix.pkl', 'rb') as handle:
            D_Matrice = pickle.load(handle)
            if nx == 12:
                D = D_Matrice[6+maze_type] / nx * 12
            elif nx == 24:
                D = D_Matrice[3+maze_type] / nx * 12
            elif nx == 48:
                D = D_Matrice[maze_type] / nx * 12
            else:
                self.is_cease = True
                print("self.res value error! Report by self._dis")
                return
            D2 = D**2
        self.D2 = D2
        self.D = D
        print("done.")
        return D2

    # ----------------------------------- FITTING ------------------ Fitting ----------------------------------------------
    
    def fit(self,Spikes_train,MazeID_train):
        if self.is_cease == True:
            print("    Fitting aborts! self.is_cease has been set as True for some reasons.")
            return

        # basic length
        T = Spikes_train.shape[1] # Total time frames of training set.
        n = Spikes_train.shape[0] # number of neurons.
        nx = self.res             # maze length

        # generate distance matrix;
        D2 = self._Generate_D_Matrix()
        self.Spikes_train = Spikes_train
        self.MazeID_train = MazeID_train
        
    # ----------------------------------- PREDICTION ------------- Prediction --------------------------------------------
    
    def _Generate_pext(self, l = 0.01, Sj = 2):
        l = self.l
        T = self.Spikes_train.shape[1]
        MazeID_train = self.MazeID_train
        n = self.Spikes_train.shape[0]
        nx = self.res
        N = np.zeros((T,nx*nx), np.float64)
        print("    Generate N matrix:")
        for i in tqdm(range(nx*nx)):
            N[np.where(MazeID_train == i+1)[0],i] = 1
        self.N = N
        
        pext = np.zeros((n,nx*nx),np.float64)
        print("    Generate p_extent array")
        Freq = np.dot(self.Spikes_train, N)  # Frequency of Spikes
        Freq_A = np.nansum(N,axis=0)
        pext = (Freq + l) / (Freq_A + l*Sj)
        pext_A = (Freq_A + l) / (np.nansum(Freq_A) + nx*nx*l)
        log_A = np.log(pext_A)
        
        return pext, pext_A
    
    # Generate tuning curve matrix (pext in previous versions) to equavalently substitute probability matrix.
    def _Generate_TuningCurve(self):
        print("Generating tuning curve")
        _nbins = self.res**2
        _coords_range = [0, _nbins +0.0001 ]
        n_neuron = self.Spikes_train.shape[0]
        maze_type = self.maze_type
        
        spike_freq_all = np.zeros([n_neuron,_nbins], dtype = np.float64)
        count_freq = np.zeros(_nbins, dtype = np.float64)
        for i in range(n_neuron):
            spike_freq_all[i,] ,_ ,_= scipy.stats.binned_statistic(
                self.MazeID_train,
                self.Spikes_train[i,:],
                bins=_nbins,
                statistic="sum",
                range=_coords_range)
            
        count_freq ,_ ,_= scipy.stats.binned_statistic(
                self.MazeID_train,
                self.Spikes_train[i,:],
                bins=_nbins,
                statistic="count",
                range=_coords_range)

        count_freq[count_freq < 1] = 0.000000001
        pext = spike_freq_all / count_freq
        
        with open('decoder_SmoothMatrix.pkl','rb') as handle:
            ms_set = pickle.load(handle)
        
        if self.res == 12:
            ms = ms_set[maze_type + 6]
        elif self.res == 24:
            ms = ms_set[maze_type + 3]
        elif self.res == 48:
            ms = ms_set[maze_type]
        
        self.smooth_matrix = ms
        pext_A = count_freq / np.nansum(count_freq)
        
        clear_pext = clear_NAN(pext)[0]   # clear_NAN value
        smooth_pext = np.dot(clear_pext, ms)
        
        x_idx, y_idx = np.where(smooth_pext == 0)
        for i in range(len(x_idx)):
            smooth_pext[x_idx[i],y_idx[i]] = 0.000000001  
            
        pext_A = pext_A / np.nansum(pext_A)
        self.pext = smooth_pext
        self.pext_A = pext_A
        return smooth_pext, pext_A
    
    # Bayesian estimation with Laplacian smoothing (BELS)
    def _BayesianEstimation(self, Spikes_test, l = 1, Sj = 2):
        Spikes_train = self.Spikes_train
        T = Spikes_train.shape[1] # Total time frames of training set.
        n = Spikes_train.shape[0] # number of neurons in training set.
        nx = self.res             # maze length
        self.n = n
        self.Sj = 2
        T_test = Spikes_test.shape[1] # Total time frames of testing set.
        n_test = Spikes_test.shape[0] # number of neurons in tesing set. n_test should be equal to n.
        if n_test != n:
            print("    Warning! number of neurons in training set is not equal to that in testing set! prediction ceased.")
            self.is_cease = True
            return

        P = np.ones((nx*nx,T_test), dtype = np.float64)
        pext, pext_A = self._Generate_TuningCurve()
        self.pext = pext
        self.pext_A = pext
        log_A = np.log(pext_A)
  
        # generate P matrix.
        for t in tqdm(range(T_test)):
            spike_idx = np.where(Spikes_test[:,t]==1)[0]
            nonspike_idx = np.where(Spikes_test[:,t]==0)[0]
            p = np.concatenate((pext[spike_idx,:],(1-pext[nonspike_idx,:])),axis=0)
            P[:,t] = np.nanprod(p, axis = 0) * pext_A
            
        self.P = P
        return P

    def predict(self,Spikes_test, MazeID_test):
        if self.is_cease == True:
            print("    Prediction aborts! self.is_cease has been set as True for some reasons.")
            return

        #Get values saved in "fit" function
        P = self._BayesianEstimation(Spikes_test = Spikes_test, l = self.l, Sj = 2)
        D = self.D2
        self.Spikes_test = Spikes_test
        self.MazeID_test = MazeID_test

        # output matrix = D x P
        # output = argmin(D x P, axis = 0)
        dp = np.dot(D,P)
        self.dp = dp
        MazeID_predicted = np.argmin(dp, axis = 0) + 1
        self.MazeID_predicted = MazeID_predicted

        return MazeID_predicted #Return predictions



    # Ab(solute) D(istance), predicted error, added by YAO Shuyang, August 26th, 2022
    def metrics_MSE(self):
        print("MSE")
        maze_type = self.maze_type
        nx = self.res
        D = self.D
        if self.MazeID_predicted.shape[0] != self.MazeID_test.shape[0]:
            print("    Warning! MazeID_test shares different dimension with MazeID_prediected.")

        abd = np.zeros_like(self.MazeID_predicted,dtype = np.float64)
        for k in range(abd.shape[0]):
            abd[k] = D[self.MazeID_predicted[k]-1,self.MazeID_test[k]-1] * 8
        
        # average of AbD
        qMSE = np.nanmean(abd)
        MSE = np.nanmean(abd**2)
        std_abd = np.std(abd**2)
        RMSE = np.sqrt(MSE)

        return MSE, std_abd, RMSE, qMSE
    
    def metrics_Accuracy(self):
        print("Accuracy")
        abHit = np.zeros(self.Spikes_test.shape[1], dtype = np.float64)
        for i in range(abHit.shape[0]):
            if self.MazeID_test[i] == self.MazeID_predicted[i]:
                abHit[i] = 1
        
        # if hits the same father node we think it is a general hit event.
        if self.res in [24,48]:
            geHit = np.zeros(self.Spikes_test.shape[1], dtype = np.float64)
            S2FGraph = Quarter2FatherGraph if self.res == 24 else Son2FatherGraph
            for i in range(geHit.shape[0]):
                if S2FGraph[int(self.MazeID_test[i])] == S2FGraph[int(self.MazeID_predicted[i])]:
                    geHit[i] = 1
            return np.nanmean(abHit), np.nanmean(geHit)
        
        return np.nanmean(abHit)