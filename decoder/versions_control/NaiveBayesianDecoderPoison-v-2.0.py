from mylib.maze_utils3 import *

class NaiveBayesDecoderPoison(object):
    '''
    version: 2.0
    Author: YAO Shuyang
    Date: September 20th, 2022
    -------------------------------------
    
    A naive Bayesian decoder based on new principles which aims at decoding neural signals.
    This decoder is written for improving the decoding accuracy and declining the MSE (Mean Square Error) for neural recordings in specific environment, 
    like maze. We perform time-window based method while version 1.0~1.3 are a special condition of decoder-v-2.0.

    '''

    def __init__(self, maze_type = 1, res=12, l = 0.01, _version = 2.0, Loss_function = '0-1', is_smooth = True, time_window = 30):
        '''
        This decoder is compatible with open field and different maze type. Some parameters is set to modulate the decoding efficiency.
        -----------------------------
        Parameters:
        
        maze type: int, different maze type
        
        res: int, the size of spatial bin. Default = 12, which is equal to a bin size of 8 cm.

        l: int, Laplacian parameter, to smooth spike sequence.

        _version: float, set the decoder version information

        Loss_function: str, only has 3 valid value - 'd', 'd2', '0-1'

        is_smooth: bool, whether smooth or not.

        time_window: int, frames.
        '''

        self.res=res
        self.is_cease = False
        self.maze_type = maze_type
        self.l = l
        self.version = 'NaiveBayesDecoderPoison v '+str(_version)
        self.Loss_function = Loss_function
        self.is_smooth = is_smooth
        self.tao = int(time_window)
        return
    
    # broadth first search (BFS) to calculating the distance of 2 maze bin
    # A Dijkstra strategy
    # for NaiveBayesDecoder_ForMaze class to generate distance matrix D during model fitting.
    # Generate distance matrix D
    def _Generate_D_Matrix(self):
        print("    Generate D matrix")
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

        # D 0-1 function
        D01 = np.ones((self.res**2,self.res**2), dtype = np.float64)
        for i in range(self.res**2):
            D01[i-1,i-1] = 0

        if self.Loss_function == '0-1':
            self.d = D01
        elif self.Loss_function == 'd':
            self.d = D
        elif self.Loss_function == 'd2':
            self.d = D2
        else:
            return("Loss function type Error! Only '0-1','d','d2' are valid value. Report by self._Generate_D_Matrix()")
            self.is_cease = True
            return

        print("    D matrix successfully generated!")
        return self.d

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
        d = self._Generate_D_Matrix()
        self.Spikes_train = Spikes_train
        self.MazeID_train = MazeID_train
        
    # ----------------------------------- PREDICTION ------------- Prediction --------------------------------------------
    
    # Generate tuning curve matrix (pext in previous versions) to equavalently substitute probability matrix.
    def _Generate_TuningCurve(self):
        print("    Generating tuning curve")
        _nbins = self.res**2
        _coords_range = [0, _nbins +0.0001 ]
        n_neuron = self.Spikes_train.shape[0]
        Spikes_train = self.Spikes_train
        MazeID_train = self.MazeID_train
        tao = int(self.tao)
        T = Spikes_train.shape[1]
        
        count_freq = np.zeros(_nbins, dtype = np.float64)
            
        count_freq ,_ ,_= scipy.stats.binned_statistic(
            self.MazeID_train,
            self.MazeID_train,
            bins=_nbins,
            statistic="count",
            range=_coords_range)

        count_freq[count_freq < 1] = 1  # 1/25
        pext_A = count_freq / np.nansum(count_freq)

        pext = np.zeros((n_neuron,_nbins,tao), dtype = np.float64)

        for t in range(T-tao+2):
            dis = np.nansum(Spikes_train[:, t: t+tao-1], axis = 1)
            for n in range(n_neuron):
                pext[n,int(MazeID_train[t])-1,int(dis[n])] += 1.0

        for n in range(n_neuron):
            for i in range(_nbins):
                pext[n,i,:] = (pext[n,i,:] + self.l) / np.nansum(pext[n,i,:] + self.l)
        self.pext = pext
        self.pext_A = pext_A
        
        print("    Tuning curve successfully generated!")
        return pext, pext_A
    
    # Bayesian estimation with Laplacian smoothing (BELS)
    def _BayesianEstimation(self, Spikes_test, l = 1, Sj = 2):
        Spikes_train = self.Spikes_train
        T = Spikes_train.shape[1] # Total time frames of training set.
        n_neuron = Spikes_train.shape[0] # number of neurons in training set.
        nx = self.res             # maze length
        self.n = n_neuron
        self.Sj = 2
        T_test = Spikes_test.shape[1] # Total time frames of testing set.
        n_test = Spikes_test.shape[0] # number of neurons in tesing set. n_test should be equal to n.
        if n_test != n_neuron:
            print("    Warning! number of neurons in training set is not equal to that in testing set! prediction ceased.")
            self.is_cease = True
            return

        P = np.ones((nx*nx,T_test), dtype = np.float64)
        pext, pext_A = self._Generate_TuningCurve()
        log_A = np.log(pext_A)
  
        # generate P matrix.
        print("    Generating P matirx...")
        for t in tqdm(range(T_test)):
            if t <= T_test - self.tao + 1:
                dis = np.nansum(Spikes_test[:, t : t+self.tao-1], axis = 1)
            else:
                dis = np.nansum(Spikes_test[:, t::], axis = 1) / (T_test - t+1) * self.tao

            p = np.ones((n_neuron, self.res**2), np.float64)
            for n in range(n_neuron):
                p[n,:] *= pext[n,:,int(dis[n])]

            P[:,t] = np.nanprod(p, axis = 0) * pext_A            
            
        self.P = P
        return P

    def predict(self,Spikes_test, MazeID_test):
        if self.is_cease == True:
            print("    Prediction aborts! self.is_cease has been set as True for some reasons.")
            return

        #Get values saved in "fit" function
        P = self._BayesianEstimation(Spikes_test = Spikes_test, l = self.l, Sj = 2)
        D = self.d
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
        MAE = np.nanmean(abd)
        MSE = np.nanmean(abd**2)
        std_abd = np.std(abd**2)
        RMSE = np.sqrt(MSE)
        print("  ",RMSE, MAE)
        self.RMSE = RMSE
        self.MAE = MAE

        return MSE, std_abd, RMSE, MAE
    
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
            self.abHit = np.nanmean(abHit)
            self.geHit = np.nanmean(geHit)
            print("  ",self.abHit,self.geHit)
            return np.nanmean(abHit), np.nanmean(geHit)
        
        print("  ",self.abHit)
        self.abHit = np.nanmean(abHit)
        return np.nanmean(abHit)