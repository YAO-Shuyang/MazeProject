from mylib.maze_utils3 import GetDMatrices, spike_nodes_transform
import numpy as np
import scipy.stats
import pickle
from tqdm import tqdm

class NaiveBayesDecoderPoison(object):
    '''
    version: 2.3
    Author: YAO Shuyang
    Date: September 20th, 2022
    -------------------------------------
    
    A naive Bayesian decoder based on new principles which aims at decoding neural signals.
    This decoder is written for improving the decoding accuracy and declining the MSE (Mean Square Error) for neural recordings in specific environment, 
    like maze. We perform time-window based method while version 1.0~1.3 are a special condition of decoder-v-2.0.

    Log(v2.1):
        1. Perform a Gaussian smooth for pext across space.

    Log(v2.2):
        1. Add the same _spikes_number function
        2. Change from multi P -> add logP
        3. Smooth pext_A
        4. Fixes a bug in _Generate_TuningCurve(self)
        
    Log(v2.3):
        1. Modify the calculation of distance.

    '''

    def __init__(self, maze_type = 1, res=12, l = 0.01, _version = 2.2, Loss_function = '0-1', is_smooth = True, time_window = 30, frame_range = []):
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
        self.frame_range = frame_range
        return
    
    # broadth first search (BFS) to calculating the distance of 2 maze bin
    # A Dijkstra strategy
    # for NaiveBayesDecoder_ForMaze class to generate distance matrix D during model fitting.
    # Generate distance matrix D
    def _Generate_D_Matrix(self):
        print("    Generate D matrix")
        maze_type = self.maze_type
        nx = self.res
        
        D = GetDMatrices
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
            print("Loss function type Error! Only '0-1','d','d2' are valid value. Report by self._Generate_D_Matrix()")
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
        maze_type= self.maze_type
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

        if self.is_smooth == True:
            with open('decoder_SmoothMatrix.pkl','rb') as handle:
                ms_set = pickle.load(handle)
        
            if self.res == 12:
                ms = ms_set[maze_type + 6]
            elif self.res == 24:
                ms = ms_set[maze_type + 3]
            elif self.res == 48:
                ms = ms_set[maze_type]
        
            self.smooth_matrix = ms
            smooth_pext = np.zeros_like(pext, dtype = np.float64)
            for i in range(self.tao):
                smooth_pext[:,:,i] = np.dot(pext[:,:,i], ms)

            smooth_A = np.dot(pext_A,ms)
        else:
            smooth_pext = pext
            smooth_A = pext_A

        for t in range(T-tao):
            dis = np.nansum(Spikes_train[:, t: t+tao-1], axis = 1)
            for n in range(n_neuron):
                smooth_pext[n,int(MazeID_train[t])-1,int(dis[n])] += 1.0

        for n in range(n_neuron):
            for i in range(_nbins):
                smooth_pext[n,i,:] = (smooth_pext[n,i,:] + self.l) / np.nansum(smooth_pext[n,i,:] + self.l)
        self.pext = smooth_pext
        self.pext_A = smooth_A
        
        print("    Tuning curve successfully generated!")
        return smooth_pext, smooth_A
    
    def _spikes_number(self, t):
        if t <= self.T_test - self.tao + 1:
            return np.nansum(self.Spikes_test[:, t : t+self.tao-1], axis = 1)
        else:
            return np.int64(np.nansum(self.Spikes_test[:, t::]/(self.T_test - t+1) * self.tao, axis = 1))

    # Bayesian estimation with Laplacian smoothing (BELS)
    def _BayesianEstimation(self, Spikes_test):
        n_neuron = Spikes_test.shape[0] # number of neurons in training set.
        nx = self.res             # maze length
        self.n = n_neuron
        T_test = Spikes_test.shape[1] # Total time frames of testing set.

        self.T_test = T_test
        self.Spikes_test = Spikes_test
        P = np.zeros((T_test,nx*nx), dtype = np.float64)
        pext, pext_A = self._Generate_TuningCurve()
        log_A = np.log(pext_A)
        log_pext = np.log(pext)
  
        # generate P matrix.
        print("    Generating P matirx...")
        spikes_num = np.zeros((self.n,self.T_test), dtype = np.int64)
        for t in range(self.T_test):
            spikes_num[:,t] = self._spikes_number(t)
            P[t,:] += log_A

        for n in tqdm(range(n_neuron)):
            P += log_pext[n,:,spikes_num[n,:]]

        '''
        self.spikes_num = spikes_num

        for t in tqdm(range(T_test)):
            for n in range(n_neuron):
                P[:,t] += log_pext[n,:,spikes_num[n,t]]

            P[:,t] += log_A
        '''
        self.P = P.T
        return self.P

    def predict(self,Spikes_test, MazeID_test):
        if self.is_cease == True:
            print("    Prediction aborts! self.is_cease has been set as True for some reasons.")
            return

        #Get values saved in "fit" function
        P = self._BayesianEstimation(Spikes_test = Spikes_test)
        D = self.d
        self.Spikes_test = Spikes_test
        self.MazeID_test = MazeID_test

        # output matrix = D x P
        # output = argmin(D x P, axis = 0)
        dp = np.dot(D,np.exp(P))
        self.dp = dp
        MazeID_predicted = np.argmin(dp, axis = 0) + 1
        self.MazeID_predicted = MazeID_predicted

        return MazeID_predicted #Return predictions

    # Ab(solute) D(istance), predicted error, added by YAO Shuyang, August 26th, 2022
    def metrics_mae(self, y_test: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """metrics_mae

        Parameters
        ----------
        y_test : np.ndarray
            The testing data
        y_pred : np.ndarray
            The predicted data

        Returns
        -------
        tuple[float, float, float, float]
            MSE, std_abd, RMSE, MAE

        Raises
        ------
        ValueError
            If y_test shares different dimension with y_pred
        """
        if y_pred.shape[0] != y_test.shape[0]:
            raise ValueError("y_test shares different dimension with MazeID_prediected.")

        
        D = GetDMatrices(maze_type=self.maze_type, nx=self.res)
        print(f"  Maze Type: {self.maze_type}, nx {self.res}, max value in DistanceMatrix is {np.nanmax(D)}")
        abd = np.zeros_like(y_pred,dtype = np.float64)
        for k in range(abd.shape[0]):
            abd[k] = D[int(y_pred[k])-1,int(y_test[k])-1]
        
        # average of AbD
        MAE = np.nanmean(abd)
        std_mae = np.std(abd)
        MSE = np.nanmean(abd**2)
        std_abd = np.std(abd**2)
        RMSE = np.sqrt(MSE)
        print("  RMSE: ",RMSE, "MAE: ", MAE)
        RMSE = RMSE
        MAE = MAE

        return MSE, std_abd, RMSE, MAE, std_mae, abd
    
    def metrics_accuracy(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """metrics_accuracy

        Parameters
        ----------
        y_test : np.ndarray
            The testing data
        y_pred : np.ndarray
            The predicted data

        Returns
        -------
        float
            Accuracy

        Raises
        ------
        ValueError
            If y_test shares different dimension with y_pred
        """
        if y_pred.shape[0] != y_test.shape[0]:
            raise ValueError("y_test shares different dimension with MazeID_prediected.")
        
        abHit = np.mean(np.where(y_pred - y_test == 0, 1, 0))
        geHit = np.nan
        
        if self.res in [48]:
            y_pred_old = spike_nodes_transform(y_pred, 12)
            y_test_old = spike_nodes_transform(y_test, 12)
            
            geHit = np.mean(np.where(y_pred_old - y_test_old == 0, 1, 0))

        print(f"  Absolute Accuracy: {round(abHit*100, 2)}% General Accuracy: {round(geHit*100, 2)}%")
        return abHit, geHit