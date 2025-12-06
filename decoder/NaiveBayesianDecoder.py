from mylib.maze_utils3 import GetDMatrices, clear_NAN, SmoothMatrix, spike_nodes_transform
from mylib.maze_graph import Quarter2FatherGraph, Son2FatherGraph
from tqdm import tqdm
import pickle
import scipy.stats
import numpy as np
from numba import jit
import time
from mazepy.datastruc.neuact import SpikeTrain
from mazepy.datastruc.variables import VariableBin

@jit(nopython=True)
def _generate_tuning_curve(
    Spikes: np.ndarray, 
    spike_nodes: np.ndarray,
    nx: int = 48
):
    rate_map = np.zeros((Spikes.shape[0], nx**2), dtype=np.float64)
    for i in range(nx**2):
        indices = np.where(spike_nodes==i)[0]
        frame_num = len(indices)
        if frame_num == 0:
            continue
        rate_map[:, i] = np.sum(Spikes[:, indices], axis=1) / frame_num
        
    return rate_map

@jit(nopython=True)
def _get_post_prob(Spikes_test: np.ndarray, pext: np.ndarray, pext_A: np.ndarray, nx: int = 48):
    P = np.ones((nx*nx, Spikes_test.shape[1]), dtype = np.float64)
    log_A = np.log(pext_A)
  
    # generate P matrix.
    print("    Generating P matirx...")
    for t in range(Spikes_test.shape[1]):
        spike_idx = np.where(Spikes_test[:,t]==1)[0]
        nonspike_idx = np.where(Spikes_test[:,t]==0)[0]
        p = np.log(np.concatenate((pext[spike_idx,:],(1-pext[nonspike_idx,:])), axis=0))
        P[:,t] = np.sum(p, axis = 0) + log_A
        
    return P

from mylib.decoder.calc_p import compute_P
class NaiveBayesDecoder(object):
    '''
    version: 1.3
    Author: YAO Shuyang
    Date: August 18th, 2023
    -------------------------------------
    
    A naive Bayesian decoder based on new priciples which aims at decoding neural signals.
    This decoder is written for improving the decoding correction rate for some neural recording in specific environment, like maze.
    This decoder can be run significantly faster than KordingLab's and has a reletively equivalent outcome.

    Log(-v1.1):
        1. Delete self._Generate_pext() and Modify self._Generate_TuningCurve(). I compared the results of self._Generate_pext() and 
        self._Generate_TuningCurve() and find the former is really a redundancy.

    Log(-v1.2):
        1. Revert D matrix to its prototype (Correspond to 0-1 function). For convience, we compare the True Naive Bayesian method with
        self-defined(modified) naive Bayesian method and set a parameter to simply perform a switch between them.
        2. Add an optional parameter to modulate whether smooth or not.

    Log(-v1.3):
        1. More precise method to calculate the distance errors.
        2. Use jit to accelerate the decoding process.
        3. More precise decoding performance.
    
    '''

    def __init__(
        self,
        maze_type: int, 
        res: int=12, 
        l: float = 0.01,
        _version: float = 1.3, 
        Loss_function: str = '0-1', 
        smooth_matrix: np.ndarray | None = None, 
        is_smooth: bool = True,
        frame_range: list | np.ndarray = []
    ) -> None:
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
        '''

        self.res=res
        self.is_cease = False
        self.maze_type = maze_type
        self.l = l
        self.version = 'NaiveBayesDecoder v '+str(_version)
        self.Loss_function = Loss_function
        self.is_smooth = is_smooth
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
        D = GetDMatrices(maze_type=maze_type, nx=nx)
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
    
    def fit(self, Spikes_train, MazeID_train):
        if self.is_cease == True:
            print("    Fitting aborts! self.is_cease has been set as True for some reasons.")
            return

        # basic length
        T = Spikes_train.shape[1] # Total time frames of training set.
        n = Spikes_train.shape[0] # number of neurons.
        nx = self.res # maze length

        # generate distance matrix;
        d = self._Generate_D_Matrix()
        self.Spikes_train = Spikes_train
        self.MazeID_train = MazeID_train
        
    # ----------------------------------- PREDICTION ------------- Prediction --------------------------------------------
    
    # Generate tuning curve matrix (pext in previous versions) to equavalently substitute probability matrix.
    def _Generate_TuningCurve(self):
        print("    Generating tuning curve")
        t1 = time.time()
        _nbins = self.res**2
        _coords_range = [0, _nbins +0.0001 ]
        n_neuron = self.Spikes_train.shape[0]
        maze_type = self.maze_type
        
        t1 = time.time()
        
        spike_train = SpikeTrain(
            activity=self.Spikes_train,
            time=np.arange(self.Spikes_train.shape[1])*1000,
            variable=VariableBin(self.MazeID_train.astype(np.int64))-1
        )
        pext = spike_train.calc_tuning_curve(nbins=self.res**2, is_remove_nan=True)
        print(f"    Tuning curve generation time: {time.time() - t1} s")
        """
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

        count_freq[count_freq < 1] = 1  # 1/25
        pext = spike_freq_all / count_freq
        t2 = time.time()
        """
        #pext = _generate_tuning_curve(self.Spikes_train, self.MazeID_train, self.res)
        peak = np.nanmax(pext, axis=1)

        if self.is_smooth == True:
            Ms = SmoothMatrix(self.maze_type, sigma=1, _range = 20, nx=self.res)
            """
            with open(r'E:\Anaconda\envs\maze\Lib\site-packages\mylib\decoder\decoder_SmoothMatrix.pkl','rb') as handle:
                ms_set = pickle.load(handle)
        
            if self.res == 12:
                ms = ms_set[maze_type + 6]
            elif self.res == 24:
                ms = ms_set[maze_type + 3]
            elif self.res == 48:
                ms = ms_set[maze_type]
        
            self.smooth_matrix = ms
            smooth_pext = np.dot(clear_pext, ms)
            """
            smooth_pext = np.dot(pext, Ms.T)
            
        else:
            smooth_pext = pext

        count_freq = spike_train.calc_occu_time(nbins=self.res**2)
        pext_A = count_freq / np.nansum(count_freq)

        smooth_pext[np.where(smooth_pext < 0.000001)] = 0.000001
        smooth_pext[np.where(smooth_pext > 0.999999)] = 0.999999
        
        smooth_peak = np.nanmax(smooth_pext, axis=1)
        smooth_pext = (smooth_pext.T * peak/smooth_peak).T
        
        pext_A = pext_A / np.nansum(pext_A)
        self.pext = smooth_pext
        self.pext_A = pext_A
        print("    Tuning curve successfully generated!")
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

        
        pext, pext_A = self._Generate_TuningCurve()
        
        P = np.ones((nx*nx,T_test), dtype = np.float64)
        self.pext = pext
        self.pext_A = pext
        
        pext[np.isnan(pext)] = 1e-10
        pext_A[np.isnan(pext_A)] = 1e-10
        pext[pext < 1e-10] = 1e-10
        pext_A[pext_A < 1e-10] = 1e-10
  
        # generate P matrix.
        print("    Generating P matirx Cython...")
        t1 = time.time()
        """ 
        P = np.exp(compute_P(Spikes_test.astype(np.int64), pext.astype(np.float64), pext_A.astype(np.float64)))
        print(f"    P matrix generation time: {time.time() - t1} s")
        """
        for t in tqdm(range(T_test)):
            spike_idx = np.where(Spikes_test[:,t]==1)[0]
            nonspike_idx = np.where(Spikes_test[:,t]==0)[0]
            p = np.concatenate((pext[spike_idx,:],(1-pext[nonspike_idx,:])),axis=0)
            P[:,t] = np.nanprod(p, axis = 0) * pext_A
        """ 
        P = _get_post_prob(Spikes_test, pext, pext_A, self.res) 
        """ 
        self.P = P
        return P

    def predict(self,Spikes_test, MazeID_test):
        if self.is_cease == True:
            raise ValueError("    Prediction aborts! self.is_cease has been set as True for some reasons.")

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