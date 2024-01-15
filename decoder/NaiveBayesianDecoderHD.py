from mylib.calcium.smooth.gaussian import gaussian_smooth_matrix1d
from tqdm import tqdm
import pickle
import scipy.stats
import numpy as np
from numba import jit
import time

@jit(nopython=True)
def _generate_tuning_curve(
    Spikes: np.ndarray, 
    spike_nodes: np.ndarray,
    nbin: int = 120
):
    rate_map = np.zeros((Spikes.shape[0], nbin), dtype=np.float64)
    for i in range(nbin):
        indices = np.where(spike_nodes==i)[0]
        frame_num = len(indices)
        if frame_num == 0:
            continue
        rate_map[:, i] = np.sum(Spikes[:, indices], axis=1) / frame_num
        
    return rate_map

@jit(nopython=True)
def _get_post_prob(Spikes_test: np.ndarray, pext: np.ndarray, pext_A: np.ndarray, nbin: int = 120):
    P = np.ones((nbin, Spikes_test.shape[1]), dtype = np.float64)
    log_A = np.log(pext_A)
  
    # generate P matrix.
    print("    Generating P matirx...")
    for t in range(Spikes_test.shape[1]):
        spike_idx = np.where(Spikes_test[:,t]==1)[0]
        nonspike_idx = np.where(Spikes_test[:,t]==0)[0]
        p = np.log(np.concatenate((pext[spike_idx,:],(1-pext[nonspike_idx,:])), axis=0))
        P[:,t] = np.sum(p, axis = 0) + log_A
        
    return P

class NaiveBayesDecoderHD(object):
    '''
    version: 3.0
    Author: YAO Shuyang
    Date: Jan 9th, 2024
    -------------------------------------
    Version for the decoding of head-direction cells.
    
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

    Log(-v.3.0):
        1. Replace 2D smoothing kernel with 1D Gaussian Kernal.
        2. Providing function to calculate metrics.
        3. Modify the generation of D matrix
    
    '''

    def __init__(
        self,
        nbin: int=120, 
        l: float = 0.01,
        _version: float = 3.0, 
        is_smooth: bool = True,
        frame_range: list | np.ndarray = []
    ) -> None:
        '''
        This decoder is compatible when decoding 1D parameters if everything goes well. 
        Some parameters is set to modulate the decoding efficiency.
        -----------------------------
        Parameters:
        
        nbin: int, the number of spatial bin. Default = 12, which is equal to a bin size of 8 cm.

        l: int, Laplacian parameter, to smooth spike sequence.

        _version: float, set the decoder version information

        is_smooth: bool, whether smoothing or not.
        '''
        self.nbin=nbin
        self.is_cease = False
        self.l = l
        self.version = 'NaiveBayesDecoder v '+str(_version)
        self.is_smooth = is_smooth
        self.frame_range = frame_range


    def _Generate_D_Matrix(self):
        print("    Generate D matrix")
        D = np.zeros((self.nbin, self.nbin), dtype=np.int64)

        for i in range(self.nbin-1):
            for j in range(i+1, self.nbin):
                D[i, j] = D[j, i] = (j-i)
        
        idx = np.where(D > self.nbin*0.5)
        D[idx] = self.nbin - D[idx]
        self.d = D

        print("    D matrix successfully generated!")
        return self.d
    # ----------------------------------- FITTING ------------------ Fitting ----------------------------------------------
    
    def fit(self, Spikes_train: np.ndarray, PosID: np.ndarray):
        """
        fit: Fit NB decoder with spikes and BinID

        Parameters
        ----------
        Spikes_train : np.ndarray, (n_neuron, n_frames)
            Spikes data, you can input either a binary matrix or a non-negative matrix.
        PosID : np.ndarray, (n_frames)
            The Position information, which must have the same length with Spikes.
        """
        if self.is_cease == True:
            print("    Fitting aborts! self.is_cease has been set as True for some reasons.")
            return

        # basic length
        T = Spikes_train.shape[1] # Total time frames of training set.
        n = Spikes_train.shape[0] # number of neurons.
        nbin = self.nbin # maze length

        # generate distance matrix;
        d = self._Generate_D_Matrix()
        self.Spikes_train = Spikes_train
        self.PosID = PosID
        
    # ----------------------------------- PREDICTION ------------- Prediction --------------------------------------------
    
    # Generate tuning curve matrix (pext in previous versions) to equavalently substitute probability matrix.
    def _Generate_TuningCurve(self):
        print("    Generating tuning curve")
        t1 = time.time()
        _nbins = self.nbin
        _coords_range = [0, _nbins +0.0001 ]
        n_neuron = self.Spikes_train.shape[0]
        
        spike_freq_all = np.zeros([n_neuron,_nbins], dtype = np.float64)
        count_freq = np.zeros(_nbins, dtype = np.float64)
        for i in range(n_neuron):
            spike_freq_all[i,] ,_ ,_= scipy.stats.binned_statistic(
                self.PosID,
                self.Spikes_train[i,:],
                bins=_nbins,
                statistic="sum",
                range=_coords_range)
            
        count_freq ,_ ,_= scipy.stats.binned_statistic(
            self.PosID,
            self.Spikes_train[i,:],
            bins=_nbins,
            statistic="count",
            range=_coords_range)

        count_freq[count_freq < 1] = 1  # 1/25
        pext = spike_freq_all / count_freq
        t2 = time.time()
        #pext = _generate_tuning_curve(self.Spikes_train, self.PosID, self.nbin)
        peak = np.nanmax(pext, axis=1)

        if self.is_smooth == True:
            # Generate Gaussian smooth kernel
            Ms = gaussian_smooth_matrix1d(shape=self.nbin, window=50, sigma=1)
            smooth_pext = np.dot(pext, Ms.T)
            
        else:
            smooth_pext = pext

        pext_A = count_freq / np.nansum(count_freq)
        
        smooth_pext[np.where(smooth_pext < 0.000001)] = 0.000001
        smooth_pext[np.where(smooth_pext > 0.999999)] = 0.999999
        
        #smooth_peak = np.nanmax(smooth_pext, axis=1)
        #smooth_pext = (smooth_pext.T * peak/smooth_peak).T
        
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
        nbin = self.nbin             # maze length
        self.n = n
        self.Sj = 2
        T_test = Spikes_test.shape[1] # Total time frames of testing set.
        n_test = Spikes_test.shape[0] # number of neurons in tesing set. n_test should be equal to n.
        if n_test != n:
            print("    Warning! number of neurons in training set is not equal to that in testing set! prediction ceased.")
            self.is_cease = True
            return

        pext, pext_A = self._Generate_TuningCurve()
        
        P = np.ones((nbin, T_test), dtype = np.float64)
        self.pext = pext
        self.pext_A = pext_A
        log_A = np.log(pext_A)
  
        # generate P matrix.
        print("    Generating P matirx...")
        for t in tqdm(range(T_test)):
            spike_idx = np.where(Spikes_test[:,t]==1)[0]
            nonspike_idx = np.where(Spikes_test[:,t]==0)[0]
            p = np.concatenate((pext[spike_idx,:],(1-pext[nonspike_idx,:])),axis=0)
            P[:,t] = np.nanprod(p, axis = 0)*pext_A
        """   
        P = _get_post_prob(Spikes_test, pext, pext_A, self.nbin) 
        """
        self.P = P
        return P

    def predict(self,Spikes_test, PosID_test):
        if self.is_cease == True:
            raise ValueError("    Prediction aborts! self.is_cease has been set as True for some reasons.")

        #Get values saved in "fit" function
        P = self._BayesianEstimation(Spikes_test = Spikes_test, l = self.l, Sj = 2)
        D = self.d
        self.Spikes_test = Spikes_test
        self.PosID_test = PosID_test

        # output matrix = D x P
        # output = argmin(D x P, axis = 0)
        dp = np.dot(D, P)
        self.dp = dp
        PosID_predicted = np.argmin(dp, axis = 0) + 1
        self.PosID_predicted = PosID_predicted

        return PosID_predicted #Return predictions


    # Ab(solute) D(istance), predicted error, added by YAO Shuyang, August 26th, 2022
    def metrics_mae(self, y_test: np.ndarray, y_pred: np.ndarray, unit_per_bin: float = 3) -> float:
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

        #### UNIT: BIN

        Raises
        ------
        ValueError
            If y_test shanres different dimension with y_pred
        """
        if y_pred.shape[0] != y_test.shape[0]:
            raise ValueError("y_test shanbin different dimension with PosID_prediected.")

        dis = np.abs(y_pred - y_test)
        idx = np.where(dis > 0.5*self.nbin)[0]
        dis[idx] = self.nbin - dis[idx]

        print(f"MAE: {round(np.mean(dis) * unit_per_bin, 2)}")
        return np.mean(dis) * unit_per_bin
    
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
            raise ValueError("y_test shares different dimension with PosID_prediected.")
        
        accuracy = np.mean(np.where(y_pred - y_test == 0, 1, 0))

        print(f"Accuracy: {round(accuracy*100, 2)}%")
        return accuracy