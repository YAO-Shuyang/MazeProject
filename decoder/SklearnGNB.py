import numpy as np
from mylib.divide_laps.lap_split import LapSplit
from mylib.maze_graph import maze_graphs, Quarter2FatherGraph, Son2FatherGraph
from mylib.maze_utils3 import spike_nodes_transform, GetDMatrices, SpikeType, temporal_events_rate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class SklearnGNB:
    """ 
    SklearnGNB v1.2.2
    """
    
    def __init__(
        self,
        maze_type: int, 
        res: int=12, 
        _version: float = GaussianNB.__name__
    )-> None:
        """__init__ of SklearnGNB

        Parameters
        ----------
        maze_type : int
            The maze type
        res : int, optional
            The number of spatial bin, by default 12
        _version : float, optional
            The version, by default GaussianNB.__name__
        """
                
        self.res=res
        self.is_cease = False
        self.maze_type = maze_type
        self._version = 'sklearn 1.2.2 '+str(_version)
        self.model = GaussianNB()
    
    @property
    def version(self):
        return self._version
    
    def _shift_shuffle(self, X: np.ndarray) -> np.ndarray:
        """_shift_shuffle

        Parameters
        ----------
        X : np.ndarray
            The input data with shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            The shifted and shuffled data
        """
        rand_lim = X.shape[1]
        num = X.shape[0]
        
        rand_shift = np.random.randint(0, rand_lim, num)
        rand_X = np.zeros_like(X)
        print("    Shuffle:")
        for i in tqdm(range(num)):
            rand_X[i, :] = np.roll(X[i, :], shift = rand_shift[i])
            
        return rand_X
        
    
    def preprocessing(
        self, 
        trace: dict, 
        transient: str = 'RawTraces', 
        test_size: float = 0.4, 
        spike_thre: float = 3,
        data_type: str = 'Actual',
        sigma: float = 3,
        time_bins: int = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """preprocessing for NaiveBayesDecoder

        Parameters
        ----------
        trace : dict
            Data struct that saves calcium imaging data.
        transient : str, optional
            The calcium data that used to decoding, by default 'RawTraces'
        test_size : float, optional
            The test size, by default 0.4。

        Returns
        -------
        X_train : array-like, shape = [n_samples, n_features]
            The training input samples.
        X_test : array-like, shape = [n_samples, n_features]
            The test input samples.
        y_train : array-like, shape = [n_samples]
            The target values (class labels) as integers.
        y_test : array-like, shape = [n_samples]
            The test target values.

        Raises
        ------
        ValueError
            Only RawTraces and DeconvSignal are supported.
        """
        try:
            assert transient in ['RawTraces', 'DeconvSignal', 'EventsTrain']
        except:
            raise ValueError("Only RawTraces and DeconvSignal are supported.")
        
        if transient == 'EventsTrain':
            key = 'DeconvSignal'
        else:
            key = transient
        
        Deconv = []
        Spikes = []
        X = []
        t = []

        beg, end = trace['lap_begin_index'], trace['lap_end_index']
        #pc = np.where(trace['is_placecell'] == 1)[0]
        for i in range(beg.shape[0]):
            indices = np.where((trace['ms_time'] >= trace['correct_time'][beg[i]])&(trace['ms_time'] <= trace['correct_time'][end[i]]))[0]
            xn = trace[key][:, indices]
            tn = trace['ms_time'][indices]
            
            if transient == 'EventsTrain':
                xn = SpikeType(xn, threshold=spike_thre)
                xn = temporal_events_rate(xn, tn, sigma=sigma, time_bins=time_bins)
            
            X.append(xn)
            t.append(tn)
        
        X = np.concatenate(X, axis=1)
        t = np.concatenate(t)
        
        y = np.zeros(X.shape[1])
        for i in tqdm(range(X.shape[1])):
            a, b = np.where(trace['correct_time'] >= t[i])[0][0], np.where(trace['correct_time'] <= t[i])[0][-1]
        
            t1 = trace['correct_time'][a]
            t2 = trace['correct_time'][b]
    
            if t1 - t[i] < t[i] - t2:
                y[i] = trace['correct_nodes'][a]
            else:
                y[i] = trace['correct_nodes'][b]  
                
        if self.res != 48:
            y = spike_nodes_transform(y, self.res)

        
        if data_type == 'Shuffle':
            rand_n = np.random.randint(low=1, high=X.shape[1]-1, size=X.shape[0])
            print("  Deassociate the calcium activity with behavior frames (Shuffle):")
            for i in tqdm(range(X.shape[0])):
                X[i, :] = np.roll(X[i, :], rand_n[i])
        
        X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=test_size, shuffle=False)
        # T0 = int(X.shape[1] * (1-test_size))
        # X_train, X_test, y_train, y_test = X[:, :T0].T, X[:, T0:].T, y[:T0], y[T0:]
        
        return X_train, X_test, y_train, y_test
                
    def fit(self, X_train, y_train, **kwargs):
        '''
        fit the model
        
        Parameters
        ----------
        X_train : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y_train : array-like, shape = [n_samples]
            Target vector relative to X_train.
        '''
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X_test):
        """predict class labels

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Testing vector, where n_samples is the number of samples and
            n_features is the number of features.
        """
        return self.model.predict(X_test)
        
    def metrics_mae(self, y_test: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
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
    
    
if __name__ == '__main__':
    import os
    
    with open(r'E:\Data\Cross_maze\10209\20230728\session 2\trace.pkl', 'rb') as handle:
        trace = pickle.load(handle)
        
    model1 = SklearnGNB(maze_type=trace['maze_type'], res=48)
    X_train1, X_test1, y_train1, y_test1 = model1.preprocessing(trace, transient='RawTraces', test_size=0.4)
    
    model1.fit(X_train1, y_train1)
    y_pred1 = model1.predict(X_test1)
    
    model1.metrics_accuracy(y_test1, y_pred1)
    model1.metrics_mae(y_test1, y_pred1)
    
