# This code defined betti_curves class to perform clique topology analysis 
# More details see in Giusti et al 2015, doi: 10.1073/pnas.1506407112; Jiang et al 2022, doi: 10.1038/s41593-022-01212-4
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from mylib.AssertError import ValueErrorCheck
from mylib.maze_utils3 import calc_PVC, calc_pearsonr, Clear_Axes, DateTime
import os
import ripser
import networkx as nx


VERTION = 'betti_curves 1.0'
AUTHOR = "YAO Shuyang, 姚舒扬"

class betti_curves(object):
    '''
    Author: YAO Shuyang
    Date: Jan 28th, 2023
    Version: betti_curves 1.0
    Note: Perform a betti curves analysis refer to Giusti et al 2015 and Jiang et al 2022.
    '''
    def __init__(self, input_mat1:np.ndarray, input_mat2:np.ndarray = None, corr_type:str = 'betti', intervals:float = 0.01, **kwargs):
        '''
        Parameters
        ----------
        input_mat1: NDarray[float], (n_neuron, n_spatial_bins), required.
            usually you can input a matrix with shape of (n, 144) or (n, 2304)

        input_mat2: NDarray[float], (n_neuron, n_spatial_bins), optional.
            * you can input this parameter if you want. 
            * If you input this one, note that it must have the save shape as input_mat1
            * If you don't input this one, it will automatically set input_mat2 as input_mat1 to perform a self-correlation.

        corr_type: str, optional, you can choose a value from this set {'pvc', 'pearson', 'betti'}
            * 'pvc' means population vector correlation between spatial bins. It will return a matrix with shape of (n_spatial_bins, n_spatial_bins).
                Note that if you perform a 'pvc', you have to input 'input_mat2' or it will switch to 'pearson' mode automatically and raise a warning.
            * 'pearson' means a pearson correlation cross cells. It will return a matrix with shape of (n_neuron, n_neuron). 
            * 'betti' means to calculate 'betti curve correlogram'
            * Default value is 'betti'.

        intervals: float, the step length for threshold varying.

        **kwargs: the required keywords that will be input in _Betti_Correlogram()

    
        Returns
        -------
        - Initialize the data structure.
        '''
        # To check if input parameters satisfy our requirements.
        ValueErrorCheck(corr_type, ['pearson', 'pvc', 'betti'])
        if input_mat2 is None:
            input_mat2 = cp.deepcopy(input_mat1)
        else:
            assert input_mat1.shape == input_mat2.shape # The shape of input two matrix must be the same.

        # Initiate basic member.
        print("Clique Topological Analysis Begins -------------------------------------------")
        print("Step 1 - Initiate Key Variables and Members.")
        self.corr_type = corr_type
        self.mat1 = input_mat1
        self.mat2 = input_mat2
        self._version = VERTION
        self._author = AUTHOR
        self.AnalysisBegTime = DateTime()
        self.n_neuron = input_mat1.shape[0]
        self.T_frame = input_mat1.shape[1]
        self.intervals = intervals

        # Initiate results matrix.
        print("Step 2 - Calculate Correlation Matrix. [Mode '"+corr_type+"']")
        if corr_type == 'pvc':
            self._PopulationVectorMatrix
        elif corr_type == 'pearson':
            self._PearsonMatrix
        elif corr_type == 'betti':
            self._Betti_Correlation(T = 0, **kwargs) # follows the method introduced in Jiang et al 2022.
        
        #self._normalization
        print("Step 3 - Calculate Betti Curve.\n    Note: This Step May Be Time-consuming...")
        self.Perform_Betti_Curve(intervals = intervals)

    @property
    # property
    def _normalization(self):
        MIN = np.nanmin(self.ResMat)
        self.ResMat[np.arange(self.n_neuron), np.arange(self.n_neuron)] = MIN


    def _Betti_Correlation(self, 
                           tau:float = 1, # second,  the time delay
                           T:float = None, # second, the total duration of the recording considered.
                           ):
        '''
        Parameters
        ----------
        tau: float, and usually set as 1.


        Note
        ----
        The Cross-correlogram of spike trains of two neurons at time delay tau is computed as:

                ccg_ij(\tau) = \frac{1}{T}\int_{0}^{T} f_i(t)f_j(t+\tau)dt
                (see in '.\bitti-1.jpg')
        
        where f_i(t) is the spike train of i th neuron, T is the total duration of the recording
        considered. 

        Then, the correlation of firing of two neurons on a time scale of \tau_max is computed as:

                C_{ij} = \frac {1}{\tau_{max} r_i r_j} \max{({\int_{0}^{\tau_{max}}ccg_{ij}\tau d\tau}, {\int_{0}^{\tau_{max}}} ccg_{ji}\tau d\tau)}
                (see in '.\bitti-2.jpg')

        
        '''
        self.T = T
        print()

    @property
    def _PopulationVectorMatrix(self):
        '''
        Returns:
        --------
        NDarray[float], (n_spatial_bins, n_spatial_bins)

        Notes:
        ------
        Population vector correlation is not a pearson-independent method. Instead, it bases on pearson correlation.
        The different between PVC and pearson correlation focuses on the vector used to calculate the value.
        
        To calculate common pearson correlation is to calculate the correlation between two neurons' rate map. 
        But PVC is to test on the same neuron pupulation vector at two different spatial bins. 
        '''
        # Create Matrix To Store the Results.
        PVCResults = calc_PVC(self.mat1, self.mat2)
        
        self.ResMat = PVCResults
        self.ResSize = self.mat1.shape[1]
        return PVCResults

    @property
    def _PearsonMatrix(self):
        '''
        Returns:
        --------
        NDarray[float], (n_spatial_bins, n_spatial_bins)

        Notes:
        ------
        Population vector correlation is not a pearson-independent method. Instead, it bases on pearson correlation.
        The different between PVC and pearson correlation focuses on the vector used to calculate the value.
        
        To calculate common pearson correlation is to calculate the correlation between two neurons' rate map. 
        But PVC is to test on the same neuron pupulation vector at two different spatial bins. 
        '''
        # Create Matrix To Store the Results.
        PearsonResults = calc_pearsonr(self.mat1, self.mat2)
        
        self.ResMat = PearsonResults
        self.ResSize = self.mat1.shape[1]

        return PearsonResults

    def plot_Matrix(self, figsize:tuple = (8,6), xlabel:str = 'Mat2', ylabel:str = 'Mat1', title:str = '', save_loc:str = None, **kwargs):
        '''
        Parameters
        ----------
        - save_loc: str
            The directory that you want to save the figure. If you want to show it instead of saving it, don't input.
        - Others: The parameter used to adjust the figure. 

        Notes
        -----
        Used to plot the rate map of the results matrix.
        '''

        plt.figure(figsize = figsize)
        ax  = plt.axes()

        im = ax.imshow(self.ResMat, **kwargs)
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label('Pearson Correlation') if self.corr_type == 'pearson' else cbar.set_label('Population Vector Correlation')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if save_loc is not None:
            plt.savefig(os.path.join(save_loc, 'Correlation Rate Map - '+self.corr_type+' [betti_curves_results].png'), dpi = 600)
            plt.savefig(os.path.join(save_loc, 'Correlation Rate Map - '+self.corr_type+' [betti_curves_results].svg'), dpi = 600)
            plt.close()
        else:
            plt.show()

    def _calc_edge_density(self, Mat:np.ndarray):
        '''
        Calculate the edge density for the input matrix.
        '''
        G = nx.Graph(Mat)
        n = G.number_of_nodes()
        return G.number_of_edges() / (n*(n-1)/2)

    def Perform_Betti_Curve(self, intervals:float = 0.01):
        '''
        Parameter
        ---------
        Interval: float, default value is 0.01.
            represents the step length of each trial to get a edge density figure like figure 2d~i in Jiang et al 2022.
        '''
        assert intervals <= 2 # Intervals larger than 2 is meaningless.

        # Get the max and min value.
        vmin = np.nanmin(self.ResMat)
        vmax = np.nanmax(self.ResMat)

        # Create the threshold vector with each step length equal to the 'intervals'.
        trials_num = int((vmax - vmin)/intervals) + 1
        thre_vec = np.linspace(vmin, vmax, trials_num)
        self.thre_vec = thre_vec # Save the value of each threshold.
        self.thre_len = trials_num
        betti = np.zeros((4, trials_num), dtype = np.int64)
        edge_density = np.zeros(trials_num, dtype = np.float64)

        # For each threshold in thre_vec, get the varid correlation matrix like figure 2b in Jiang et al, 2022
        for j in range(len(thre_vec)):
            # Build TestMat
            TestMat = np.where(self.ResMat >= thre_vec[j], 1, 0)
            print('      ',DateTime(),' -> ', end='')
            # Build Homology # Refer to <https://zhuanlan.zhihu.com/p/559484032>
            dgm = ripser.ripser(TestMat, maxdim = 3, distance_matrix = True)
            print(DateTime(),'[',vmin,vmax,trials_num,'] - Finished '+str(j))
            diagrams = dgm['dgms']
            # Count Betti
            #            
            for i in range(len(diagrams)):
                betti[i,j] = np.sum(diagrams[i][:,1]==np.inf)

            edge_density[j] = self._calc_edge_density(TestMat)

        self.betti = betti
        self.edge_density = edge_density
        self.AnalysisEndTime = DateTime()
        print("Done Successfully. End Procedure.")
    
    def plot_betti_curve(self, figsize = (4,3), save_loc:str = None, x_ticks:list|np.ndarray = None, y_ticks:list|np.ndarray = None):
        '''
        plot betti curve.
        '''
        print("Plot Betti Curve.")
        fig = plt.figure(figsize = (4,3))
        ax = Clear_Axes(plt.axes(), close_spines = ['top', 'right'], ifxticks = True, ifyticks = True)
        ax.plot(self.edge_density, self.betti[1,:], label = 'H1')
        ax.plot(self.edge_density, self.betti[2,:], label = 'H2')
        ax.plot(self.edge_density, self.betti[3,:], label = 'H3')
        ax.legend(edgecolor = 'white', facecolor = 'white', ncol = 3, title = 'Cycle Dimension', loc = 'upper center', fontsize = 8, title_fontsize = 8)
        ax.set_ylabel('# cycles')
        ax.set_xlabel('edge density')
        ax.set_title('Betti Curve')
        ax.axis([0,1,0, np.nanmax(self.betti)*1.3])
        if x_ticks is not None:
            ax.set_xticks(np.linspace(0,1,6))
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        if save_loc is None:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(save_loc+' [betti-curve].svg', dpi = 600)
            plt.savefig(save_loc+' [betti-curve].png', dpi = 600)
            plt.close()

# Usage:
if __name__ == '__main__':
    arr1 = np.random.randn(100)  # number of neuron is 100
    arr2 = np.random.randn(100)  # number of neuron is 100

    Betti = betti_curves(arr1, arr2)  # Initiate
    Betti.Perform_Betti_Curve(intervals = 0.01) # Threshold step. Calcuate betti number matrix.