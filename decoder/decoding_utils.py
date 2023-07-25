from mylib.maze_utils3 import *

def preprocessing_Kording(trace, Spikes, ms_time_behav):
    from Neural_Decoding.preprocessing_funcs import bin_spikes
    from Neural_Decoding.preprocessing_funcs import bin_output

    n_neuron = trace['n_neuron']
    behav_time = trace['behav_time']
    processed_pos_new = trace['processed_pos_new']
    
    spike_times = []
    for i in range(n_neuron):
        spike_time = ms_time_behav[np.where(Spikes[i,:] == 1)]
        spike_times.append(spike_time)
    spike_times = np.array(spike_times, dtype = object)

    t_start = min(ms_time_behav[0], behav_time[0])
    t_end = max(ms_time_behav[-1], behav_time[-1])
    downsample_factor = 10
    dt = 100 # ms

    neural_data=bin_spikes(spike_times,dt,t_start,t_end)
    #Bin output (velocity) data using "bin_output" function
    if 'correct_pos' in trace.keys():
        pos_binned = bin_output(trace['correct_pos'],behav_time,dt,t_start,t_end,downsample_factor)
    else:
        pos_binned=bin_output(processed_pos_new,behav_time,dt,t_start,t_end,downsample_factor)
    # pos_binned=bin_output(np.expand_dims(behav_nodes_interpolated,1),behav_time_original,dt,t_start,t_end,downsample_factor) 

    bins_before=7 #How many bins of neural data prior to the output are used for decoding
    bins_current=1 #Whether to use concurrent time bin of neural data
    bins_after=4 #How many bins of neural data after the output are used for decoding

    #Remove neurons with too few spikes in HC dataset
    nd_sum=np.nansum(neural_data,axis=0) #Total number of spikes of each neuron
    rmv_nrn=np.where(nd_sum<20) #Find neurons who have less than 100 spikes total
    neural_data=np.delete(neural_data,rmv_nrn,1) #Remove those neurons
    X=neural_data

    #Set decoding output
    y=pos_binned

    #Number of bins to sum spikes over
    N=bins_before+bins_current+bins_after 

    #Remove time bins with no output (y value)
    rmv_time=np.where(np.isnan(y[:,0]) | np.isnan(y[:,1]))
    X=np.delete(X,rmv_time,0)
    y=np.delete(y,rmv_time,0)

    #Set what part of data should be part of the training/testing/validation sets

    training_range=[0, 0.8]
    valid_range=[0.8,0.9]
    testing_range=[0.9, 1]

    #Number of examples after taking into account bins removed for lag alignment
    num_examples=X.shape[0]

    #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
    #This makes it so that the different sets don't include overlapping neural data
    training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
    testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

    #Get training data
    X_train=X[training_set,:]
    y_train=y[training_set,:]

    #Get testing data
    X_test=X[testing_set,:]
    y_test=y[testing_set,:]

    #Get validation data
    X_valid=X[valid_set,:]
    y_valid=y[valid_set,:]
    
    #Initialize matrices for neural data in Naive bayes format
    num_nrns=X_train.shape[1]
    X_b_train=np.empty([X_train.shape[0]-N+1,num_nrns])
    X_b_valid=np.empty([X_valid.shape[0]-N+1,num_nrns])
    X_b_test=np.empty([X_test.shape[0]-N+1,num_nrns])

    #Below assumes that bins_current=1 (otherwise alignment will be off by 1 between the spikes and outputs)

    #For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
    #Do this for the training/validation/testing sets
    for i in range(num_nrns):
        X_b_train[:,i]=N*np.convolve(X_train[:,i], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
        X_b_valid[:,i]=N*np.convolve(X_valid[:,i], np.ones((N,))/N, mode='valid')
        X_b_test[:,i]=N*np.convolve(X_test[:,i], np.ones((N,))/N, mode='valid')

    #Make integer format
    X_b_train=X_b_train.astype(int)
    X_b_valid=X_b_valid.astype(int)
    X_b_test=X_b_test.astype(int)

    #Make y's aligned w/ X's
    #e.g. we have to remove the first y if we are using 1 bin before, and have to remove the last y if we are using 1 bin after
    if bins_before>0 and bins_after>0:
        y_train=y_train[bins_before:-bins_after,:]
        y_valid=y_valid[bins_before:-bins_after,:]
        y_test=y_test[bins_before:-bins_after,:]
    
    if bins_before>0 and bins_after==0:
        y_train=y_train[bins_before:,:]
        y_valid=y_valid[bins_before:,:]
        y_test=y_test[bins_before:,:]

        
    #Declare model
    #The parameter "encoding_model" can either be linear or quadratic, although additional encoding models could later be added.

    #The parameter "res" is the number of bins used (resolution) for decoding predictions
    #So if res=100, we create a 100 x 100 grid going from the minimum to maximum of the output variables (x and y positions)
    #The prediction the decoder makes will be a value on that grid 
    return X_b_train, y_train, X_b_test, y_test

def MSE(pred, test, nx = 48, maze_type = 1):
    print("    MSE")
    with open('decoder_DMatrix.pkl', 'rb') as handle:
        D_Matrice = pickle.load(handle)
        if nx == 12:
            D = D_Matrice[6+maze_type] / nx * 12
        elif nx == 24:
            D = D_Matrice[3+maze_type] / nx * 12
        elif nx == 48:
            D = D_Matrice[maze_type] / nx * 12
        else:
            print("self.res value error! Report by self._dis")
            return
    
    if pred.shape[0] != test.shape[0]:
        print("    Warning! MazeID_test shares different dimension with MazeID_prediected.")

    abd = np.zeros_like(pred,dtype = np.float64)
    for k in range(abd.shape[0]):
        abd[k] = D[pred[k]-1, test[k]-1] * 8
        
    # average of AbD
    MAE = np.nanmean(abd)
    MSE = np.nanmean(abd**2)
    std_abd = np.std(abd**2)
    RMSE = np.sqrt(MSE)
    print("    RMSE:",RMSE)
    print('    MAE', MAE)
    return MSE, std_abd, RMSE, MAE

def Accuracy(pred, test, nx = 48):
    print("Accuracy")
    abHit = np.zeros(test.shape[0], dtype = np.float64)
    for i in range(abHit.shape[0]):
            if test[i] == pred[i]:
                abHit[i] = 1
        
    # if hits the same father node we think it is a general hit event.
    if nx in [24,48]:
        geHit = np.zeros(test.shape[0], dtype = np.float64)
        S2FGraph = Quarter2FatherGraph if nx == 24 else Son2FatherGraph
        for i in range(geHit.shape[0]):
            if S2FGraph[int(test[i])] == S2FGraph[int(pred[i])]:
                geHit[i] = 1
        abHit = np.nanmean(abHit)
        geHit = np.nanmean(geHit)
        print("  ",abHit,geHit)
        return abHit, geHit
    else:
        abHit = np.nanmean(abHit)
        print("  ",abHit)
        return abHit


from mylib.decoding_utils import *

# pred are results of predictions of a certain decoder
# test are actual maze id value for a vector
pred = np.array([...], dtype = np.int64)
test = np.array([...], dtype = np.int64)
mse, Standard_deviation, RMSE, MAE = MSE(pred, test, nx = 48, maze_type = 1)
# input correct maze_type or you will get wrong output.

absolute_accuracy, general_accuracy = Accuracy(pred, test, nx = 48)
