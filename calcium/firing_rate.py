
from mylib.preprocessing_ms import place_field, shuffle_test, Generate_SilentNeuron, calc_ratemap
import scipy.stats
import numpy as np

def calc_rate_map_properties(
    maze_type: int,
    ms_time_behav: np.ndarray,
    Spikes: np.ndarray,
    spike_nodes: np.ndarray,
    ms_speed_behav: np.ndarray,
    dt: np.ndarray,
    Ms: np.ndarray,
    save_loc: str,
    kwargs: dict = {}
):
    n_neuron = Spikes.shape[0]
    _nbins = 2304
    _coords_range = [0, _nbins +0.0001 ]

    occu_time, _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            dt,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)

    # Generate silent neuron
    spike_num_thre = 10
    SilentNeuron = Generate_SilentNeuron(Spikes = Spikes, threshold = spike_num_thre)
    print(f'       These neurons have spikes less than {spike_num_thre}:', SilentNeuron)
    # Calculating firing rate
    
    rate_map_all, rate_map_clear, smooth_map_all, nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = spike_nodes,
                                                                        _nbins = 48*48, occu_time = occu_time, Ms = Ms, is_silent = SilentNeuron)

    # Generate place field
    place_field_all = place_field(n_neuron = n_neuron, smooth_map_all = smooth_map_all, maze_type = maze_type)
    
    
    print("    D. Shuffle test for spatial information of each cells to identified place cells. Shuffle method including 1) inter spike intervals(isi), 2) rigid spike shifts, 3) purely random rearrangement of spikes.")
    # total occupation time
    t_total = np.nansum(occu_time)/1000
    # time fraction at each spatial bin
    t_nodes_frac = occu_time / 1000 / (t_total+ 1E-6)

    # Save all variables in a dict
    trace_ms = {'Spikes':Spikes, 'spike_nodes':spike_nodes, 'ms_time_behav':ms_time_behav, 'ms_speed_behav':ms_speed_behav, 'n_neuron':n_neuron, 
                't_total':t_total, 'dt': dt, 't_nodes_frac':t_nodes_frac, 'SilentNeuron':SilentNeuron, 'rate_map_all':rate_map_all, 'rate_map_clear':rate_map_clear, 
                'smooth_map_all':smooth_map_all, 'nanPos':nanPos, 'place_field_all':place_field_all, 'occu_time_spf': occu_time, 'p': save_loc}
    
    # Shuffle test
    trace_ms = shuffle_test(trace_ms, Ms, **kwargs)
    #plot_field_arange(trace, save_loc=os.path.join(trace['p'], 'PeakCurve'))
    return trace_ms
