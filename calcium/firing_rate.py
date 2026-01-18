
from mylib.preprocessing_ms import place_field, shuffle_test, Generate_SilentNeuron, calc_ratemap
from mylib.preprocessing_ms import half_half_correlation, odd_even_correlation, CrossLapsCorrelation
from mylib.preprocessing_ms import OldMapSplit, field_specific_correlation, count_field_number, field_register, calc_SI
import scipy.stats
import numpy as np
import pickle
import os

def calc_rate_map_properties(
    maze_type: int,
    ms_time_behav: np.ndarray,
    Spikes: np.ndarray,
    spike_nodes: np.ndarray,
    ms_speed_behav: np.ndarray,
    dt: np.ndarray,
    Ms: np.ndarray,
    save_loc: str,
    behavior_paradigm: str,
    kwargs: dict = {},
    spike_num_thre = 10,
    placefield_kwargs: dict = {"thre_type": 2, "parameter": 0.2},
    is_shuffle: bool = True,
    is_calc_fields: bool = True
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
    SilentNeuron = Generate_SilentNeuron(Spikes = Spikes, threshold = spike_num_thre)
    print(f'       These neurons have spikes less than {spike_num_thre}:', SilentNeuron)
    # Calculating firing rate
    
    rate_map_all, rate_map_clear, smooth_map_all, nanPos = calc_ratemap(Spikes = Spikes, spike_nodes = spike_nodes,
                                                                        _nbins = 48*48, occu_time = occu_time, Ms = Ms, is_silent = SilentNeuron)
    
    print("    D. Shuffle test for spatial information of each cells to identified place cells. Shuffle method including 1) inter spike intervals(isi), 2) rigid spike shifts, 3) purely random rearrangement of spikes.")
    # total occupation time
    t_total = np.nansum(occu_time)/1000
    # time fraction at each spatial bin
    t_nodes_frac = occu_time / 1000 / (t_total)

    # Save all variables in a dict
    trace_ms = {'Spikes':Spikes, 'spike_nodes':spike_nodes, 'ms_time_behav':ms_time_behav, 'ms_speed_behav':ms_speed_behav, 'n_neuron':n_neuron, 
                't_total':t_total, 'dt': dt, 't_nodes_frac':t_nodes_frac, 'SilentNeuron':SilentNeuron, 'rate_map_all':rate_map_all, 'rate_map_clear':rate_map_clear, 
                'smooth_map_all':smooth_map_all, 'nanPos':nanPos, 'occu_time_spf': occu_time, 'p': save_loc, 'maze_type': maze_type}

    # Shuffle test
    if is_shuffle:
        trace_ms = shuffle_test(trace_ms, Ms, **kwargs)
    else:
        trace_ms['SI_all'] = calc_SI(
            trace_ms['Spikes'], 
            rate_map = trace_ms['rate_map_all'], 
            t_total = trace_ms['t_total'], 
            t_nodes_frac = trace_ms['t_nodes_frac']
        )
    #plot_field_arange(trace, save_loc=os.path.join(trace['p'], 'PeakCurve'))
    
    # Generate place field
    if is_calc_fields:
        trace_ms['place_field_all'] = place_field(
            trace=trace_ms,
            **placefield_kwargs
        )
        
        trace_ms = count_field_number(trace_ms)
        try:
            trace_ms = field_register(trace_ms)
        except:
            pass
    return trace_ms
