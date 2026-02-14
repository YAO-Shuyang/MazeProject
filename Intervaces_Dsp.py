from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *
from mylib.local_path import *
from scipy.stats import poisson, norm, nbinom, gamma, kstest, ks_2samp, anderson, anderson_ksamp 
from scipy.stats import linregress, pearsonr, chisquare, ttest_1samp, ttest_ind
from scipy.stats import ttest_rel, levene, spearmanr
from mylib.stats.ks import poisson_kstest, normal_discrete_kstest, nbinom_kstest
from mylib.behavior.correct_rate import lap_wise_decision_rate
from mylib.stats.indept import indept_field_properties, indept_field_properties_mutual_info, indept_field_evolution_CI
from mylib.stats.ks import lognorm_kstest
from mylib.behavior.correct_rate import calc_behavioral_score_dsp
from mylib.calcium.dsp_ms import classify_lap
from mylib.dsp.neural_traj import lda_dim_reduction, calc_trajectory_similarity
from mylib.maze_graph import CorrectPath_maze_1 as CP
from sklearn.preprocessing import StandardScaler

seg1 = np.array([1,13,14,26,27,15,3,4,5])
seg2 = np.array([6,18,17,29,30,31,19,20,21,9,10,11,12,24])
seg3 = np.array([23,22,34,33,32,44,45,46,47,48,60,59,58,57,56,68,69,70,71,72,84,83,95])
seg4 = np.array([94,82,81,80,92,104,103,91,90,78,79,67,55,54])
seg5 = np.array([66,65,64,63,75,74,62,50,51,39,38,37,49,61,73,85,97])
seg6 = np.array([109,110,122,123,111,112,100])
seg7 = np.array([99,87,88,76,77,89,101,102,114,113,125,124,136,137,138,126,127,115,116,117,129,141,142,130,131,132,144])
segs = np.array([
    seg1.shape[0], 
    seg2.shape[0], 
    seg3.shape[0], 
    seg4.shape[0], 
    seg5.shape[0], 
    seg6.shape[0], 
    seg7.shape[0]
])
segs = np.cumsum(segs)
color = sns.color_palette("rainbow", 7)
def plot_segments(ax: Axes, dx: float = 0, dy: float = 0) -> Axes:
    ax.plot(np.arange(segs[0]) + dx, np.repeat(dy, seg1.shape[0]), linewidth = 0.5, color = color[0])
    ax.plot(np.arange(segs[0], segs[1]) + dx, np.repeat(dy, seg2.shape[0]), linewidth = 0.5, color = color[1])
    ax.plot(np.arange(segs[1], segs[2]) + dx, np.repeat(dy, seg3.shape[0]), linewidth = 0.5, color = color[2])
    ax.plot(np.arange(segs[2], segs[3]) + dx, np.repeat(dy, seg4.shape[0]), linewidth = 0.5, color = color[3])
    ax.plot(np.arange(segs[3], segs[4]) + dx, np.repeat(dy, seg5.shape[0]), linewidth = 0.5, color = color[4])
    ax.plot(np.arange(segs[4], segs[5]) + dx, np.repeat(dy, seg6.shape[0]), linewidth = 0.5, color = color[5])
    ax.plot(np.arange(segs[5], segs[6]) + dx, np.repeat(dy, seg7.shape[0]), linewidth = 0.5, color = color[6])
    return ax

# Fig0803
def ROINumber_DSP_Interface(trace: dict, spike_threshold = 30, variable_names = None):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['ROI Number'])
    return np.array([trace['n_neuron']])

# Fig0805
def PlaceCellsProportion_DSP_Interface(trace: dict, spike_threshold = 30, variable_names = None):   
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Proportion'])
    is_pc = np.vstack([trace[f'node {i}']['is_placecell'] for i in range(10)])
    return np.array([
        np.where(np.sum(is_pc, axis=0) > 0)[0].shape[0] / is_pc.shape[1]
    ])

# Fig0806
def ProportionOfPCAcrossRoutes_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Route', 'Proportion'])
    is_pc = np.vstack([trace[f'node {i}']['is_placecell'] for i in range(10)])
    
    
    return np.array([1, 2, 3, 4, 1, 1, 5, 6, 7, 1]), np.sum(is_pc, axis=1) / is_pc.shape[1]

# Generate learning curve for cross maze paradigm. Fig0020
def LearningCurve_DSP_Interface(trace: dict, spike_threshold = 30, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'paradigm'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Lap ID', 'Route', 'Lap-wise time cost'])

    if trace['maze_type'] == 0:
        return np.array([]), np.array([])
    
    behav_time = trace['correct_time']
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    navigating_time = (behav_time[end_idx] - behav_time[beg_idx])/1000
    
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx, trace['start_from'])
    routes[routes == 4] = 0
    
    if trace['start_from'] == 'correct':
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    else:
        routes[np.where(routes != 0)[0]] += 3
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
        routes_used[routes_used != 0] = routes_used[routes_used != 0] + 3
    
    laps_id = np.array([i for i in range(1, beg_idx.shape[0] + 1)])
    return laps_id, routes_id, navigating_time

def LearningCurveBehavioralScore_DSP_Interface(trace: dict, variable_names: list):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Route', 'Correct Rate', 'Pass Number', 'Error Number', 'Pure Guess Correct Rate'])
    
    behav_nodes = spike_nodes_transform(trace['correct_nodes'], 12)
    behav_time = cp.deepcopy(trace['correct_time'])
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx)
    
    correct_rates, pass_nums, err_nums, pureguess = [], [], [], []
    
    for i in range(7):
        idx = np.where(routes == i)[0]
        
        err_num, pass_num, std_err = calc_behavioral_score_dsp(
            route=i,
            behav_nodes=np.concatenate([behav_nodes[beg_idx[j]:end_idx[j]+1] for j in idx]),
            behav_time=np.concatenate([behav_time[beg_idx[j]:end_idx[j]+1] for j in idx])
        )
        
        correct_rates.append(1-err_num/pass_num)
        err_nums.append(err_num)
        pass_nums.append(pass_num)
        pureguess.append(1-std_err)
    
    return np.unique(routes), np.array(correct_rates, np.float64), np.array(pass_nums, np.int64), np.array(err_nums, np.int64), np.array(pureguess, np.float64)

# Fig0807 Running Speed
def RunningSpeed_DSP_Interface(trace: dict, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Route', 'Lap', 'Position', 'Speed'])
    
    D = GetDMatrices(1, 48)
    
    beg_idx, end_idx = LapSplit(trace, trace['paradigm'])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx)

    beg_time_ori, end_time_ori = trace['lap beg time'], trace['lap end time']
    speed = []
    poses = []
    lap = []
    route = []
    
    for i in range(beg_time_ori.shape[0]):
        idx = np.where(
            (trace['correct_time'] >= beg_time_ori[i]) & 
            (trace['correct_time'] <= end_time_ori[i])
        )[0]
        smoothed_speed = trace['correct_speed'][idx]
        
        CP = CP_DSP[routes[i]]
        
        behav_nodes = trace['correct_nodes'][idx].astype(np.int64)
        old_nodes = spike_nodes_transform(behav_nodes, 12)
        
        idx = np.where(np.isin(old_nodes, CP))[0]
        dist = D[behav_nodes-1, 0] / (np.max(D)+1) * 111
        dist = (dist // 1).astype(np.int64)
        
        smoothed_speed = smoothed_speed[idx]
        dist = dist[idx]
        
        speed_mean = np.full(111, np.nan)
        for j in range(111):
            speed_mean[j] = np.nanmean(smoothed_speed[np.where(dist == j)[0]])
        
        speed.append(speed_mean)
        poses.append(np.arange(111))
        lap.append(np.repeat(i, 111))
        route.append(np.repeat(routes[i], 111))
    
    return np.concatenate(route), np.concatenate(lap), np.concatenate(poses), np.concatenate(speed)

# Fig0809
def LapwiseTimeImprovement_DSP_Interface(trace: dict, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['correct_time', 'correct_nodes', 'maze_type'])
    VariablesInputErrorCheck(input_variable = variable_names, check_variable = ['Route', 'Lap', 'Time'])
    
    beg, end = LapSplit(trace, trace['paradigm'])
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg)
    
    R = np.vstack([np.repeat(i, 10) for i in range(7)])
    L = np.hstack([np.repeat(i, 7)[:, np.newaxis] for i in range(10)])
    T = np.zeros((7, 10))
    
    for i in range(7):
        idx = np.where(routes == i)[0]
        if len(idx) < 10:
            T[i, :len(idx)] = trace['lap end time'][idx] - trace['lap beg time'][idx]
            T[i, len(idx):] = np.nan
        else:
            T[i, :] = trace['lap end time'][idx[:10]] - trace['lap beg time'][idx[:10]]
    
    return R.flatten(), L.flatten(), T.flatten()
    
def MazeSegmentsPVC_DSP_Interface(trace: dict, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['segments_pvc'])
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segments', 'Mean PVC', 'Compare Groups']
    )
    return (
        np.concatenate([trace['segments_x'] for i in range(4)]),
        trace['segments_pvc'].flatten(),
        np.array(['0-4', '4-5', '5-9', '0-9'] * trace['segments_x'].shape[0])
    )

from mylib.calcium.dsp_ms import get_son_area
#0812 - Segmented Correlation across Routes
def SegmentedCorrelationAcrossRoutes_DSP_Interface(trace: dict, variable_names = None):
    KeyWordErrorCheck(trace, __file__, ['segments_pvc'])
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Position', 'aPVC', 'Routes']
    )
    
    segment_pieces = []
    segment_pvc = []
    routes = []
    
    D = GetDMatrices(1, 48)

    for j in range(10):
        if j in [0, 5]:
            continue
        
        if j >= 5:
            i=5
            r_offset = 2
        else:
            i=0
            r_offset = 0
  
        pc_idx = np.where(
            (trace[f'node {i}']['is_placecell'] == 1) |
            (trace[f'node {i+4}']['is_placecell'] == 1)
        )
        
        n_bins = CP_DSPs[1][0].shape[0]
        pvc = np.zeros((n_bins, 16), np.float64)
        for k in range(n_bins):
            sonbins = Father2SonGraph[CP_DSPs[1][0][k]]
            for b in range(16):
                pvc[k, b] = np.corrcoef(
                    trace[f'node {i}']['smooth_map_all'][pc_idx, sonbins[b]-1],
                    trace[f'node {j}']['smooth_map_all'][pc_idx, sonbins[b]-1]
                )[0, 1]
        
        segment_pieces.append(np.arange(n_bins))
        segment_pvc.append(np.nanmean(pvc, axis=1))
        routes.append(np.repeat(j-r_offset, n_bins)) if j not in [4, 9] else routes.append(np.repeat(0, n_bins))
    
    return np.concatenate(segment_pieces), np.concatenate(segment_pvc), np.concatenate(routes)

def SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface(trace: dict, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Position', 'rPVC', 'Routes', 'Type']
    )
    pos = []
    rpvc = []
    routes = []
    isctrl = []
    D = GetDMatrices(1, 48)
    for i, k in zip([0, 5], [4, 9]):
        jrange = range(1,4) if i == 0 else range(6, 9)
        rrange = range(1,4) if i == 0 else range(4, 7)
        for j, rt in zip(jrange, rrange):
            
            pc_idx = np.where(
                (trace[f'node {i}']['is_placecell'] == 1) |
                (trace[f'node {i+4}']['is_placecell'] == 1)
            )
            rt_len = CP_DSPs[1][rt].shape[0]
            ctrl_bins = CP_DSPs[1][0][-rt_len:]
            
            PVC = np.zeros((2, rt_len, 16), np.float64)
            for b in range(rt_len):
                sonbins = np.asarray(Father2SonGraph[CP_DSPs[1][rt][b]])
                sb_ordered_rt = np.argsort(D[sonbins-1, SP_DSPs[1][rt]-1])
                sb_ordered_rt = sonbins[sb_ordered_rt]
                
                sonbins_ctrl = np.asarray(Father2SonGraph[CP_DSPs[1][0][-rt_len+b]])
                sonbins_0 = np.asarray(Father2SonGraph[CP_DSPs[1][0][b]])
            
                for n in range(16):
                    PVC[0, b, n] = np.corrcoef(
                        trace[f'node {j}']['smooth_map_all'][pc_idx, sb_ordered_rt[n]-1],
                        trace[f'node {i}']['smooth_map_all'][pc_idx, sonbins_0[n]-1]
                    )[0, 1]
                    PVC[1, b, n] = np.corrcoef(
                        trace[f'node {k}']['smooth_map_all'][pc_idx, sonbins_ctrl[n]-1],
                        trace[f'node {i}']['smooth_map_all'][pc_idx, sonbins_0[n]-1]
                    )[0, 1]
            
            for dtype in range(2):
                pos.append(np.arange(rt_len))
                rpvc.append(np.nanmean(PVC[dtype, :, :], axis=1))
                routes.append(np.repeat(rt, rt_len))
                isctrl.append(np.repeat(['Real', 'Ctrl'][dtype], rt_len))
        
    return np.concatenate(pos), np.concatenate(rpvc), np.concatenate(routes), np.concatenate(isctrl)

def SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface2(trace: dict, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segments', 'Mean PVC', 'Compare Groups', 'Control For Route']
    )
    segment_pieces = []
    segment_pvc = []
    groups = []
    routes = []
    control_for_route = []
    seg_temp = np.concatenate([[0], segs])
    
    D = GetDMatrices(1, 48)
    
    seg_temp = np.array([seg_temp[0], seg_temp[2], seg_temp[4], seg_temp[6], seg_temp[1], seg_temp[3], seg_temp[5], seg_temp[7]])
    for u, i in enumerate([6, 1, 7, 2, 8, 3]):
        for v, j in enumerate([6, 1, 7, 2, 8, 3]):
            if u >= v:
                continue
            
            son_bins1 = get_son_area(CP_DSP[trace[f'node {j}']['Route']])
            son_bins2 = get_son_area(CP_DSP[trace[f'node {i}']['Route']])
            
            D1 = D[son_bins1-1, SP_DSP[trace[f'node {j}']['Route']]-1]
            D2 = D[son_bins2-1, SP_DSP[trace[f'node {i}']['Route']]-1]
            
            idx1 = np.argsort(D1)
            idx2 = np.argsort(D2)[:idx1.shape[0]]
            
            pc_idx = np.where(
                (trace[f'node {i}']['is_placecell'] == 1) |
                (trace[f'node {j}']['is_placecell'] == 1)
            )[0]
            
            PVC = np.zeros(idx1.shape[0], np.float64)
            for k in range(idx1.shape[0]):
                PVC[k] = np.corrcoef(
                    trace[f'node {j}']['smooth_map_all'][pc_idx, son_bins1[idx1[k]]-1],
                    trace[f'node {i}']['smooth_map_all'][pc_idx, son_bins2[idx2[k]]-1]
                )[0, 1]
                
            n_len = CP_DSP[trace[f'node {j}']['Route']].shape[0]
            dist = D1[idx1]
            dist = dist / (np.max(dist) + 0.0001) * n_len
            dist = (dist // 1).astype(np.int64)
            
            pvc_norm = np.zeros(n_len)
            for k in range(n_len):
                pvc_norm[k] = np.nanmean(PVC[dist == k])
            
            segment_pieces.append(np.arange(0, n_len))
            segment_pvc.append(pvc_norm)
            groups.append(np.repeat(f'{i}-{j}', n_len))
            control_for_route.append(np.repeat('Real', n_len))    
            
            # Control
                
            PVC = np.zeros(idx1.shape[0], np.float64)
            for k in range(idx1.shape[0]):
                PVC[k] = np.corrcoef(
                    trace[f'node {i}']['smooth_map_all'][pc_idx, son_bins1[idx1[k]]-1],
                    trace[f'node {i}']['smooth_map_all'][pc_idx, son_bins2[idx2[k]]-1]
                )[0, 1]
                        
            n_len = CP_DSP[trace[f'node {j}']['Route']].shape[0]
            dist = D1[idx1]
            dist = dist / (np.max(dist) + 0.0001) * n_len
            dist = (dist // 1).astype(np.int64)
                    
            pvc_norm = np.zeros(n_len)
            for k in range(n_len):
                pvc_norm[k] = np.nanmean(PVC[dist == k])
                    
            segment_pieces.append(np.arange(0, n_len))
            segment_pvc.append(pvc_norm)
            groups.append(np.repeat(f'{i}-{j}', n_len))
            control_for_route.append(np.repeat('Control', n_len)) 
 
    return np.concatenate(segment_pieces), np.concatenate(segment_pvc), np.concatenate(groups), np.concatenate(control_for_route)

def SegmentedTrajectoryDistance_DSP_Interface(
    trace, 
    variable_names = None
):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = [
            'Segments', 'Routes', 'Group', 
            'Mean Trajectory Distance', 'SubSpace Type'
        ]
    )
    
    neural_traj = trace['neural_traj']
    pos_traj = trace['pos_traj']
    lap_ids = trace['traj_lap_ids']
    route_ids = trace['traj_route_ids']
    segment_traj = trace['traj_segment_ids']
    
    pos_traj = spike_nodes_transform(pos_traj, 12).astype(np.float64)
    
    # Set bin at the incorrect track as NAN
    for i in range(pos_traj.shape[0]):
        if pos_traj[i] not in CP:
            pos_traj[i] = np.nan
            
    # Delete NAN 
    idx = np.where(np.isnan(pos_traj) == False)[0]
    pos_traj = pos_traj[idx].astype(np.int64)
    lap_ids = lap_ids[idx]
    route_ids = route_ids[idx]
    neural_traj = neural_traj[:, idx]
    segment_traj = segment_traj[idx]

    # Extract Place cells for analysis only
    pc_idx = np.unique(
        np.concatenate(
            [np.where(trace[f'node {i}']['is_placecell'] == 1)[0] for i in range(10)]
        )
    )
    neural_traj = neural_traj[pc_idx, :]
    
    # Convert to graph
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]

    # Computation Starts.
    seg = []
    route = []
    targ_route = []
    mean_trajectory_distance = []
    sub_space_type = []
    
    route_order = np.array([0, 4, 1, 5, 2, 6, 3])
    
    # LDA Dim reduction maximizing the separability of position.
    print(trace['p'])
    for i in range(1, 7):
        legal_route = np.intersect1d(
            CP_DSP[0], CP_DSP[i]
        )
        idx = np.where(
            (segment_traj == i) & (pos_traj_reord <= 111)
        )[0]
        
        reduced_data = lda_dim_reduction(
            neural_traj[:, idx],
            pos_traj_reord[idx],
            n_components=6
        )
        
        mat, laps, routes = calc_trajectory_similarity(
            reduced_data=reduced_data,
            lap_ids=lap_ids[idx],
            route_ids=route_ids[idx],
            dim=6
        )        
        
        
        for j, rt in enumerate(route_order[:i+1]):
            seg.append(i)
            route.append(rt)
            targ_route.append("Within-Routes")
                
            idx1 = np.where(routes == rt)[0]
            idx2 = np.where(routes == rt)[0]
                
            mean_trajectory_distance.append(
                np.nanmean(
                    mat[idx1, :][:, idx2]
                )
            )
                
            sub_space_type.append(0)
            
            idx2 = np.where(routes != rt)[0]
            seg.append(i)
            route.append(rt)
            targ_route.append("Across-Routes")
            mean_trajectory_distance.append(
                np.nanmean(
                    mat[idx1, :][:, idx2]
                )
            )
            sub_space_type.append(0)
                            
        ndim=i
        reduced_data = lda_dim_reduction(
            neural_traj[:, idx],
            route_ids[idx],
            n_components=ndim
        )
        
        mat, laps, routes = calc_trajectory_similarity(
            reduced_data=reduced_data,
            lap_ids=lap_ids[idx],
            route_ids=route_ids[idx],
            dim=ndim
        )        
        
        for j, rt in enumerate(route_order[:i+1]):
            seg.append(i)
            route.append(rt)
            targ_route.append("Within-Routes")
                
            idx1 = np.where(routes == rt)[0]
            idx2 = np.where(routes == rt)[0]
                
            mean_trajectory_distance.append(
                np.nanmean(
                    mat[idx1, :][:, idx2]
                )
            )
                
            sub_space_type.append(1)
            
            idx2 = np.where(routes != rt)[0]
            seg.append(i)
            route.append(rt)
            targ_route.append("Across-Routes")
            mean_trajectory_distance.append(
                np.nanmean(
                    mat[idx1, :][:, idx2]
                )
            )
            sub_space_type.append(1)
    
    print()
    return (
        np.array(seg),
        np.array(route),
        np.array(targ_route),
        np.array(mean_trajectory_distance),
        np.array(sub_space_type)
    )
    
def LatentSpaceOrthogonality_DSP_Interface(
    trace, 
    variable_names = None
):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = [
            'Segments', 'Routes', 'Target Route', 
            'Mean Trajectory Distance', 'SubSpace Type'
        ]
    )
    
    neural_traj = trace['neural_traj']
    pos_traj = trace['pos_traj']
    lap_ids = trace['traj_lap_ids']
    route_ids = trace['traj_route_ids']
    segment_traj = trace['traj_segment_ids']
    
    pos_traj = spike_nodes_transform(pos_traj, 12).astype(np.float64)
    
    # Set bin at the incorrect track as NAN
    for i in range(pos_traj.shape[0]):
        if pos_traj[i] not in CP:
            pos_traj[i] = np.nan
            
    # Delete NAN 
    idx = np.where(np.isnan(pos_traj) == False)[0]
    pos_traj = pos_traj[idx].astype(np.int64)
    lap_ids = lap_ids[idx]
    route_ids = route_ids[idx]
    neural_traj = neural_traj[:, idx]
    segment_traj = segment_traj[idx]

    # Extract Place cells for analysis only
    pc_idx = np.unique(
        np.concatenate(
            [np.where(trace[f'node {i}']['is_placecell'] == 1)[0] for i in range(10)]
        )
    )
    neural_traj = neural_traj[pc_idx, :]
    
    # Convert to graph
    G = NRG[1]
    pos_traj_reord = np.zeros_like(pos_traj)
    for i in range(pos_traj.shape[0]):
        pos_traj_reord[i] = G[pos_traj[i]]
    
    
    # Computation Starts.
    seg = []
    route = []
    targ_route = []
    mean_trajectory_distance = []
    sub_space_type = []
    
    # LDA Dim reduction maximizing the separability of position.
    for i in range(1, 7):
        idx = np.where(segment_traj == i)[0]
        reduced_data = lda_dim_reduction(
            neural_traj[:, idx],
            pos_traj_reord[idx],
            n_components=3
        )
        mat, laps, routes = calc_trajectory_similarity(
            reduced_data=reduced_data,
            lap_ids=lap_ids[idx],
            route_ids=route_ids[idx],
            dim=3
        )        
        
        for j in range(6):
            for k in range(j, 7):
                seg.append(i)
                route.append(j)
                targ_route.append(k)
                
                idx1 = np.where(routes == j)[0]
                idx2 = np.where(routes == k)[0]
                
                mean_trajectory_distance.append(
                    np.nanmean(
                        mat[idx1, :][:, idx2]
                    )
                )
                
                sub_space_type.append(0)
                
    # LDA Dim reduction maximizing the separability of routes.
    for i in range(2, 7):
        idx = np.where(segment_traj == i)[0]
        reduced_data = lda_dim_reduction(
            neural_traj[:, idx],
            route_ids[idx],
            n_components=2
        )
        mat, laps, routes = calc_trajectory_similarity(
            reduced_data=reduced_data,
            lap_ids=lap_ids[idx],
            route_ids=route_ids[idx],
            dim=2
        )        
        
        for j in range(6):
            for k in range(j, 7):
                seg.append(i)
                route.append(j)
                targ_route.append(k)
                
                idx1 = np.where(routes == j)[0]
                idx2 = np.where(routes == k)[0]
                
                mean_trajectory_distance.append(
                    np.nanmean(
                        mat[idx1, :][:, idx2]
                    )
                )
                
                sub_space_type.append(1)
                
    return (
        np.array(seg),
        np.array(route),
        np.array(targ_route),
        np.array(mean_trajectory_distance),
        np.array(sub_space_type)
    )

from mylib.dsp.neural_traj import preprocess_neural_traj, pca_dim_reduction, lda_dim_reduction
def SubspacesOrthogonality_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segment', 'Smallest Singlar Value', 'Conditional Number', 'Subspace Comparison Type','Shinkage']
    )
    
    PCA_Dimensions = [100, 50, 30]
    
    Shrink = []
    segments = []
    singvals = []
    condno = []
    subspace_type = []
    
    for shrinkage in tqdm([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        for i in range(1, 7):
            
            res = preprocess_neural_traj(trace, segment=i)

            neural_traj = res['neural_traj']
            lap_ids = res['traj_lap_ids']
            route_ids = res['traj_route_ids']
            segment_traj = res['traj_segment_ids']
            pos_traj = res['pos_traj']
            pos_traj_reord = res['pos_traj_reord']
            
            if neural_traj.shape[0] <= 30:
                continue
            
            pca_n = min(neural_traj.shape[0], shrinkage)
            #reduced_data_pca, pca = pca_dim_reduction(neural_traj, pca_n)
            reduced_data_pca = neural_traj.T
            _, lda_pos = lda_dim_reduction(
                reduced_data_pca.T, 
                pos_traj_reord, 
                n_components=6,
                solver="eigen",
                shrinkage=shrinkage
            )
            _, lda_route = lda_dim_reduction(
                reduced_data_pca.T, 
                route_ids, 
                n_components=i,
                solver="eigen",
                shrinkage=shrinkage
            )
            
            idx = np.where(route_ids == 0)[0]
            _, lda_lap = lda_dim_reduction(
                reduced_data_pca.T[:, idx], 
                lap_ids[idx], 
                n_components=6,
                solver="eigen",
                shrinkage=shrinkage
            )
            
            """
            # Get total weights of each neuron
            W1 = pca.components_.T @ lda_pos.scalings_
            W2 = pca.components_.T @ lda_route.scalings_
            W3 = pca.components_.T @ lda_lap.scalings_
            """
            W1 = lda_pos.scalings_
            W2 = lda_route.scalings_
            W3 = lda_lap.scalings_
            
            C1 = np.hstack([W1[:, :6], W2])
            C2 = np.hstack([W1[:, :6], W3[:, :6]])
            C3 = np.hstack([W2, W3[:, :6]])
            
            if np.where(np.isnan(C1))[0].shape[0] > 0:
                raise ValueError("nan detected in C1")
        
            if np.where(np.isnan(C2))[0].shape[0] > 0:
                raise ValueError("nan detected in C2")
            
            if np.where(np.isnan(C3))[0].shape[0] > 0:
                raise ValueError("nan detected in C3")
            
            # pos subspace vs. route subspace
            cond_number = np.linalg.cond(C1)
            u, s, vh = np.linalg.svd(C1)
            Shrink.append(shrinkage)
            segments.append(i)
            singvals.append(s[-1])
            condno.append(cond_number)
            subspace_type.append(0)
            
            # pos subspace vs. lap subspace
            cond_number = np.linalg.cond(C2)
            u, s, vh = np.linalg.svd(C2)
            Shrink.append(shrinkage)
            segments.append(i)
            singvals.append(s[-1])
            condno.append(cond_number)
            subspace_type.append(1)
            
            # route subspace vs. lap subspace
            cond_number = np.linalg.cond(C3)
            u, s, vh = np.linalg.svd(C3)
            Shrink.append(shrinkage)
            segments.append(i)
            singvals.append(s[-1])
            condno.append(cond_number)
            subspace_type.append(2)
            
            del res
            
    return (
        np.array(segments),
        np.array(singvals),
        np.array(condno),
        np.array(subspace_type),
        np.array(Shrink)
    )
    
    
# Fig0824
def AllocentricFieldProportion_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segment', 'Proportion']
    )
    
    field_reg = cp.deepcopy(trace['field_reg_modi'])
    field_reg[field_reg == 2] = 1
    field_info = trace['field_info']
    field_segs = trace['field_segs']

    prop = np.ones(6)
    print(np.unique(field_segs))
    for seg in range(1, 7):
        field_idx = np.where((field_segs == seg+1))[0]
        
        allo_field_idx = np.where(
            (field_segs == seg+1) & 
            (
                (field_reg[0, :] >= 1) &
                (field_reg[4, :] >= 1) &
                (field_reg[5, :] >= 1) &
                (field_reg[9, :] >= 1)
            ) &
            (np.nansum(field_reg[1:4, :], axis=0)+np.nansum(field_reg[6:9, :], axis=0) == seg)
        )[0]
        
        if field_idx.shape[0] == 0:
            prop[seg-1] = np.nan
        else:
            prop[seg-1] = allo_field_idx.shape[0] / field_idx.shape[0]
        
    return np.arange(1, 7), prop

def FieldStateSwitchWithSegment_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segment', 'Proportion', 'Category']
    )
    
    field_reg = trace['field_reg_modi']
    field_segs = trace['field_segs']
    
    segments = np.concatenate([np.arange(7) for i in range(9)])
    
    prop = np.ones(63)
    category = np.concatenate([np.repeat(i, 7) for i in range(9)])
    
    for seg in range(7):
        field_idx = np.where(
            (field_segs == seg+1) &
            ((field_reg[0, :] != 0) | (field_reg[4, :] != 0))
        )[0]
        
        formed_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[0, :] == 0) &
            (field_reg[4, :] >= 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[0, :] >= 1) &
            (field_reg[4, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[0, :] >= 1) &
            (field_reg[4, :] >= 1)
        )[0]
        
        prop[(seg-1)] = formed_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 7] = disped_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 14] = retain_field_idx.shape[0] / field_idx.shape[0]

    for seg in range(7):
        field_idx = np.where(
            (field_segs == seg+1) &
            ((field_reg[5, :] != 0) | (field_reg[9, :] != 0))
        )[0]
        
        formed_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[5, :] == 0) &
            (field_reg[9, :] >= 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[5, :] >= 1) &
            (field_reg[9, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[5, :] >= 1) &
            (field_reg[9, :] >= 1)
        )[0]
        
        prop[(seg-1) + 21] = formed_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 28] = disped_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 35] = retain_field_idx.shape[0] / field_idx.shape[0]

    for seg in range(7):
        field_idx = np.where(
            (field_segs == seg+1) &
            ((field_reg[4, :] != 0) | (field_reg[5, :] != 0))
        )[0]
        
        formed_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[4, :] == 0) &
            (field_reg[5, :] >= 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[4, :] >= 1) &
            (field_reg[5, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[4, :] >= 1) &
            (field_reg[5, :] >= 1)
        )[0]
        
        prop[(seg-1) + 42] = formed_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 49] = disped_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 56] = retain_field_idx.shape[0] / field_idx.shape[0]

    return segments, prop, category

def Exclusivity_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["X", "Y", "P"]
    )
    
    field_reg = trace['field_reg_modi'][:, np.where(trace['field_segs'] >= 6)[0]]
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    p = np.zeros((10, 10))
    
    for i in range(10):
        for j in range(10):
            p[i, j] = np.where((field_reg[i, :] >= 1) & (field_reg[j, :] >= 1))[0].shape[0] / np.where((field_reg[i, :] >= 1) & (np.isnan(field_reg[j, :]) == False))[0].shape[0]
    
    return x.flatten(), y.flatten(), p.flatten()

def ProportionOfStartingPointTuningCell_Interface(
    trace, 
    variable_names: list | None = None
):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Proportion"]
    )
    
    mask = np.zeros(trace['n_neuron'])
    for i in range(trace['n_neuron']):
        if np.intersect1d(trace['SC_EncodePath'][i], np.array([1, 2, 3, 6, 7, 8])).shape[0] > 0:
            mask[i] = 1
    return np.array([
        np.nansum(trace['SC'] * mask) / trace['n_neuron']
    ])

# Fig0829
def StartingPointEncodingPath_Interface(
    trace, 
    variable_names: list | None = None
):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Proportion"]
    )
    
    mask = np.zeros(trace['n_neuron'])
    for i in range(trace['n_neuron']):
        if np.intersect1d(trace['SC_EncodePath'][i], np.array([1, 2, 3, 6, 7, 8])).shape[0] > 0:
            mask[i] = 1
            
# Fig0830
def StartingCellEncodedRouteNumberDistribution_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Number Of Routes", "Proportion"]
    )
    
    encoded_num = np.zeros(trace['n_neuron'])
    for i in range(trace['n_neuron']):
        if trace['SC_EncodePath'][i].shape[0] <= 1:
            continue
        other_routes = np.intersect1d(trace['SC_EncodePath'][i], np.array([1, 2, 3, 6, 7, 8]))
        route1 = np.intersect1d(trace['SC_EncodePath'][i], np.array([0, 4, 5, 9]))
        encoded_num[i] = other_routes.shape[0] + 1 if route1.shape[0] != 0 else other_routes.shape[0]
        if trace['SC'][i] == 1:
            pass
        else:
            encoded_num[i] = 0
    
    route_num = np.zeros(6)
    for i in range(6):
        route_num[i] = np.where(encoded_num == i+2)[0].shape[0] / np.where(encoded_num >= 2)[0].shape[0]
    return np.arange(2, 8), route_num

# Fig0831
def StartingCellEncodedRouteDensityDistribution_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Route", "Number"]
    )
    
    route_encoded = np.concatenate([trace['SC_EncodePath'][i] for i in np.where(trace['SC'] == 1)[0]]).astype(np.int64)
    counts = np.histogram(route_encoded, bins=10, range=(-0.5, 9.5), density=True)[0]
    
    return np.arange(10), counts

# Fig0832
def StartingFieldSpatialDistribution_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Relative Pos", "Density"]
    )
    
    pos = trace['SC_FieldCenter'][trace['SC'] == 1]
    dist = np.histogram(pos, bins=20, range=(0, 66.12), density=True)[0]
    
    return np.linspace(2.5, 97.5, 20), dist

# Fig0833
def  CombinatorialCoding_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Segment", "Route Num", "Proportion"]
    )
    
    segments = trace['field_segs']
    
    segs = []
    n_routes = []
    n_counts = []
    
    _route_indices = [
        np.array([0, 4, 5, 6, 9]),
        np.array([0, 1, 4, 5, 6, 9]),
        np.array([0, 1, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9])
    ]
    
    field_reg = trace['field_reg_modi']
    field_reg[field_reg > 1] = 1
    for seg in range(2, 8):
        idx = np.where(
            (np.sum(trace['field_reg_modi'][np.array([0, 4, 5, 9]), :], axis=0) == 4) &
            (segments == seg)
        )[0]
        n = _route_indices[seg-2].shape[0]-3
        segs.append(np.repeat(seg, n))
        n_route = np.nansum(trace['field_reg_modi'][_route_indices[seg-2], :][:, idx], axis=0) - 3
        n_count = np.histogram(n_route, bins=n, range=(0.5, n+0.5), density=True)[0]
        
        n_routes.append(np.arange(1, n+1))
        n_counts.append(n_count)
        
    segs = np.concatenate(segs)
    n_routes = np.concatenate(n_routes)
    n_counts = np.concatenate(n_counts)
    
    return segs, n_routes, n_counts

# Fig0834
def ProportionOfReliableFields_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Proportion", "Field Number"]
    )
    
    field_reg = trace['field_reg_modi']
    field_reg[field_reg > 1] = 1
    
    return np.array([
        np.where(
            np.nansum(field_reg[np.array([0, 4, 5, 9]), :], axis=0) == 4
        )[0].shape[0] / 
        np.where(
            np.nansum(field_reg[np.array([0, 4, 5, 9]), :], axis=0) >= 1
        )[0].shape[0]
    ]), np.array([np.where(
            np.nansum(field_reg[np.array([0, 4, 5, 9]), :], axis=0) >= 1
        )[0].shape[0]])
    
# Fig0835
def AllocentricProportionWithSpatialDistance_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ["Position", "Field Type"] # Field Type: 0 - Route-modulated; 1 - Route-independent
    )
    segments = trace['field_segs']
    idx = np.where(segments  >= 2)[0]
    segments = segments[idx]
    field_reg = trace['field_reg_modi'][:, idx]# field 
    field_reg[field_reg > 1] = 1
    field_info = trace['field_info'][:, idx, :]

    idx = np.where(
        np.nansum(field_reg[np.array([0, 4, 5, 9]), :], axis=0) == 4
    )[0]
    
    centers = field_info[0, idx, 2].astype(np.int64)
    centers = S2F[centers-1]
    field_reg = field_reg[:, idx]
    field_info = field_info[:, idx, :]
    segments = segments[idx]
    
    reordered_centers = np.zeros_like(centers)
    for i in range(centers.shape[0]):
        reordered_centers[i] = NRG[1][centers[i]]

    _route_indices = [
        np.array([0, 4, 5, 6, 9]),
        np.array([0, 1, 4, 5, 6, 9]),
        np.array([0, 1, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9])
    ]
    
    is_allocentric = np.zeros(field_reg.shape[1])
    for seg in range(2, 8):
        idx = np.where(segments == seg)[0]
        n_route = np.nansum(field_reg[:, idx][_route_indices[seg-2], :], axis=0) - _route_indices[seg-2].shape[0]
        
        is_allocentric[idx] = np.where(n_route >= 0, 1, 0)
    
    return reordered_centers, is_allocentric

def ModulatedProportionSpatialDistribution_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(variable_names, ["Position", "Type", "Proportion"])
    
    _route_indices = [
        np.array([4]),
        np.array([4, 6]),
        np.array([4, 6, 5]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9]),
        np.array([0, 1, 2, 4, 5, 6, 7, 9])
    ]
    
    idx = np.where(np.sum(trace['field_reg_modi'][np.array([0, 4, 5, 9]), :]) == 4)[0]
    field_reg = trace['field_reg_modi'][:, idx]
    field_info = trace['field_info'][:, idx, :]
    field_segs = trace['field_segs'][idx]
    field_centers = S2F(field_info[0, :, 2].astype(np.int64)-1)
    
    
    route_bins = np.concatenate([seg2, seg3, seg4, seg5, seg6, seg7])
    seg_ids = np.concatenate([np.repeat(i, seg.shape[0]) for i, seg in enumerate([seg2, seg3, seg4, seg5, seg6, seg7])])
    
    proportion = np.full((4, route_bins.shape[0]), np.nan)
    
    for n in route_bins:
        total_n = np.where(field_centers == n)[0]
    
# Fig0839 - Behavioral reasons for remapping - initial speed
def InitSpeedForRemapping_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(variable_names, ['Speed', 'Map Type', 'Time', 'Corr'])
    
    thres = {10212: 0.114, 10224: 0.0815, 10227: 0.0751, 10232: 0.0956}
    corr_thre = thres[int(trace['MiceID'])]
    
    beg, end = LapSplit(trace, trace['paradigm'])
    route_ids = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg)
    
    speeds, map_types, times, corrs = [], [], [], []
    for i in np.where(route_ids == 6)[0]:
        if trace['is_perfect'][i] == 0:
            continue
        
        idx = np.where(
            (trace['correct_time'] >= trace['correct_time'][beg[i]]) & 
            (trace['correct_time'] <= trace['correct_time'][beg[i]] + 2000)
        )[0]
        
        dt = np.ediff1d(trace['correct_time'])
        x, y = trace['correct_pos'][:, 0]/10, trace['correct_pos'][:, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dis = np.sqrt(dx**2 + dy**2)
        speed = dis/dt*1000
        smoothed_speed = np.convolve(speed, np.ones(5)/5, mode='same')[idx[:-1]]
        
        t = trace['correct_time'][idx[:-1]] - trace['correct_time'][idx[0]]
        corr = np.repeat(trace['lapwise_corr'][i], t.shape[0])
        map_type = np.repeat(1, t.shape[0]) if corr[0] < corr_thre else np.repeat(2, t.shape[0])
        
        speeds.append(smoothed_speed)
        map_types.append(map_type)
        times.append(t)
        corrs.append(corr)
    
    return np.concatenate(speeds), np.concatenate(map_types), np.concatenate(times), np.concatenate(corrs)

# Fig0839 - Behavioral reasons for remapping - cumulative distances
def InitCumulativeDistanceForRemapping_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(variable_names, ['Distance', 'Map Type', 'Corr', 'Time'])
    
    thres = {10212: 0.114, 10224: 0.0815, 10227: 0.0751, 10232: 0.0956}
    corr_thre = thres[int(trace['MiceID'])]
    
    beg, end = LapSplit(trace, trace['paradigm'])
    route_ids = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg)
    
    dists, map_types, corrs = [], [], []
    times = []
    for i in np.where(route_ids == 6)[0]:
        idx = np.where(
            (trace['correct_time'] >= trace['correct_time'][beg[i]]) &
            (trace['correct_time'] <= trace['correct_time'][beg[i]] + 2000)
        )[0]
        
        x, y = trace['correct_pos'][idx, 0]/10, trace['correct_pos'][idx, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dis = np.nansum(np.sqrt(dx**2 + dy**2))
        
        corr = trace['lapwise_corr'][i]
        map_type = 1 if corr < corr_thre else 2
        
        dists.append(dis)
        map_types.append(map_type)
        corrs.append(corr)
        times.append('2s')

        idx = np.where(
            (trace['correct_time'] >= trace['correct_time'][beg[i]]) &
            (trace['correct_time'] <= trace['correct_time'][beg[i]] + 1000)
        )[0]
        
        x, y = trace['correct_pos'][idx, 0]/10, trace['correct_pos'][idx, 1]/10
        dx, dy = np.ediff1d(x), np.ediff1d(y)
        dis = np.nansum(np.sqrt(dx**2 + dy**2))
        
        corr = trace['lapwise_corr'][i]
        map_type = 1 if corr < corr_thre else 2
        
        dists.append(dis)
        map_types.append(map_type)
        corrs.append(corr)
        times.append('1s')
        
    return np.array(dists), np.array(map_types), np.array(corrs), np.array(times)