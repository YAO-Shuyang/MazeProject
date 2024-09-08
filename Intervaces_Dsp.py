from mylib.preprocessing_ms import *
from mylib.preprocessing_behav import *
from mylib.local_path import *
from scipy.stats import poisson, norm, nbinom, gamma, kstest, ks_2samp, anderson, anderson_ksamp 
from scipy.stats import linregress, pearsonr, chisquare, ttest_1samp, ttest_ind
from scipy.stats import ttest_rel, levene, spearmanr
from mylib.stats.ks import poisson_kstest, normal_discrete_kstest, nbinom_kstest
from mylib.behavior.correct_rate import lapwise_behavioral_score
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
    routes = classify_lap(spike_nodes_transform(trace['correct_nodes'], 12), beg_idx, trace['start_from'])
    routes[routes == 4] = 0
    
    if trace['start_from'] == 'correct':
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    else:
        routes[np.where(routes != 0)[0]] += 3
        routes_id = np.array(["Route "+str(i+1) for i in routes])
        routes_used = routes
    
    correct_rates, pass_nums, err_nums, pureguess = [], [], [], []
    
    for i in np.unique(routes_used):
        idx = np.where(routes_used == i)[0]
        
        err_num, pass_num, std_err = calc_behavioral_score_dsp(
            route=i,
            behav_nodes=np.concatenate([behav_nodes[beg_idx[j]:end_idx[j]+1] for j in idx]),
            behav_time=np.concatenate([behav_time[beg_idx[j]:end_idx[j]+1] for j in idx])
        )
        
        correct_rates.append(1-err_num/pass_num)
        err_nums.append(err_num)
        pass_nums.append(pass_num)
        pureguess.append(1-std_err)
    
    return np.unique(routes_id), np.array(correct_rates, np.float64), np.array(pass_nums, np.int64), np.array(err_nums, np.int64), np.array(pureguess, np.float64)

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
        check_variable = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes']
    )
    
    segment_pieces = []
    segment_pvc = []
    groups = []
    routes = []
    seg_temp = np.concatenate([[0], segs])
    
    D = GetDMatrices(1, 48)
    
    seg_temp = np.array([seg_temp[0], seg_temp[2], seg_temp[4], seg_temp[6], seg_temp[1], seg_temp[3], seg_temp[5], seg_temp[7]])
    for i in [0, 4, 5, 9]:
        for j in range(10):
            if i == j:
                continue
            
            if j in [0, 4, 5, 9]:
                if j <= i:
                    continue
            
            bins = get_son_area(np.intersect1d(CP_DSP[trace[f'node {i}']['Route']], CP_DSP[trace[f'node {j}']['Route']]))-1
            
            pc_idx = np.where(
                (trace[f'node {i}']['is_placecell'] == 1) |
                (trace[f'node {j}']['is_placecell'] == 1)
            )[0]
            
            pvc = np.zeros(bins.shape[0], np.float64)
            for k in range(bins.shape[0]):
                pvc[k], _ = pearsonr(
                    trace[f'node {i}']['smooth_map_all'][pc_idx, bins[k]],
                    trace[f'node {j}']['smooth_map_all'][pc_idx, bins[k]]
                )
                
            dist = D[bins, 0]
            dist = dist / (np.max(dist) + 0.0001) *111
            dist = (dist // 1).astype(np.int64)
            
            pvc_norm = np.zeros(111)
            for k in range(111):
                pvc_norm[k] = np.nanmean(pvc[dist == k])
            
            segment_pieces.append(np.arange(0, 111))
            segment_pvc.append(pvc_norm)
            groups.append(np.repeat(f'{i}-{j}', 111))
            routes.append(np.repeat(trace[f'node {j}']['Route'], 111))
    
    return np.concatenate(segment_pieces), np.concatenate(segment_pvc), np.concatenate(groups), np.concatenate(routes)

def SegmentedCorrelationAcrossRoutes_Egocentric_DSP_Interface(trace: dict, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segments', 'Mean PVC', 'Compare Groups', 'Routes', 'Control For Route']
    )
    segment_pieces = []
    segment_pvc = []
    groups = []
    routes = []
    control_for_route = []
    seg_temp = np.concatenate([[0], segs])
    
    D = GetDMatrices(1, 48)
    
    seg_temp = np.array([seg_temp[0], seg_temp[2], seg_temp[4], seg_temp[6], seg_temp[1], seg_temp[3], seg_temp[5], seg_temp[7]])
    for i in [0, 4, 5, 9]:
        for j in [1, 2, 3, 6, 7, 8]:
            
            son_bins1 = get_son_area(CP_DSP[trace[f'node {j}']['Route']])
            son_bins2 = get_son_area(CP_DSP[0])
            
            D1 = D[son_bins1-1, SP_DSP[trace[f'node {j}']['Route']]-1]
            D2 = D[son_bins2-1, SP_DSP[0]-1]
            
            idx1 = np.argsort(D1)
            idx2 = np.argsort(D2)[:idx1.shape[0]]
            
            pc_idx = np.where(
                (trace[f'node {i}']['is_placecell'] == 1) |
                (trace[f'node {j}']['is_placecell'] == 1)
            )[0]
            
            PVC = np.zeros(idx1.shape[0], np.float64)
            for k in range(idx1.shape[0]):
                PVC[k], _ = pearsonr(
                    trace[f'node {j}']['smooth_map_all'][pc_idx, son_bins1[idx1[k]]-1],
                    trace[f'node {i}']['smooth_map_all'][pc_idx, son_bins2[idx2[k]]-1]
                )
                
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
            routes.append(np.repeat(trace[f'node {j}']['Route'], n_len)) 
            control_for_route.append(np.repeat(trace[f'node {j}']['Route'], n_len))    
            
            # Control
            n_targ = np.array([0, 4, 5, 9])
            for m in n_targ:
                if m == i:
                    continue
                
                
                PVC = np.zeros(idx1.shape[0], np.float64)
                for k in range(idx1.shape[0]):
                    PVC[k], _ = pearsonr(
                        trace[f'node {i}']['smooth_map_all'][pc_idx, son_bins1[idx1[k]]-1],
                        trace[f'node {m}']['smooth_map_all'][pc_idx, son_bins2[idx2[k]]-1]
                    )
                        
                n_len = CP_DSP[trace[f'node {j}']['Route']].shape[0]
                dist = D1[idx1]
                dist = dist / (np.max(dist) + 0.0001) * n_len
                dist = (dist // 1).astype(np.int64)
                    
                pvc_norm = np.zeros(n_len)
                for k in range(n_len):
                    pvc_norm[k] = np.nanmean(PVC[dist == k])
                    
                segment_pieces.append(np.arange(0, n_len))
                segment_pvc.append(pvc_norm)
                groups.append(np.repeat(f'{i}-{m}', n_len))
                routes.append(np.repeat(trace[f'node {m}']['Route'], n_len))
                control_for_route.append(np.repeat(trace[f'node {j}']['Route'], n_len)) 
 
    return np.concatenate(segment_pieces), np.concatenate(segment_pvc), np.concatenate(groups), np.concatenate(routes), np.concatenate(control_for_route)

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
    
    field_reg = trace['field_reg']
    field_segs = trace['field_segs']

    prop = np.ones(6)
    print(np.unique(field_segs))
    for seg in range(1, 7):
        field_idx = np.where(field_segs == seg+1)[0]
        
        allo_field_idx = np.where(
            (field_segs == seg+1) & 
            (
                (field_reg[0, :] == 1) |
                (field_reg[4, :] == 1) |
                (field_reg[5, :] == 1) |
                (field_reg[9, :] == 1)
            ) &
            (np.sum(field_reg[1:4, :], axis=0)+np.sum(field_reg[6:9, :], axis=0) == seg)
        )[0]
        
        prop[seg-1] = allo_field_idx.shape[0] / field_idx.shape[0]
        
    return np.arange(1, 7), prop

def FieldStateSwitchWithSegment_DSP_Interface(trace, variable_names = None):
    VariablesInputErrorCheck(
        input_variable = variable_names, 
        check_variable = ['Segment', 'Proportion', 'Category']
    )
    
    field_reg = trace['field_reg']
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
            (field_reg[4, :] == 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[0, :] == 1) &
            (field_reg[4, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[0, :] == 1) &
            (field_reg[4, :] == 1)
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
            (field_reg[9, :] == 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[5, :] == 1) &
            (field_reg[9, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[5, :] == 1) &
            (field_reg[9, :] == 1)
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
            (field_reg[5, :] == 1)
        )[0]
        
        disped_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[4, :] == 1) &
            (field_reg[5, :] == 0)
        )[0]
        
        retain_field_idx = np.where(
            (field_segs == seg+1) &
            (field_reg[4, :] == 1) &
            (field_reg[5, :] == 1)
        )[0]
        
        prop[(seg-1) + 42] = formed_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 49] = disped_field_idx.shape[0] / field_idx.shape[0]
        prop[(seg-1) + 56] = retain_field_idx.shape[0] / field_idx.shape[0]

    return segments, prop, category