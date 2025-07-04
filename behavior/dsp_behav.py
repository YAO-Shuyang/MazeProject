# ================================================= DLC Version Code =================================================
# Run all mice function ============================================================================================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
import cv2
import copy as cp
import pickle
from mylib.maze_utils3 import mkdir, plot_trajactory, Delete_NAN, Add_NAN, Clear_Axes, DrawMazeProfile
from mylib.preprocessing_behav import plot_trajactory_comparison, clean_data, get_meanframe, calc_speed
from mylib.preprocessing_behav import PolygonDrawer, uniform_smooth_speed, Circulate_Checking
from mylib.maze_utils3 import position_transform, location_to_idx, occu_time_transform, DLC_Concatenate
from mylib.maze_graph import Father2SonGraph
from mylib.behavior import read_time_stamp, dlc_position_generation


def RotateTrajectory(pos: np.ndarray, degree: float = 0, maxHeight = 960, maxWidth = 960):
    if degree == 0:
        return pos
    if degree == 180:
        newpos = np.zeros_like(pos)
        newpos[:, 0] = maxWidth - pos[:, 0]
        newpos[:, 1] = maxHeight - pos[:, 1]

def delete_interlap_interval(behav_position_original: np.ndarray, behav_time_original: np.ndarray, lap_info: np.ndarray):
    laps = lap_info.shape[0]
    idx = np.concatenate([np.arange(lap_info[i, 0], lap_info[i, 1]+1) for i in range(laps)])
    return behav_position_original[idx, :], behav_time_original[idx]

def run_all_mice_DLC(i: int, f: pd.DataFrame, work_flow: str, speed_sm_args = {'window':30},
                     dlc_args = {'dtype':'mass', 'prefer_body_part': 'bodypart1'},
                     conc_args = {'find_chars': '.csv', 'body_part': ['bodypart1', 'bodypart2', 'bodypart3', 'bodypart4'], 'header':[1,2]},
                     output_loc: str = None,
                     cam_degree: float = 0):
    """
    run_all_mice_DLC _summary_

    Parameters
    ----------
    i : int
        Index of line in the input excel data.
    f : pd.DataFrame
        The input excel sheet.
    work_flow : str
        _description_
    speed_sm_args : dict, optional
        _description_, by default {'window':30}
    dlc_args : dict, optional
        _description_, by default {'dtype':'mass', 'prefer_body_part': 'bodypart1'}
    conc_args : dict, optional
        _description_, by default {'find_chars': '.csv', 'body_part': ['bodypart1', 'bodypart2', 'bodypart3', 'bodypart4'], 'header':[1,2]}
    output_loc : str, optional
        _description_, by default None
    """
    frames_num_tracker = np.zeros(5, np.int64)
    tits = ['raw', 'delet\nNAN', 'clean', 'out\nrange', 'rectify']

    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])   #" CrossMaze", "ReverseMaze", "DSPMaze"
    session = int(f['session'][i])
    try:
        start_from = str(f['Start From'][i])
    except:
        start_from = None
    
    totalpath = work_flow
    p = os.path.join(totalpath, str(MiceID), str(date),"session "+str(session))

    p_behav = os.path.join(p,'behav')
    mkdir(p_behav)

    # Concatenate DLC files
    DLC_Concatenate(folder, **conc_args)

    # read in behav_new.mat
    dlc_coord_path = os.path.join(folder, "dlc_coord.pkl")
    if os.path.exists(dlc_coord_path):
        with open(dlc_coord_path, 'rb') as handle:
            dlc = pickle.load(handle)
    else:
        warnings.warn(f"Didn't find {dlc_coord_path}.")
        return
    
    time_stamp_path = os.path.join(folder, "timeStamps.csv")
    if os.path.exists(time_stamp_path):
        behav_time_original = read_time_stamp(time_stamp_path)
    else:
        warnings.warn(f"Didn't find {time_stamp_path}.")
        return

    lap_info_path = os.path.join(folder, 'frame_labels.pkl')
    if os.path.exists(lap_info_path):
        with open(lap_info_path, 'rb') as handle:
            lap_info = pickle.load(handle)

        nanidx = np.where(np.isnan(lap_info))
        if len(nanidx[0]) != 0:
            warnings.warn(f"lap info contains NAN value! Relabeling please!")
        else:
            lap_info = lap_info.astype(np.int64)
    else:
        warnings.warn(f"Didn't find {lap_info_path}")
        return
    
    behav_position_original = dlc_position_generation(dlc, **dlc_args)
    frames_num_tracker[0] = behav_position_original.shape[0]

    trace = {'date':date,'MiceID':MiceID,'paradigm':behavior_paradigm,'session_path':p,'behav_folder':folder, 
             'maze_type':maze_type,'nx':48, 'ny':48, 'body_parts': list(dlc.keys()), 'dlc_position': dlc,
             'behav_position_original': cp.deepcopy(behav_position_original), 'start_from': start_from,
             'behav_time_original': cp.deepcopy(behav_time_original), 'lap beg time': behav_time_original[lap_info[:, 0]], 
             'lap end time': behav_time_original[lap_info[:, 1]], 'p': p_behav}
    
    behav_positions, behav_time = delete_interlap_interval(behav_position_original, behav_time_original, lap_info)
        
    # Clean data (Delete wrong data, delete NAN value) ------------------------------------------------------------------------
    # 1. Delete NAN values
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav,
                    file_name = 'Trajactory_WithoutAnyProcess', maze_type = maze_type)
    print("    Figure 1 has done.")
    behav_positions, behav_time = Delete_NAN(behav_positions, behav_time)
    # Add NAN value at the cross lap gap to plot the trajactory.
    behav_positions, behav_time = Add_NAN(behav_positions, behav_time, maze_type = maze_type)
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav, maze_type = maze_type,
                file_name = 'Trajactory_DeleteNANOnly')
    print("    Figure 2 has done.")
    
    frames_num_tracker[1] = behav_positions.shape[0]
    
    # 2. Data cleaning by deleting several frames near the start point and the end point.
    print("    Data cleaning...")
    print("Width:", np.nanmax(behav_positions[:,0])-np.nanmin(behav_positions[:,0]), "Height:", np.nanmax(behav_positions[:,1])-np.nanmin(behav_positions[:,1]))
    # data cleaning 1:
    behav_positions, behav_time = clean_data(behav_positions, behav_time, maze_type = maze_type, delete_start=0, delete_end=0, save_loc=p_behav)
    # Add NAN value at the cross lap gap to plot the trajactory.
    behav_positions, behav_time = Add_NAN(behav_positions, behav_time, maze_type = maze_type)
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav,
                file_name = 'Trajactory_DeleteWrongData', maze_type = maze_type)
    print("    Figure 3 has done.")

    frames_num_tracker[2] = behav_positions.shape[0]

    # Activate the affine transformation gui ---------------------------------------------------------------------------
    # define behavior x, y, roi for correction
    
    # Read the video to extract background
    video_name = os.path.join(folder,'0.avi')
    mean_frame = get_meanframe(video_name)

    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    ax.imshow(mean_frame.astype(np.int64))
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax,
                    file_name = 'TraceOnMeanFrame_Raw', color = 'red', linewidth = 1, maze_type = maze_type)
    
    behav_positions, behav_time = Delete_NAN(behav_positions, behav_time)
    print("    Figure 4 has done.")
    
    equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame))
        
    # to perform a four point transform used package opencv.
    # There'll be a window popping up and to select a four-edge shape.
    maxHeight, maxWidth = 960 - 0.00001, 960-0.00001
    pd = PolygonDrawer(equ_meanframe, behav_positions, maxHeight = 960, maxWidth = 960)
    warped_image, warped_positions, M  = pd.run()
    cv2.imwrite(os.path.join(p_behav,"polygon.png"), warped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    warped_positions, behav_time = Add_NAN(warped_positions, behav_time, maze_type = maze_type)
    print("    Polygon = %s" % pd.points)
    
    # Get processed position args. -------------------------------------------------------------------------------------------
    processed_pos = warped_positions
    processed_pos, behav_time = Delete_NAN(processed_pos, behav_time)
    # A tolerable_range for out of range. If out of range, set the data point as NAN value.
    #tolerable_range = 40 # 5 cm
    processed_pos[np.where((processed_pos[:,0] < 0)|(processed_pos[:,0] > maxWidth))[0], 0] = np.nan
    processed_pos[np.where((processed_pos[:,1] < 0)|(processed_pos[:,1] > maxHeight))[0], 1] = np.nan

    processed_pos, behav_time = Delete_NAN(processed_pos, behav_time)
    processed_pos, behav_time = Add_NAN(processed_pos, behav_time, maze_type = maze_type)    
    frames_num_tracker[3] = processed_pos.shape[0]
        
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    ax.imshow(warped_image)
    plot_trajactory(x = processed_pos[:,0], y = processed_pos[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax,
                        file_name = 'TraceOnMeanFrame', color = 'red', linewidth = 1, maze_type = maze_type)
    print("    Figure 5 has done.")

    
    trace['processed_pos'] = cp.deepcopy(processed_pos)

    if cam_degree == 0:
        trace['processed_pos_new'] = cp.deepcopy(processed_pos)
    elif cam_degree == 180:
        trace['processed_pos_new'] = [maxWidth, maxHeight] - processed_pos
    else:
        raise ValueError(f"Invalid value {cam_degree} for camera degree.")

    processed_pos_new = cp.deepcopy(trace['processed_pos_new'])
    trace['behav_time'] = cp.deepcopy(behav_time)
        
    # Plot the processed position ---------------------------------------------------------------------------------------------
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    if maze_type in [1,2,3]:
        DrawMazeProfile(axes = ax, maze_type = maze_type, nx = trace['nx'], linewidth = 2, color = 'black')
    a = position_transform(processed_pos_new)
    plot_trajactory(x = a[:,0], y = a[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax, maze_type = maze_type,
                    file_name = 'TraceOnMeanFrame_WithMazeProfile', color = 'red', linewidth = 1, is_inverty = True)
    print('    Figure 6 has done.')
        
    # Generate behav_nodes
    behav_nodes = location_to_idx(processed_pos_new[:,0], processed_pos_new[:,1], nx = trace['nx'])
    trace['behav_nodes'] = cp.deepcopy(behav_nodes)


    # For maze 1, maze 2 and hairpin maze sessions, a cross-wall correction should be performed. --------------------------------------------
    if maze_type in [1, 2]:
        # Correct trajectory cross-wall events
        trace = Circulate_Checking(trace, circulate_time = 5)
            
    # For open field, nothing need to do. ------------------------------------------------------------------------------------
    else:
        trace['correct_pos'] = cp.deepcopy(trace['processed_pos_new'])
        trace['correct_nodes'] = cp.deepcopy(trace['behav_nodes'])
        trace['correct_time'] = cp.deepcopy(trace['behav_time'])
        frames_num_tracker[3] = trace['correct_nodes'].shape[0]
        
    # Check the effect of cross-wall correction
    plot_trajactory_comparison(processed_pos_new, trace['correct_pos'], is_position_transform = True, 
                                   save_loc = p_behav, file_name = 'CrossWall-Correction_Trajactory', maze_type = maze_type)
    print('    Figure 7 has done.')
        
    # Transform nodes to x,y location to check the efficiency of correction
    plot_trajactory_comparison(behav_nodes, trace['correct_nodes'], is_node = True, maze_type = maze_type,
                                       save_loc = p_behav, file_name = 'CrossWall-Correction_Nodes')
    print("    Figure 8 has done.")
    frames_num_tracker[4] = trace['correct_nodes'].shape[0]
    # =======================================================================================================================
    # calculating ratemap ----------------------------------------------------------------------------------------------------
    pos, correct_time, behav_nodes = Delete_NAN(trace['correct_pos'], trace['correct_time'], trace['correct_nodes'])
    dt = np.append(np.ediff1d(behav_time),33)
    dt[np.where(dt > 100)[0]] = 100
    
    occu_time = np.zeros(2304, dtype = np.float64)
    print("    calculating occupation time:")
    for i in tqdm(range(2304)):
        idx = np.where(behav_nodes == i+1)[0]
        occu_time[i] = np.nan if len(idx) == 0 else np.nansum(dt[idx])

    trace['occu_time'] = occu_time
    print(np.nanmax(occu_time))
    
    # Plot bahavioral ratemap
    print("    Draw behavioral ratemap...")
    plt.figure(figsize = (8,6))
    ax = Clear_Axes(plt.axes())
    if maze_type in [1,2,3]:
        DrawMazeProfile(nx = 48, maze_type = trace['maze_type'], linewidth = 2, color = 'yellow', axes = ax)
    im = ax.imshow(np.reshape(occu_time/1000,[48,48]), vmax = 20, cmap = 'jet')
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label('occupation time: s', fontsize = 16)
    plt.savefig(os.path.join(p_behav,'behav_ratemap.png'), dpi=600)
    plt.savefig(os.path.join(p_behav,'behav_ratemap.svg'), dpi=600)
    plt.close()
    print("    Figure behavioral ratemap has done.")
    
    # Plot speed distribution figure ------------------------------------------------------------------------------------------
    # Delete All NAN Value
    trace['processed_pos_new'], trace['behav_time'], trace['behav_nodes'] = Delete_NAN(trace['processed_pos_new'], 
                                                                          trace['behav_time'], trace['behav_nodes'])
    trace['correct_pos'], trace['correct_time'], trace['correct_nodes'] = Delete_NAN(trace['correct_pos'], 
                                                                          trace['correct_time'], trace['correct_nodes'])
    # behav_speed
    behav_speed = calc_speed(behav_positions = trace['correct_pos']/10, behav_time = trace['correct_time'])
    smooth_speed = uniform_smooth_speed(behav_speed, **speed_sm_args)
    
    plt.figure(figsize = (8,6))
    MAX_X = (np.nanmax(behav_speed) // 5 + 1) * 5
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks = True, ifyticks = True)
    ax.hist(behav_speed, bins = 50)
    ax.set_xlabel('Speed (cm/s)', fontsize = 16)
    ax.set_ylabel('Counts', fontsize = 16)
    plt.savefig(os.path.join(p_behav,'speed_distb_raw.png'),dpi=600)
    plt.savefig(os.path.join(p_behav,'speed_distb_raw.svg'),dpi=600)
    plt.close()
    print('    Figure raw speed distribution has done.')
    
    plt.figure(figsize = (8,6))
    MAX_X = (np.nanmax(smooth_speed) // 5 + 1) * 5
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks = True, ifyticks = True)
    ax.hist(smooth_speed, bins = 50)
    ax.set_xlabel('Speed (cm/s)', fontsize = 16)
    ax.set_ylabel('Counts', fontsize = 16)
    plt.savefig(os.path.join(p_behav,'speed_distb_smooth.png'),dpi=600)
    plt.savefig(os.path.join(p_behav,'speed_distb_smooth.svg'),dpi=600)
    plt.close()
    print('    Figure smoothed speed distribution has done.')    

    trace['correct_speed'] = behav_speed
    trace['smooth_speed'] = smooth_speed
    
    # Qualification control curve: step by step
    plt.figure(figsize = (8,4))
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks = True, ifyticks = True)
    ax.plot(tits, frames_num_tracker, marker = '^', markerfacecolor = 'black',markeredgecolor = 'black')
    for i in range(5):
        ax.text(i, 100, str(frames_num_tracker[i]), va = 'center')
    ax.set_ylabel('Frame counts')
    ax.axis([-0.5, 4.5, 0, np.max(frames_num_tracker)+1000])
    plt.savefig(os.path.join(p_behav,'qualify-control.png'),dpi=600)
    plt.savefig(os.path.join(p_behav,'qualify-control.svg'),dpi=600)
    plt.close()    


    # Save files
    with open(os.path.join(p,"trace_behav.pkl"), 'wb') as fs:
        pickle.dump(trace, fs)
    
    f.loc[i, 'Trace Behav File'] = os.path.join(p,"trace_behav.pkl")
    f.to_excel(output_loc, index=False)
    
    print("    ",os.path.join(p,"trace_behav.pkl")," has been saved successfully!")

def concat_trace_behav(oripath1: str, oripath2: str, save_dir: None = None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.dirname(oripath1))
    
    with open(oripath1, 'rb') as handle:
        trace1 = pickle.load(handle)

    with open(oripath2, 'rb') as handle:
        trace2 = pickle.load(handle)

    add_time = trace1['behav_time_original'][-1] + 100000

    trace = {
        'date': trace1['date'],
        'MiceID': trace1['MiceID'],
        'paradigm': trace1['paradigm'],
        'session_path': save_dir, 'p': save_dir,
        'former_path': [trace1['session_path'], trace2['session_path']],
        'maze_type': 1,
        'nx': 48, 'ny': 48,
        'body_parts': trace1['body_parts'],
        'dlc_position': {k: np.concatenate([trace1['dlc_position'][k], trace2['dlc_position'][k]], axis=0) for k in trace1['dlc_position'].keys()},
        'behav_position_original': np.concatenate([trace1['behav_position_original'], trace2['behav_position_original']], axis=0),
        'behav_time_original': np.concatenate([trace1['behav_time_original'], trace2['behav_time_original'] + add_time]),
        'lap beg time': np.concatenate([trace1['lap beg time'], trace2['lap beg time'] + add_time]),
        'lap end time': np.concatenate([trace1['lap end time'], trace2['lap end time'] + add_time]),
        'processed_pos': np.concatenate([trace1['processed_pos'], trace2['processed_pos']], axis=0),
        'processed_pos_new': np.concatenate([trace1['processed_pos_new'], trace2['processed_pos_new']], axis=0),
        'behav_time': np.concatenate([trace1['behav_time'], trace2['behav_time'] + add_time]),
        'behav_nodes': np.concatenate([trace1['behav_nodes'], trace2['behav_nodes']]),
        'correct_pos': np.concatenate([trace1['correct_pos'], trace2['correct_pos']], axis=0),
        'correct_nodes': np.concatenate([trace1['correct_nodes'], trace2['correct_nodes']]),
        'correct_time': np.concatenate([trace1['correct_time'], trace2['correct_time'] + add_time]),
        'occu_time': trace1['occu_time'] + trace2['occu_time'],
        'correct_speed': np.concatenate([trace1['correct_speed'], trace2['correct_speed']]),
        'smooth_speed': np.concatenate([trace1['smooth_speed'], trace2['smooth_speed']])
    }
    
    with open(os.path.join(save_dir, 'trace_behav.pkl'), 'wb') as handle:
        pickle.dump(trace, handle)
    
    print("    ",os.path.join(save_dir, 'trace_behav.pkl')," has been saved successfully!")

if __name__ == '__main__':
    from mylib.local_path import f2_ori, f2
    """
    f = pd.read_excel(r"G:\YSY\Dsp_maze\dsp_maze_paradigm.xlsx", sheet_name='behavior')
    work_flow = "G:\YSY\Dsp_maze"
    for i in range(18, len(f)):

        print(i, int(f['MiceID'][i]), int(f['date'][i]), int(f['session'][i]))
        run_all_mice_DLC(i, f, work_flow=work_flow, output_loc=r"G:\YSY\Dsp_maze\dsp_maze_paradigm_output.xlsx",
                     dlc_args={'dtype':'prefer', 'prefer_body_part': 'bodypart2'},
                     conc_args = {'find_chars': '1030000', 'body_part': ['bodypart1', 'bodypart2', 'bodypart3', 'objectA'], 
                     'header':[1,2]}, cam_degree=int(f['cam_degree'][i]))
    
        print("Done.", end='\n\n\n')
    """
    """
    for i in range(int(len(f2_ori) / 2)):
        oripath1 = f2_ori['Trace Behav File'][i*2]
        oripath2 = f2_ori['Trace Behav File'][i*2+1]

        concat_trace_behav(oripath1, oripath2)
    """
    
    for i in range(len(f2)):
        with open(f2['Trace Behav File'][i], 'rb') as handle:
            trace_behav = pickle.load(handle)
            
        with open(f2['Trace File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        trace['dlc_position'] = trace_behav['dlc_position']
        
        with open(f2['Trace File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
            
        print(i, f2['MiceID'][i], [trace['dlc_position'][k].shape for k in trace['dlc_position'].keys()], trace['behav_position_original'].shape, trace['behav_time_original'].shape)