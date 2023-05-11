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
from mylib.preprocessing_behav import plot_trajactory_comparison, clean_data, get_meanframe, calc_speed, PolygonDrawer, Circulate_Checking
from mylib.maze_utils3 import position_transform, location_to_idx, occu_time_transform, DLC_Concatenate
from mylib.maze_graph import Father2SonGraph

def uniform_smooth_range(i: int, T: int, window: int):
    """uniform_smooth_range _summary_

    Generation of the range of smoothing.

    Parameters
    ----------
    i : int
        the current frame.
    T : int
        the total number of frames.
    window : int
        the length of window.
    """
    if window%2 == 1:
        left = i - int((window-1)/2) if i - int((window-1)/2) >= 0 else 0
        right = i + int((window-1)/2) if i + int((window-1)/2) < T - 1 else T - 1
        return left, right
    elif window%2 == 0:
        left = i - int(window/2) + 1 if i - int(window/2) + 1 >= 0 else 0
        right = i + int(window/2) if i + int(window/2) < T - 1 else T - 1
        return left, right
    else:
        raise ValueError(f"{window} is invalid value for window.")

def uniform_smooth_speed(speed: np.ndarray, window: int = 30) -> np.ndarray:
    """uniform_smooth_speed _summary_

    Smooth the speed with a uniform window.

    Parameters
    ----------
    speed : np.ndarray
        the speed to be smoothed
    window : int, optional
        smooth window length, by default 30

    Returns
    -------
    np.ndarray
        the smoothed speed
    """
    T = speed.shape[0]
    M = np.zeros((T, T), dtype=np.float64) # Smooth matrix
    
    for i in range(T):
        left, right = uniform_smooth_range(i, T, window=window)
        idx = np.arange(left, right+1)
        M[i, idx] = 1/len(idx)
    
    return np.dot(M, speed)


def read_time_stamp(file_dir: str, key_word: str = 'Time Stamp (ms)') -> np.ndarray:
    f = pd.read_csv(file_dir)
    return np.array(f[key_word])

def dlc_position_generation(dlc: dict, dtype: str = 'prefer', prefer_body_part: str = 'bodypart1'):
    """dlc_position_generation _summary_

    Generation of position from the dlc data

    Parameters
    ----------
    dlc : dict
        dlc coordinates dictionary, e.g.,  {'bodypart1': numpy.array([...]), 'bodypart2': numpy.array([...]), 
                                            'bodypart3': numpy.array([...]), 'objectA': numpy.array([...])}
    dtype : str, optional
        The way to generate the position of mice, by default 'prefer'
        Only {'prefer', 'mass'}
        'prefer' method simply adopts the positions of one point (prefered point) among 
          those bodyparts labeled as the position of a freely moving mouses.
        'mass' method adopts the mass point of all bodyparts labeled as the position of
          a freely moving mouse.
    
    """
    
    if dtype == 'prefer':
        return cp.deepcopy(dlc[prefer_body_part])

    elif dtype == 'mass':
        pos = None
        nkey = 0
        for b in dlc.keys():
            pos = pos + dlc[b] if pos is not None else dlc[b]
            nkey += 1
        return pos/nkey

def RotateTrajectory(pos: np.ndarray, degree: float = 0, maxHeight = 960, maxWidth = 960):
    if degree == 0:
        return pos
    if degree == 180:
        newpos = np.zeros_like(pos)
        newpos[:, 0] = maxWidth - pos[:, 0]
        newpos[:, 1] = maxHeight - pos[:, 1]

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
    date = int(f['date'][i])
    MiceID = int(f['MiceID'][i])
    folder = str(f['recording_folder'][i])
    maze_type = int(f['maze_type'][i])
    behavior_paradigm = str(f['behavior_paradigm'][i])   #" CrossMaze", "ReverseMaze", "DSPMaze"
    session = int(f['session'][i])

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
    
    behav_position_original = dlc_position_generation(dlc, **dlc_args)
    
    trace = {'date':date,'MiceID':MiceID,'paradigm':behavior_paradigm,'session_path':p,'behav_folder':folder, 
             'maze_type':maze_type,'nx':48, 'ny':48, 'body_parts': list(dlc.keys()), 'dlc_position': dlc,
             'behav_position_original': cp.deepcopy(behav_position_original),
             'behav_time_original': cp.deepcopy(behav_time_original)}
    
    behav_positions = cp.deepcopy(trace['behav_position_original'])
    behav_time = cp.deepcopy(trace['behav_time_original'])
        
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
    
    
    # 2. Data cleaning by deleting several frames near the start point and the end point.
    print("    Data cleaning...")
    # data cleaning 1:
    behav_positions, behav_time = clean_data(behav_positions, behav_time, maze_type = maze_type, delete_start=3, delete_end=3)
    # Add NAN value at the cross lap gap to plot the trajactory.
    behav_positions, behav_time = Add_NAN(behav_positions, behav_time, maze_type = maze_type)
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav,
                file_name = 'Trajactory_DeleteWrongData', maze_type = maze_type)
    print("    Figure 3 has done.")
    
    
    # Get a modified behav_time_original for interpolated
    start_time = behav_time[0]
    end_time = behav_time[-1]
    start_index = np.where(behav_time_original == start_time)[0][0]
    end_index = np.where(behav_time_original == end_time)[0][0]
    trace['behav_time_original'] = cp.deepcopy(behav_time_original[start_index:(end_index+1)])
    trace['behav_position_original'] = cp.deepcopy(trace['behav_position_original'][start_index:(end_index+1), :])


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
        
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    ax.imshow(warped_image)
    plot_trajactory(x = processed_pos[:,0], y = processed_pos[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax,
                        file_name = 'TraceOnMeanFrame', color = 'red', linewidth = 1, maze_type = maze_type)
    print("    Figure 5 has done.")

    
    trace['processed_pos'] = cp.deepcopy(processed_pos)

    if cam_degree == 0:
        trace['processed_pos_new'] = cp.deepcopy(processed_pos)
    else:
        trace['processed_pos_new'] = [maxWidth, maxHeight] - processed_pos

    processed_pos_new = cp.deepcopy(trace['processed_pos_new'])
    trace['behav_time'] = cp.deepcopy(behav_time)
        
    # Plot the processed position ---------------------------------------------------------------------------------------------
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    if maze_type in [1,2]:
        DrawMazeProfile(axes = ax, maze_type = maze_type, nx = trace['nx'], linewidth = 2, color = 'black')
    a = position_transform(processed_pos_new)
    plot_trajactory(x = a[:,0], y = a[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax, maze_type = maze_type,
                    file_name = 'TraceOnMeanFrame_WithMazeProfile', color = 'red', linewidth = 1, is_inverty = True)
    print('    Figure 6 has done.')
        
    # Generate behav_nodes
    behav_nodes = location_to_idx(processed_pos_new[:,0], processed_pos_new[:,1], nx = trace['nx'])
    trace['behav_nodes'] = cp.deepcopy(behav_nodes)

    # For maze 1 and maze 2 sessions, a cross-wall correction should be performed. --------------------------------------------
    if maze_type in [1,2]:
        # Correct trajectory cross-wall events
        trace = Circulate_Checking(trace, circulate_time = 5)
            
        # Check the effect of cross-wall correction
        plot_trajactory_comparison(processed_pos_new, trace['correct_pos'], is_position_transform = True, 
                                   save_loc = p_behav, file_name = 'CrossWall-Correction_Trajactory', maze_type = maze_type)
        print('    Figure 7 has done.')
        
        # Transform nodes to x,y location to check the efficiency of correction
        plot_trajactory_comparison(behav_nodes, trace['correct_nodes'], is_node = True, maze_type = maze_type,
                                       save_loc = p_behav, file_name = 'CrossWall-Correction_Nodes')
        print("    Figure 8 has done.")
    
    # For open field, nothing need to do. ------------------------------------------------------------------------------------
    elif maze_type == 0:
        trace['correct_pos'] = cp.deepcopy(trace['processed_pos_new'])
        trace['correct_nodes'] = cp.deepcopy(trace['behav_nodes'])
        trace['correct_time'] = cp.deepcopy(trace['behav_time'])

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
    if maze_type in [1,2]:
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
    
    # Save files
    with open(os.path.join(p,"trace_behav.pkl"), 'wb') as fs:
        pickle.dump(trace, fs)
    
    f.loc[i, 'Trace Behav File'] = os.path.join(p,"trace_behav.pkl")
    f.to_excel(output_loc, index=False)
    
    print("    ",os.path.join(p,"trace_behav.pkl")," has been saved successfully!")
