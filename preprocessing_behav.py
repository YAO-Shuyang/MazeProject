import h5py
import pandas as pd
from scipy.io import loadmat
from mylib.maze_utils3 import *

maxWidth = 960- 0.00001
maxHeight = 960- 0.00001
nbins = 48
coords_range = [[0,maxWidth +0.01],[0, maxHeight+0.01]]
nx = 48
ny = 48

# DLC processing code ----------------------------------------------------------------------------------------------------------------------------------------------------
# If process original video file by DLC code, we will get several processed files (actually, 8 files) for each video, consisting of 3 CSV files(.csv), 3 H5 files (.h5)
# and 2 pickle files(.pickle).

# behav data process ------------------------------------------------------------------------------------------------------------------
def get_meanframe(video_name):
    cap = cv2.VideoCapture(video_name)
    length = np.int64(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(length):    # Capture frame-by-frame
        ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
        if i == 0: # initialize mean frame
            mean_frame = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # Our operations on the frame come here    
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/length
        # img = frame/length
        mean_frame = mean_frame + img
    
    return mean_frame

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts, maxHeight = 960, maxWidth = 960):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# maxHeight = max(int(heightA), int(heightB))
	
    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# warped_positions = cv2.perspectiveTransform(np.array([ori_positions]) , M)
	# return the warped image
	return warped_image, M

FINAL_LINE_COLOR = (255, 100, 0)
WORKING_LINE_COLOR = (127, 127, 127) 

class PolygonDrawer(object):
    def __init__(self, equ_meanframe, ori_positions, maxHeight = 960, maxWidth = 960):
        self.window_name = "Original: select 4 maze corners" # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.equ_meanframe = equ_meanframe
        self.ori_positions = ori_positions
        self.maxHeight = maxHeight
        self.maxWidth = maxWidth


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every smouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("    Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("    Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.equ_meanframe)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self.equ_meanframe
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            cv2.polylines(canvas,  np.int32([self.ori_positions]), False, FINAL_LINE_COLOR, 1)

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                # And  also show what the current segment would look like
                # cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = self.equ_meanframe

        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]),True, FINAL_LINE_COLOR, thickness = 5)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()
        
        # Four points transform
        warped_image, M = four_point_transform(self.equ_meanframe, np.asarray(self.points), maxHeight = self.maxHeight, maxWidth = self.maxWidth)
        cv2.imshow("Processed Maze", warped_image)
        warped_positions = cv2.perspectiveTransform(np.array([self.ori_positions]) , M)[0]
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("Processed Maze")
       
        return warped_image, warped_positions, M

def transform_bin(bin_numbers, nx = 48):
    # rotate bin by 90 degree
    x = nx + 1 - bin_numbers[1,:]
    y = bin_numbers[0,:]
    return np.array([x,y])

# ---------------------------------------------------- Data Correction --------------------------------------------------------------
# the version 3.0
# Correct cross wall events.
def CrossWallCheck(D, dt = 0.033, check_node =  1, targ_node = 2304):
    l = D[int(check_node)-1, int(targ_node)-1]*2
    v = l/dt
    if v >= 200 and l >= 16:
        return 1
    else:
        return 0

def PutativePos(D, check_pos, targ_pos, check_node = 1, dt = 0.033, step = 10):
    x, y = check_pos - targ_pos
    dx = x / step
    dy = y / step

    if np.sqrt(dx**2 + dy**2) >= 8:
        return [np.nan,np.nan], np.nan

    for i in range(step):
        targ_pos[0] += dx
        targ_pos[1] += dy
        targ_node = location_to_idx(targ_pos[0],targ_pos[1], nx = 48)
        if CrossWallCheck(D, dt, check_node, targ_node) == 0:
            break
    
    return targ_pos, targ_node

def TrajectoryCorrection(trace):
    if trace['maze_type'] == 0:
        print("    Warning! Open field do not need to correct cross-wall data point!")
        return trace
    
    if trace['nx'] != 48:
        print("   Warning! only nx = 48 are valid value! Report by CrossWallCorrection()!")
        return trace   

    print("      TraceCorrect initiate...")
    #processed_pos_new = trace['processed_pos_new']
    if 'correct_pos' in trace.keys():
        pos_new, behav_time, behav_nodes = Delete_NAN(trace['correct_pos'], trace['correct_time'], trace['correct_nodes'])
    else:
        pos_new, behav_time, behav_nodes = Delete_NAN(trace['processed_pos_new'], trace['behav_time'], trace['behav_nodes'])

    maze_type = trace['maze_type']
    behav_nodes = location_to_idx(pos_new[:,0], pos_new[:,1], nx = trace['nx'])

    D = D_Matrice[maze_type]

    for k in tqdm(range(behav_time.shape[0]-1)):
        if np.isnan(pos_new[k+1,0]) or np.isnan(pos_new[k+1,1]) or np.isnan(pos_new[k,0]) or np.isnan(pos_new[k,1]):
            continue
        check_pos = pos_new[k,:]
        check_node = behav_nodes[k]
        targ_pos = pos_new[k+1,:]
        targ_node = behav_nodes[k+1]
        dt = (behav_time[k+1] - behav_time[k]) / 1000
        
        if dt > 4: # interim of laps
            continue

        if CrossWallCheck(D, dt = dt, check_node = check_node, targ_node = targ_node) == 1:
            # A putative cross wall event.
            # Correct it and set a putative position.
            pos_new[k+1,:], behav_nodes[k+1] = PutativePos(D, check_pos, targ_pos, check_node = check_node, dt = dt, step = 10)
    
    # Delete NAN Value
    pos_new, behav_time, behav_nodes = Delete_NAN(pos_new, behav_time, behav_nodes)
    # Add cross-lap NAN gap
    pos_new, behav_time, behav_nodes = Add_NAN(pos_new, behav_time, behav_nodes, maze_type = trace['maze_type'])
    
    #print("    Shape of correct_:",pos_new.shape, behav_time.shape, behav_nodes.shape)
    #print("    Shape of behav_:",trace['processed_pos_new'].shape,trace['behav_time'].shape,trace['behav_nodes'].shape)
    trace['correct_pos'] = pos_new
    trace['correct_nodes'] = behav_nodes
    trace['correct_time'] = behav_time
    return trace

# Check until There's no cross wall event are detected.
def Circulate_Checking(trace, circulate_time = 4):
    if 'correct_time' in trace.keys():
        length = trace['correct_time'].shape[0]
    else:
        length = trace['behav_time'].shape[0]

    is_stop = 1
    while is_stop < circulate_time:
        trace = TrajectoryCorrection(trace)
        len_temp = trace['correct_time'].shape[0]
        if len_temp == length:
            break
        else:
            length = len_temp
            is_stop += 1
    
    print('Cross-Wall events checking for '+str(is_stop)+' times and the final length of frames is: ',length)
    return trace

def TrajectoryInterpolated(trace, P = None):
    if 'correct_pos' not in trace.keys():
        print("    Warning! you should run Cross-Wall correction function (TrajactoryCorrection) first!")
        return trace
    pos, behav_time, behav_nodes = Delete_NAN(trace['correct_pos'], trace['correct_time'], trace['correct_nodes'])
    behav_time_original = trace['behav_time_original']
    interpolated_pos = cp.deepcopy(pos)
    interpolated_nodes = cp.deepcopy(behav_nodes)
    interpolated_time = cp.deepcopy(behav_time)
    
    for k in tqdm(range(behav_time.shape[0]-1)):
        if np.isnan(behav_nodes[k]) or np.isnan(behav_nodes[k+1]):
            print("Warning! behav_nodes["+str(k)+"] is nan.")
            continue
        beg_time = behav_time[k]
        end_time = behav_time[k+1]
        beg_node = int(behav_nodes[k])
        end_node = int(behav_nodes[k+1])
        
        if P is None:
            print("ERROR! Args P is lost!")
            return trace

        if end_time - beg_time >= 4000 or P[beg_node-1, end_node-1,0] / (end_time - beg_time) * 1000 >= 200:
            continue
            
        beg_idx = np.where(behav_time_original == beg_time)[0]
        end_idx = np.where(behav_time_original == end_time)[0]
        
        if len(beg_idx) == 0 or len(end_idx) == 0:
            print("WARNING! beg_idx, end_idx meet an empty error! Report by TrajectoryInterpolated")
            break
        
        if beg_idx[0] + 1 == end_idx[0] or beg_node == end_node:
            continue

        path = P[beg_node-1, end_node-1, np.where(P[beg_node-1,end_node-1,:] != 0)[0]]
        dp = end_idx[0] - beg_idx[0] - 1
        interpolated_idx = np.linspace(0,len(path)-1, dp + 2) // 1
        interpolated_idx = interpolated_idx.astype(np.int64)
        insert_nodes = path[interpolated_idx[1:-1]]
        insert_pos = np.zeros((insert_nodes.shape[0],2), dtype = np.float64)
        insert_pos[:,0], insert_pos[:,1] = idx_to_loc(insert_nodes, nx = trace['nx'], ny = trace['nx'])
        
        idx = np.where(interpolated_time == beg_time)[0][0]
        
        interpolated_nodes = np.insert(interpolated_nodes, idx+1, insert_nodes)
        interpolated_pos = np.insert(interpolated_pos, idx+1, insert_pos, axis = 0)
        interpolated_time = np.insert(interpolated_time, idx+1, behav_time_original[beg_idx[0]+1:end_idx[0]])
    
    interpolated_pos, interpolated_time, interpolated_nodes= Add_NAN(interpolated_pos, interpolated_time, interpolated_nodes, 
                                                                     maze_type = trace['maze_type'])
    #print('    Shape:',interpolated_nodes.shape, interpolated_pos.shape, interpolated_time.shape)
    trace['interpolated_pos'] = interpolated_pos
    trace['interpolated_nodes'] = interpolated_nodes
    trace['interpolated_time'] = interpolated_time
    return trace

def plot_trajactory_comparison(pos_bef = None, pos_aft = None, save_loc = None, file_name = None, is_show = False, 
                               is_position_transform = False, is_node = False, maze_type = 1):
    
    # if the data is behav_node instead of position, transform them into x,y
    if is_node == True:
        x_bef, y_bef = idx_to_loc(pos_bef, nx = 48, ny = 48)
        x_aft, y_aft = idx_to_loc(pos_aft, nx = 48, ny = 48)
    # if position ranges from (0,960)(0,960), transform them into (0,48)(0,48)
    else:
        if is_position_transform == True:
            pos_bef = position_transform(processed_pos = pos_bef)
            pos_aft = position_transform(processed_pos = pos_aft)
            
        x_bef, y_bef = pos_bef[:,0], pos_bef[:,1]
        x_aft, y_aft = pos_aft[:,0], pos_aft[:,1]
    
    plt.figure(figsize=(6,6))
    ax = Clear_Axes(plt.axes())
    DrawMazeProfile(maze_type = maze_type, nx = 48, axes = ax, linewidth = 2, color = 'black')
    ax = plot_trajactory(x_bef, y_bef, is_ExistAxes = True, ax = ax, is_GetAxes = True,
                    color = 'gray', linewidth = 2, is_inverty = True)
    plot_trajactory(x_aft, y_aft, is_ExistAxes = True, ax = ax, save_loc = save_loc, is_show = is_show,
                        file_name = file_name, color = 'red', linewidth = 2)    

# Clean data(during laps)
def clean_data(behav_positions = None, behav_time = None, maze_type = 1, v_thre: float = 200.,
               delete_start = 20, delete_end = 20):
    if behav_positions is None or behav_time is None:
        print("    Wrong! Both behav_positions and behav_time are required!")
        return 
    
    # delete first and last 30 frames (~1s), when the mouse is not in the maze
    # should replace to exact starting time later!
    delete_start = delete_start
    delete_end = delete_end
    le = behav_time.shape[0]
    behav_positions = np.delete(behav_positions, range(le-1,le-1-delete_start,-1),0)
    behav_time = np.delete(behav_time, range(le-1,le-1-delete_start,-1))
    behav_positions = np.delete(behav_positions, range(0,delete_start),0)
    behav_time = np.delete(behav_time, range(0,delete_start))

    # delete laps interval
    behav_time_tmp = behav_time
    behav_positions_tmp = behav_positions

    k = behav_time.shape[0]-1
    while k > 0:
        # if the interval time is more than 4000ms, we set it as Arival_time.
        if behav_time[k]-behav_time[k-1] > 4000:
            # delete first 30 and last 30 frames of the
            behav_time_tmp = np.delete(behav_time_tmp, range(k-delete_end,k+delete_end))
            behav_positions_tmp = np.delete(behav_positions_tmp, range(k-delete_end,k+delete_end),0)
            k = k - delete_end - 1
            continue
            
        else:
            k -= 1

    behav_positions, behav_time = Delete_NAN(behav_positions_tmp, behav_time_tmp)
    
    behav_positions_tmp = cp.deepcopy(behav_positions)
    behav_time_tmp = cp.deepcopy(behav_time)
    # Delete abnormal points (instantly shift a long distance away.)
    curr_f, next_f = 0, 1  # current frame, next 1/more frame(s)
    while curr_f < behav_time_tmp.shape[0] - 1 and next_f < behav_time_tmp.shape[0]:
        if np.sqrt((behav_positions_tmp[next_f, 0] - behav_positions_tmp[curr_f, 0])**2 + (behav_positions_tmp[next_f, 1] - behav_positions[curr_f, 1])**2)/(behav_time[next_f] - behav_time[curr_f]) * 1000 < v_thre:
            curr_f = next_f
            next_f += 1
        else:
            behav_positions_tmp[next_f] = [np.nan, np.nan]
            next_f += 1
        
    behav_positions, behav_time = Delete_NAN(behav_positions_tmp, behav_time_tmp)

    if maze_type != 0:
        behav_positions, behav_time = Add_NAN(behav_positions, behav_time, maze_type = maze_type)
    
    return  behav_positions, behav_time

# -------------------------------------------------------------------------------------------------------------------------------------

with open(r'F:\YSY\Simulation_pc\decoder_DMatrix.pkl', 'rb') as handle:
    D_Matrice = pickle.load(handle)

def P_Matrice():
    with open(r'F:\YSY\Simulation_pc\P_Matrix-Maze1.pkl', 'rb') as handle:
        P1 = pickle.load(handle)
    print("    Path Matrix P1 is loaded successful! ")

    with open(r'F:\YSY\Simulation_pc\P_Matrix-Maze1.pkl', 'rb') as handle:
        P2 = pickle.load(handle)
    print("    Path Matrix P1 is loaded successful! ")
    return P1, P2

def LocomotionDirection(check_node = 1, targ_node = 1, maze_type = 1, nx = 48):
    if nx == 12:
        D = D_Matrice[6+maze_type] / nx * 12
    elif nx == 24:
        D = D_Matrice[3+maze_type] / nx * 12
    elif nx == 48:
        D = D_Matrice[maze_type] / nx * 12
    else:
        print("self.res value error! Report by mylib.maze_utils3.LocomotionDirection")
        return
    
    check_to_goal = D[check_node-1, 2303]
    targ_to_goal = D[targ_node-1, 2303]

    if check_to_goal > targ_to_goal:
        return 1
    elif check_to_goal == targ_to_goal:
        return 0
    elif check_to_goal < targ_to_goal:
        return -1
    

def TrajectoryTurnAround(behav_nodes, standard = 10, maze_type = 1):
    Direction = np.ones_like(behav_nodes, dtype = np.float64)
    TurnRound = np.zeros_like(behav_nodes, dtype = np.float64)
    for k in range(standard, behav_nodes.shape[0] - standard):
        Direction[k] = LocomotionDirection(check_node = behav_nodes[k - standard], targ_node = behav_nodes[k], maze_type = maze_type)

# Calculating behavior speed if it is not exist in behav_new.mat
def calc_speed(behav_positions = None, behav_time = None):
    # Do not delete NAN value! to keep the same vector length with behav_positions_original and behav_time_original
    # behav_positions, behav_time = Delete_NAN(behav_positions = behav_positions, behav_time = behav_time)
    dx = np.append(np.ediff1d(behav_positions[:,0]),0)
    dy = np.append(np.ediff1d(behav_positions[:,1]),0)
    dt = np.append(np.ediff1d(behav_time),33)
    dl = np.sqrt(dx**2+dy**2)
    behav_speed = dl / dt * 1000
    #_, _, behav_speed = Add_NAN(behav_positions = behav_positions, behav_time = behav_time, behav_nodes = behav_speed)
    return behav_speed


# Run all mice function ============================================================================================================================================
def run_all_mice(mylist: list, behavior_paradigm = 'CrossMaze', P = None, cam_degree = 180):
    date = mylist[0]      # str
    MiceID = mylist[1]    # str
    folder = mylist[2]    # str
    common = mylist[3]   # int
    maze_type = mylist[4] # int

    if behavior_paradigm == 'CrossMaze':
        totalpath = 'G:\YSY\Cross_maze'
        session = common
        c = 'session'
        p = os.path.join(totalpath, MiceID, date,"session "+str(session))
    elif behavior_paradigm == 'SimpleMaze':
        totalpath = 'G:\YSY\Simple_maze'
        training_day = common
        c = 'training_day'
        p = os.path.join(totalpath, MiceID, date)
    elif behavior_paradigm == 'ReverseMaze':
        totalpath = 'G:\YSY\Reverse_maze'
        training_day = common
        c = 'training_day'
        p = os.path.join(totalpath, MiceID, date)
    else:
        print("WARNING! Only 'CrossMaze','SimpleMaze','ReverseMaze' are valid value for argument behavior_paradigm!")
        return

    p_behav = os.path.join(p,'behav')
    mkdir(p_behav)

    # read in behav_new.mat
    behav_mat_path = os.path.join(folder,"behav_new.mat")
    if os.path.exists(behav_mat_path):
        behav_mat = loadmat(behav_mat_path); # read data    
        behav_positions = np.array(behav_mat['behav']["position_original"][0][0], dtype = np.float64)
        behav_time = np.array(np.concatenate(behav_mat['behav']['time'][0][0]), dtype = np.int64)
        behav_time_original = cp.deepcopy(behav_time)
        if behavior_paradigm == 'CrossMaze':
            behav_speed = np.array(behav_mat['behav']['speed'][0][0][0])
    else:
        print("WARNING!!! This session meet a run_code_error! behav_new.mat file loss!")
        #return
    
    trace = {'date':date,'MiceID':MiceID,c:common,'paradigm':'Cross Maze','maxWidth':maxWidth,'maxHeight':maxHeight,
             '_nbins':nbins,'_coords_range':coords_range,'p':p,'behav_folder':folder,'behav_path':behav_mat_path,
             'maze_type':maze_type,'nx':nx, 'ny':ny, 'behav_positions_original':cp.deepcopy(behav_positions), 
             'behav_time_original':cp.deepcopy(behav_time_original)}
        
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
    behav_positions, behav_time = clean_data(behav_positions, behav_time, maze_type = maze_type)
    # Add NAN value at the cross lap gap to plot the trajactory.
    behav_positions, behav_time = Add_NAN(behav_positions, behav_time, maze_type = maze_type)
    plot_trajactory(x = behav_positions[:,0], y = behav_positions[:,1], save_loc = p_behav,
                file_name = 'Trajactory_DeleteWrongData', maze_type = maze_type)
    print("    Figure 3 has done.")
    
    # save key args in trace.
    trace['behav_positions'] = cp.deepcopy(behav_positions)
    
    # Get a modified behav_time_original for interpolated
    start_time = behav_time[0]
    end_time = behav_time[-1]
    start_index = np.where(behav_time_original == start_time)[0][0]
    end_index = np.where(behav_time_original == end_time)[0][0]
    behav_time_original = behav_time_original[start_index:(end_index+1)]
    trace['behav_time_original'] = cp.deepcopy(behav_time_original)
    trace['behav_positions_original'] = trace['behav_positions_original'][start_index:(end_index+1)]

    # behav_speed
    if behavior_paradigm == 'CrossMaze':
        behav_speed = behav_speed[start_index:(end_index+1)]
        trace['behav_speed_original'] = cp.deepcopy(behav_speed)
    else:
        behav_speed = calc_speed(behav_positions = trace['behav_positions_original'], behav_time = trace['behav_time_original'])
        trace['behav_speed_original'] = cp.deepcopy(behav_speed)

    # Loop for trace correction. If do not satisfied with the result of one trial, you can input any number instead of 1 to 
    # repeat and start a new trial. Only for maze sessions because there's no need to do this for openfield.
    is_continue = 0

    # Affine transformation manually, 手动仿射变换框选 ---------------------------------------------------------------------------
    # define behavior x, y, roi for correction
    behav_x = behav_positions[:,0]
    behav_y = behav_positions[:,1]
    behav_roi = behav_mat['behav']['ROI'][0][0][0]
    track_length = behav_mat['behav']['trackLength'][0][0][0]
    
    # Read the video
    if behavior_paradigm == 'CrossMaze':
        video_name = os.path.join(folder,'0.avi')
    elif behavior_paradigm == 'SimpleMaze':
        video_name = os.path.join(folder,'behavCam1.avi')
    elif behavior_paradigm == 'ReverseMaze':
        video_name = os.path.join(folder,'0.avi')
    mean_frame = get_meanframe(video_name)
    
    ori_positions = behav_positions * behav_roi[2]/track_length +  [behav_roi[0], behav_roi[1]]

    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    ax.imshow(mean_frame.astype(np.int64))
    plot_trajactory(x = ori_positions[:,0], y = ori_positions[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax,
                    file_name = 'TraceOnMeanFrame_Raw', color = 'red', linewidth = 1, maze_type = maze_type)
    
    ori_positions, behav_time = Delete_NAN(ori_positions, behav_time)
    trace['ori_positions'] = cp.deepcopy(ori_positions)
    trace['behav_time_ori'] = cp.deepcopy(behav_time)
    print("    Figure 4 has done.")
    
    roi = np.int64([behav_roi[0], behav_roi[0]+behav_roi[2], behav_roi[1], behav_roi[1]+behav_roi[3]] )
    equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame))
        
    # to perform a four point transform used package opencv.
    # There'll be a window popping up and to select a four-edge shape.
    pd = PolygonDrawer(equ_meanframe,ori_positions, maxHeight = 960, maxWidth = 960)
    warped_image, warped_positions, M  = pd.run()
    cv2.imwrite(os.path.join(p_behav,"polygon.png"), warped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    warped_positions, behav_time = Add_NAN(warped_positions, behav_time, maze_type = maze_type)
    print("    Polygon = %s" % pd.points)
    
    # Get processed position args. -------------------------------------------------------------------------------------------
    processed_pos = warped_positions
    processed_pos, behav_time = Delete_NAN(processed_pos, behav_time)
    # A tolerable_range for out of range. If out of range, set the data point as NAN value.
    tolerable_range = 80 # 5 cm
    processed_pos[np.where((processed_pos[:,0] < 0)&(processed_pos[:,0] >= - tolerable_range))[0], 0] = 0
    processed_pos[np.where(processed_pos[:,0] < -tolerable_range)[0], 0] = np.nan
    processed_pos[np.where((processed_pos[:,1] < 0)&(processed_pos[:,1] >= - tolerable_range))[0], 1] = 0
    processed_pos[np.where(processed_pos[:,1] < -tolerable_range)[0], 1] = np.nan
    processed_pos[np.where((processed_pos[:,0] > maxWidth)&(processed_pos[:,0] <= maxWidth + tolerable_range))[0], 0] = maxWidth
    processed_pos[np.where(processed_pos[:,0] > maxWidth + tolerable_range)[0], 0] = np.nan
    processed_pos[np.where((processed_pos[:,1] > maxHeight)&(processed_pos[:,1] <= maxHeight + tolerable_range))[0], 1] = maxHeight
    processed_pos[np.where(processed_pos[:,1] > maxHeight + tolerable_range)[0], 1] = np.nan
    processed_pos, behav_time = Delete_NAN(processed_pos, behav_time)
    processed_pos, behav_time = Add_NAN(processed_pos, behav_time, maze_type = maze_type)    
        
    plt.figure(figsize = (6,6))
    ax = Clear_Axes(plt.axes())
    ax.imshow(warped_image)
    plot_trajactory(x = processed_pos[:,0], y = processed_pos[:,1], save_loc = p_behav, is_ExistAxes = True, ax = ax,
                        file_name = 'TraceOnMeanFrame', color = 'red', linewidth = 1, maze_type = maze_type)
    print("    Figure 5 has done.")


    if behavior_paradigm == 'SimpleMaze':
        trace['processed_pos_new'] = np.zeros_like(processed_pos)
        trace['processed_pos_new'] = cp.deepcopy(maxHeight - processed_pos[:,1])
        trace['processed_pos_new'] = cp.deepcopy(processed_pos[:,0])
    else:
        if cam_degree == 0:
            trace['processed_pos_new'] = cp.deepcopy(processed_pos)
        elif cam_degree == 180:
            trace['processed_pos_new'] = [maxWidth, maxHeight] - processed_pos

    processed_pos_new = cp.deepcopy(trace['processed_pos_new'])

    trace['processed_pos'] = cp.deepcopy(processed_pos)
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
    
        # Interpolate values

        trace = TrajectoryInterpolated(trace, P = P)
    
        # Check the effect of cross-wall correction
        plot_trajactory_comparison(trace['correct_pos'], trace['interpolated_pos'], is_position_transform = True, 
                                   save_loc = p_behav, file_name = 'Interpolated-Correction_Trajactory', maze_type = maze_type)
        print('    Figure 9 has done.')
    
        # Transform nodes to x,y location to check the efficiency of correction
        plot_trajactory_comparison(trace['correct_nodes'], trace['interpolated_nodes'], is_node = True, 
                                   save_loc = p_behav, file_name = 'Interpolated-Correction_Nodes', maze_type = maze_type)
        print("    Figure 10 has done.")

    # For open field, nothing need to do. ------------------------------------------------------------------------------------
    elif maze_type == 0:
        trace['correct_pos'] = cp.deepcopy(trace['processed_pos_new'])
        trace['correct_nodes'] = cp.deepcopy(trace['behav_nodes'])
        trace['correct_time'] = cp.deepcopy(trace['behav_time'])
        trace['interpolated_pos'] = cp.deepcopy(trace['processed_pos_new'])
        trace['interpolated_nodes'] = cp.deepcopy(trace['behav_nodes'])
        trace['interpolated_time'] = cp.deepcopy(trace['behav_time'])

    # =======================================================================================================================
    # calculating ratemap ----------------------------------------------------------------------------------------------------
    pos, correct_time, behav_nodes = Delete_NAN(trace['correct_pos'], trace['correct_time'], trace['correct_nodes'])
    activation = np.append(np.ediff1d(behav_time),33)
    activation[np.where(activation > 100)[0]] = 100
    occu_time = np.zeros(2304, dtype = np.float64)
    print("    calculating occupation time:")
    for i in tqdm(range(2304)):
        idx = np.where(behav_nodes == i+1)[0]
        if len(idx) == 0:
            occu_time[i] = np.nan
        else:
            occu_time[i] = np.nansum(activation[idx])

    trace['occu_time'] = cp.deepcopy(occu_time)
    occu_time_tmp = cp.deepcopy(occu_time)
    
    for i in tqdm(range(2304)):
        if i+1 in Father2SonGraph[1] or i+1 in Father2SonGraph[144]:
            occu_time_tmp[i] = 0
    nan_idx = np.where(np.isnan(occu_time_tmp))[0]
    occu_time_tmp[nan_idx] = 0
    
    # Plot bahavioral ratemap
    print("    Draw behavioral ratemap...")
    plt.figure(figsize = (8,6))
    ax = Clear_Axes(plt.axes())
    if maze_type in [1,2]:
        DrawMazeProfile(nx = 48, maze_type = trace['maze_type'], linewidth = 2, color = 'yellow', axes = ax)
    im = ax.imshow(np.reshape(occu_time_tmp/1000,[48,48]), cmap = 'hot')
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label('occupation time: s', fontsize = 16)
    plt.savefig(os.path.join(p_behav,'behav_ratemap.png'),dpi=600)
    plt.savefig(os.path.join(p_behav,'behav_ratemap.svg'),dpi=600)
    plt.close()
    print("    Figure behavioral ratemap has done.")
    
    # Plot speed distribution figure ------------------------------------------------------------------------------------------
    # Delete All NAN Value
    trace['processed_pos_new'], trace['behav_time'], trace['behav_nodes'] = Delete_NAN(trace['processed_pos_new'], 
                                                                          trace['behav_time'], trace['behav_nodes'])
    trace['correct_pos'], trace['correct_time'], trace['correct_nodes'] = Delete_NAN(trace['correct_pos'], 
                                                                          trace['correct_time'], trace['correct_nodes'])
    speed_idx = []
    behav_time = trace['behav_time']
    for i in tqdm(range(behav_time.shape[0])):
        idx = np.where(trace['behav_time_original'] == behav_time[i])[0][0]
        speed_idx.append(idx)
    trace['original_to_behav_idx'] = np.array(speed_idx,dtype = np.int64)
    trace['behav_speed'] = cp.deepcopy(trace['behav_speed_original'][trace['original_to_behav_idx']])
    
    plt.figure(figsize = (8,6))
    MAX_X = (np.nanmax(trace['behav_speed']) // 5 + 1) * 5
    ax = Clear_Axes(plt.axes(), close_spines = ['top','right'], ifxticks = True, ifyticks = True)
    ax.hist(trace['behav_speed'], bins = 50)
    ax.set_xlabel('Speed (cm/s)', fontsize = 16)
    ax.set_ylabel('Counts', fontsize = 16)
    ax.set_xticks(np.linspace(0,MAX_X,int(MAX_X/5)+1))
    plt.savefig(os.path.join(p_behav,'speed_distribution.png'),dpi=600)
    plt.savefig(os.path.join(p_behav,'speed_distribution.svg'),dpi=600)
    plt.close()
    print('    Figure speed distribution has done.')
    

    # Save files
    with open(os.path.join(p,"trace_behav.pkl"), 'wb') as f:
        pickle.dump(trace, f)
    
    print("    ",os.path.join(p,"trace_behav.pkl")," has been saved successfully!")




if __name__ == '__main__':
    totalpath = "G:\YSY\Reverse_maze"
    print("Initiate key matrix that will be used in interpolated algorithm. It'll take tens of seconds.")
    P1, P2 = P_Matrice()
    file = pd.read_excel(os.path.join(totalpath,"Back_and_Forth_metadata_time.xlsx"), sheet_name = "behavior")
    
    i = 1

    print(i,"Mice "+str(int(file['MiceID'][i]))+" , date "+str(int(file['date'][i]))+", training_day "+str(int(file['training_day'][i]))
          +", maze type "+str(int(file['maze_type'][i]))+"............................................................")
    mylist = [str(int(file['date'][i])), str(int(file['MiceID'][i])), str(file['recording_folder'][i]), int(file['training_day'][i]),
              int(file['maze_type'][i])]
    P = P1 if file['maze_type'][i] == 1 else P2
    run_all_mice(mylist, behavior_paradigm = 'ReverseMaze', P = P)
    print("Done.",end='\n\n\n')
