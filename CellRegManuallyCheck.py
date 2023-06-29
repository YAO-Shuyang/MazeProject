from mylib.statistic_test import *
import networkx as ntx

class Nodes():
    def __init__(self, idx = 0, level = 1):
        self.cell_index = idx
        self.level = level
        self.nodes = []
    
    def append(self,node):
        self.nodes.append(node)

class CellIndexTree(Nodes):
    def __init__(self, root, level, dateset = [], CellReg_loc = '', SFP_loc = r'G:\YSY\Cross_maze\11095\Maze2-footprint'):
        self.cell_index = np.int64(root)
        self.level = np.int64(level)
        self.MAX_LEVEL = len(dateset)
        self.dateset = dateset
        self._build_tree = False
        self.CellRegLoc = CellReg_loc
        self.SFP_loc = SFP_loc
        self._build_graph = False

    def _left_generate(self, left):
        file_name = self.dateset[self.level-1][6::]+'-'+self.dateset[self.level-1-left][6::]
        loc = os.path.join(self.CellRegLoc, file_name, 'cellRegistered.mat')
        if os.path.exists(loc):
            index_map = ReadCellReg(loc, open_type = 'h5py').T
        else:
            print(loc, ' is not exist! =====================================')
            return
        idx = np.where(index_map[:,0] == self.cell_index)[0][0]
        if index_map[idx,1] == 0:
            return
        new_nodes = CellIndexTree(index_map[idx,1],self.level-left, dateset = self.dateset, 
                                  CellReg_loc = self.CellRegLoc, SFP_loc = self.SFP_loc)
        self.nodes.append(new_nodes)

    def _right_generate(self,right):
        file_name = self.dateset[self.level + right - 1][6::]+'-'+self.dateset[self.level-1][6::]
        loc = os.path.join(self.CellRegLoc, file_name, 'cellRegistered.mat')
        if os.path.exists(loc):
            index_map = ReadCellReg(loc, open_type = 'h5py').T
        else:
            print(loc, ' is not exist! =====================================')
            return
        idx = np.where(index_map[:,1] == self.cell_index)[0][0]
        if index_map[idx,0] == 0:
            return
        new_nodes = CellIndexTree(index_map[idx,0],self.level+right, dateset = self.dateset, 
                                  CellReg_loc = self.CellRegLoc, SFP_loc = self.SFP_loc)
        self.nodes.append(new_nodes)
    
    def node_id(self):
        return "Node(date: "+str(self.dateset[self.level-1])+",idx:"+str(self.cell_index)+")"
    
    def Build_Tree(self):
        self.nodes = []
        self._build_tree = True
        print("Build tree...")

        if self.level >= 4:
            left = 3
        else:
            left = self.level - 1
        if self.level <= self.MAX_LEVEL - 3:
            right = 3
        else:
            right = self.MAX_LEVEL - self.level

        print(self.node_id()+' has (left:'+str(left)+',right:'+str(right)+')')

        for l in np.arange(left,0,-1):
            self._left_generate(l)
        for r in np.arange(1,right+1):
            self._right_generate(r)

    def Visualize(self, traceset = []):
        print('Visualizing '+self.node_id())
        maze_type = traceset[0]['maze_type']

        if len(traceset) != len(self.dateset):
            print('warning! len(traceset) != len(self.dateset)',len(traceset),len(self.dateset))
            return
        
        fig, axes = plt.subplots(ncols = self.MAX_LEVEL, nrows = 3, figsize = (7*self.MAX_LEVEL,18))
        for i in range(self.MAX_LEVEL):
            axes[0,i] = Clear_Axes(axes[0,i])
            axes[1,i] = Clear_Axes(axes[1,i])
            axes[2,i] = Clear_Axes(axes[2,i])
        
        print(self.level-1, self.cell_index-1)
        color = 'red' if traceset[self.level-1]['is_placecell'][self.cell_index-1] == 1 else 'black'
        SI = traceset[self.level-1]['SI_all'][self.cell_index-1]
        im = axes[0,self.level-1].imshow(np.reshape(traceset[self.level-1]['smooth_map_all'][self.cell_index-1],[48,48]), cmap = 'jet')
        cbar = plt.colorbar(im, ax = axes[0,self.level-1])
        axes[0,self.level-1].set_title('Date:'+self.dateset[self.level-1]+'\nCell '+str(self.cell_index)+'\nSI = '+str(round(SI,3)),color = color)
        if maze_type in [1,2]:
            DrawMazeProfile(axes = axes[0,self.level-1], maze_type = maze_type, nx = 48) 
        self._SFPMap(ax = axes[1,self.level-1], cell_index = self.cell_index, ax_ZoomIn = axes[2,self.level-1], level = self.level)

        for i in range(len(self.nodes)):
            l = self.nodes[i].level-1
            v = np.int64(self.nodes[i].cell_index)
            if v == 0:
                axes[0,l].set_title("0")
                continue
            color = 'red' if traceset[l]['is_placecell_isi'][v-1] == 1 else 'black'
            SI = traceset[l]['SI_all'][v-1]
            im = axes[0,l].imshow(np.reshape(traceset[l]['smooth_map_all'][v-1],[48,48]), cmap = 'jet')
            cbar = plt.colorbar(im, ax = axes[0,l])
            axes[0,l].set_title('Date:'+self.dateset[l]+'\nCell '+str(v)+'\nSI = '+str(round(SI,3)), color = color)
            if maze_type in [1,2]:
                DrawMazeProfile(axes = axes[0,l], maze_type = maze_type, nx = 48) 
            self._SFPMap(ax = axes[1,l], cell_index = v, ax_ZoomIn = axes[2,l], level = l+1)

        plt.show()


    def DoubleCheck(self, traceset = []):
        print("Double check!-------------------------------------------------------------------------------")
        if self._build_tree == False:
            print("Please Build_Tree() first! Abort --------------------------")
            return

        for i in range(len(self.nodes)):
            if self.nodes[i].cell_index == 0:
                continue
            print('  set root as '+self.nodes[i].node_id())
            self.nodes[i].Build_Tree()
            self.nodes[i].Visualize(traceset = traceset)
        print("----------------------------------------------------------------------------------------------")

    def list_node(self):
        if len(self.nodes) == 0:
            print("There's no self.nodes.")
            return
        else:
            for n in self.nodes:
                print("  "+n.node_id())
   
    def MannualCheck(self, input_list = []):
        print("Here we have these "+str(len(self.nodes))+" nodes:")
        self.list_node()
        print()
        print("The input list is:", input_list)     
        is_sure = input("Do you sure about the input list? Please input yes/no \n")
        
        if is_sure in ['no','No','n','N','f','F','false','False','abort','c','C']:
            print("    Abort.")
            return

        if np.nanmax(input_list) >= len(self.nodes):
            print("input list error! Please input again! Abort.")
            return

        self.nodes = [self.nodes[i] for i in input_list]
        print('Mannually check successfully!')
        self.list_node()

    def rebuilt(self, traceset = []):
        self.DoubleCheck(traceset)
        self.list_node()
        self.Visualize(traceset)
        idx = int(input("Which nodes do you prefer to test it's sons.\n\n"))
        if idx >= len(self.nodes):
            print('Input number is too big! Abort ------------')
            return
        self.nodes[idx].list_node()
        self.nodes[idx].Visualize(traceset)
        idx2 = int(input('Which one do you prefer to append into cell list as a son of CellIndexTree object? If you do not want to rebuilt the tree, please input -1.\n'))
        if idx2 == -1:
            print('Abort.')
            return
        elif idx2 >= len(self.nodes[idx].nodes) or idx2 < 0:
            print('Input number is too big! Abort ------------ 2')
            return
        else:
            self.append(self.nodes[idx].nodes[idx2])

        self.list_node()
        self.Visualize(traceset)

    def _is_detect(self, l1 = 1, l2 = 2, idx1 = 1, idx2 = 2):
        if l1 >= l2:
            print('l1 must be smaller than l2, please check again.')
            return False
        file_name = self.dateset[l2-1][6::]+'-'+self.dateset[l1-1][6::]
        loc = os.path.join(self.CellRegLoc, file_name, 'cellRegistered.mat')
        if os.path.exists(loc):
            index_map = ReadCellReg(loc, open_type = 'h5py').T
        else:
            print(loc, ' is not exist! =====================================')
            return False
        
        idx = np.where(index_map[:,0] == idx2)[0][0]
        if index_map[idx,1] == idx1:
            return True
        else:
            return False

    def _SFPMap(self, ax = None, ax_ZoomIn = None, cell_index = 1, level = 1):
        if ax is None:
            print("WARNING! Please input axes object! Report by self._SFPMap()")
            return

        SFPpath = os.path.join(self.SFP_loc, 'SFP'+str(self.dateset[level-1])+'.mat')
        if os.path.exists(SFPpath):
            with h5py.File(SFPpath, 'r') as f:
                sfp = np.array(f['SFP'])
                footprint = np.nanmax(sfp,axis = 2)
        
        fp_min = np.nanmin(footprint)
        fp_max = np.nanmax(footprint)
        cell_footprint =  np.where(sfp[:,:,cell_index-1] != 0, fp_max, np.nan)
        ax.imshow(footprint.T, vmin = fp_min, vmax = fp_max, cmap = 'Greys_r')
        im = ax.imshow(cell_footprint.T, vmin = fp_min, vmax = fp_max)
        ax.set_aspect('equal')

        if ax_ZoomIn is None:
            print("WARNING! Please input axes_ZoomIn object! Report by self._SFPMap()")
            return

        cfp_x, cfp_y = np.where(np.isnan(cell_footprint) == False)
        dim0_min_idx = np.nanmin(cfp_x)
        dim0_max_idx = np.nanmax(cfp_x)
        dim1_min_idx = np.nanmin(cfp_y)
        dim1_max_idx = np.nanmax(cfp_y)
        dim0_length = dim0_max_idx - dim0_min_idx
        dim1_length = dim1_max_idx - dim1_min_idx

        if dim0_min_idx >= dim0_length*2:
            dim0_beg = dim0_min_idx - dim0_length*2
        else:
            dim0_beg = 0
        
        if dim0_max_idx + dim0_length*2 <= footprint.shape[0]-1:
            dim0_end = dim0_max_idx + dim0_length*2
        else:
            dim0_end = footprint.shape[0]-1
        
        if dim1_min_idx >= dim1_length*2:
            dim1_beg = dim1_min_idx - dim1_length*2
        else:
            dim1_beg = 0
        
        if dim1_max_idx + dim1_length*2 <= footprint.shape[0]-1:
            dim1_end = dim1_max_idx + dim1_length*2
        else:
            dim1_end = footprint.shape[0]-1

        ax_ZoomIn.imshow(footprint[dim0_beg:dim0_end+1, dim1_beg:dim1_end+1].T, vmin = fp_min, vmax = fp_max, cmap = 'Greys_r')
        im = ax_ZoomIn.imshow(cell_footprint[dim0_beg:dim0_end+1, dim1_beg:dim1_end+1].T, vmin = fp_min, vmax = fp_max)
        ax_ZoomIn.set_aspect('equal')

    def Build_Graph(self, cell_list = None):
        if cell_list is not None and len(cell_list) != len(self.dateset):
            print("User Notes:\n    If you input nothing, it will use member 'nodes' of mylib.CellRegManuallyCheck.CellIndexTree object to build a graph. If you input a list, the list must have the same length as dateset. ", end='')
            return

        if cell_list is None:
            # Initiate cell list or reset self.nodes
            cell_list = [0,0,0,0,0,0]
            cell_list[self.level-1] = self.cell_index
            if len(self.nodes) != 0:
                for i in range(len(self.nodes)):
                    cell_list[self.nodes[i].level-1] = self.nodes[i].cell_index

        # Initiate graph and generate graph
        G = nx.Graph()
        for i in range(self.MAX_LEVEL):
            if cell_list[i] != 0:
                G.add_node('L'+str(i+1)+'-'+'C'+str(cell_list[i]), size = 15)

        print("Generate Graph:")
        for i in tqdm(range(self.MAX_LEVEL-1)):
            for j in range(i+1, self.MAX_LEVEL):
                if cell_list[i] != 0 and cell_list[j] != 0:
                    is_overdetect = self._is_detect(l1 = i+1, l2 = j+1, idx1 = cell_list[i], idx2 = cell_list[j])
                    if is_overdetect == True:
                        G.add_edge('L'+str(i+1)+'-'+'C'+str(cell_list[i]),'L'+str(j+1)+'-'+'C'+str(cell_list[j]), weight = 1)
        
        pos=ntx.circular_layout(G)
        print(pos)
        print("Draw Graph:")
        fig = plt.figure(figsize=(6,6))
        ax = Clear_Axes(plt.axes())
        ntx.draw(G,pos = pos, with_labels=True)
        ax.set_title('Graph of input vector')
        plt.show()
                
if __name__ == '__main__':
    dateset = ['20220820','20220822','20220824','20220826','20220828','20220830']
    idx = np.where((f1['date'] >= 20220820)&(f1['MiceID']==11095)&(f1['maze_type']==2))[0]
    trace_set = TraceFileSet(idx, tp = 'E:\Data\Cross_maze')
    root = 26
    level = 1
    cell = CellIndexTree(root = idx, level = level, dateset = dateset, 
                         CellReg_loc = r'E:\Data\Cross_maze\11095\Maze1-footprint\Cell_reg', SFP_loc = r'E:\Data\Cross_maze\11095\Maze2-footprint')
    cell.Build_Tree()
    cell.DoubleCheck(traceset = trace_set)
    cell.list_node()
    cell.Visualize(traceset = trace_set)
     
