import numpy as np
from mylib.maze_graph import maze_graphs, EndPoints, correct_paths
from mylib.maze_utils3 import GetDMatrices, spike_nodes_transform

class BehavEvents(object):
    def __init__(
        self, 
        graph: dict, 
        correct_path: np.array,
        behav_nodes: np.ndarray, 
        DMatrix: np.ndarray, 
        ending: int = 144
    ) -> None:
        """
        Parameters
        ----------
        graph : dict
            The graph of the environment.
        behav_nodes : np.ndarray
            The trajectory of the animal.
        DMatrix : np.ndarray
            The distance matrix to determine the moving direction of the mice
        """
        self.G = graph
        self.correct_path = correct_path
        self.behav_nodes = behav_nodes.astype(np.int64)
        self.D = DMatrix 
        self.ending = ending
        self._node_to_dis()
        self._monitor_orient()
    
    def _node_to_dis(self) -> None:
        dis = np.zeros_like(self.behav_nodes, dtype=np.int64)
        

        for i in range(self.behav_nodes.shape[0]):
            dis[i] = self.D[self.behav_nodes[i]-1, self.ending-1]

        self.dis = dis

    def _monitor_orient(self):
        event_frames = [0]
        event_direcs = []
        delete = []
        behav_nodes = self.behav_nodes

        prev_dir = 0
        prev_dis = self.dis[0]
        curr_dir = self.dis[0]

        for i in range(1, behav_nodes.shape[0]):
            if behav_nodes[i] not in self.correct_path:
                delete.append(i)
                continue
            else:
                dl = self.dis[i] - prev_dis

                if dl == 0:
                    continue

                curr_dir = -1 if dl > 0 else 1
                if curr_dir != prev_dir:
                    if prev_dir == 0:
                        event_direcs.append(curr_dir)
                        prev_dir = curr_dir
                    else:
                        event_frames.append(i)
                        event_direcs.append(curr_dir)
                        prev_dir = curr_dir

                prev_dis = self.dis[i]
        
        event_frames.append(behav_nodes.shape[0])
        event_direcs.append(curr_dir)

        self.event_frames, self.event_direcs, self.delete_frames = np.array(event_frames, np.int64), np.array(event_direcs, np.int64), np.array(delete, np.int64)
    
    def get_range(self, center_frame: int, beg_frame: int, end_frame: int, window_length: int = 30):
        beg = center_frame - window_length if center_frame >= window_length else 0
        end = center_frame + window_length if end_frame-1 - center_frame >= window_length else end_frame-1
        return beg, end


    def set_delete_frames(self, **kwargs):
        if not self.event_direcs.shape[0] == self.event_frames.shape[0]:
            print(self.event_direcs.shape[0], self.event_frames.shape[0])
            print(self.event_direcs, self.event_frames, self.behav_nodes.shape[0])
        assert self.event_direcs.shape[0] == self.event_frames.shape[0]
        
        frame_labels = np.zeros_like(self.behav_nodes, np.float64)
        delete_frames = self.delete_frames

        # set direction first
        for i in range(len(self.event_frames)-1):
            frame_labels[self.event_frames[i]:self.event_frames[i+1]] = self.event_direcs[i]

        # delete frames: set frames to be deleted as nan values.
        for i in range(1, len(self.event_frames)-1):
            beg, end = self.get_range(i, 0, self.behav_nodes.shape[0], **kwargs)
            frame_labels[beg: end+1] = np.nan

        frame_labels[delete_frames] = np.nan
        self.frame_labels = frame_labels

    @staticmethod
    def get_frame_labels(behav_nodes: np.ndarray, maze_type: int, **kwargs):
        graph = maze_graphs[(maze_type, 12)]
        co_path = correct_paths[maze_type]
        behav_nodes = spike_nodes_transform(behav_nodes, nx = 12)
        DMatrix = GetDMatrices(maze_type, 12)
        
        Obj = BehavEvents(graph, co_path, behav_nodes, DMatrix, ending=EndPoints[maze_type])
        Obj.set_delete_frames(**kwargs)
        return Obj.frame_labels
                