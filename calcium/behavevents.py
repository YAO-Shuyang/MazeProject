import numpy as np
from mylib.maze_utils3 import GetDMatrices

class BehavEvents(object):
    def __init__(self, graph: dict, behav_nodes: np.ndarray, DMatrix: np.ndarray) -> None:
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
        self.behav_nodes = behav_nodes
        self.D = DMatrix 
    
    def node_to_dis(self) -> None:
        dis = np.zeros_like(self.behav_nodes, dtype=np.int64)

        for i in range(self.behav_nodes.shape[0]):
            dis[i] = self.D[self.behav_nodes[i]-1, 2303]

        self.dis = dis

    def get_dir(self, dl: int):
        if dl < 0:
            return 1
        elif dl > 0:
            return -1
        else:
            assert False

    def monitor_orient(self):
        self.curr_direc = 0
        self.curr_frame = 0
        self.curr_dis = self.dis[0]

        self.event_frame = []
        self.direc = []

        for i in range(1, self.dis.shape[0]):
            if self.get_dir(self.dis[i] - self.curr_dis) != self.curr_direc:
                if self.curr_direc == 0:
                    self.event_frame.append(i)
                    self.