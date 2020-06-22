from utils import *

import numpy as np
from numpy.linalg import qr
from numpy.linalg import inv



class Node(object):
    def __init__(self, idx, pose):
        self.idx = idx
        self.pose = pose

class Edge(object):
    def __init__(self, idxs, mean, infm):
        self.idxs = idxs # from to
        self.mean = mean
        self.infm = infm

        self.factor = np.dot(np.dot(self.mean.T, infm), self.mean)

class Graph(object):
    def __init__(self):
        self.max_node = 2000
        self.max_edge = 10000
        self.nodes = []
        self.edges = []
        self.A_frame = np.zeros(shape=(3*(self.max_edge+1),3*(self.max_node)))
        self.A_frame[:3, :3] = np.eye(3, dtype=np.float64)         # information matrix corresponding to factor graph
        self.b_frame = np.zeros(shape=(3*(self.max_edge+1), 1), dtype=np.float64)         # information vector
        self.nb_edges = 0

        self.A = []
        self.b = []


    def addNode(self, idx, pose):
        """
        (1) Function
            - Add node object to graph

        (2) Input
            - idx  : index of node
            - pose : pose information of node
        """

        node = Node(idx, pose)
        self.nodes.append(node)



    def addEdge(self, idx_from, idx_to, mean, infm):
        """
        (1) Function
            - Add factor between two nodes

        (2) Input
            - idx_from : index of the node where the observation was made
            - idx_to   : index of the observed node
            - mean     : mean of measurement
            - infm     : information matrix of measurement
        """
        assert(idx_from > idx_to)

        edge = Edge([idx_from, idx_to], mean, infm)
        self.edges.append(edge)
        self.nb_edges += 1


    def linearize(self, idx_edge):

        """
        (1) Function
            - linearize one observation
        """
        edge = self.edges[idx_edge]
        mean = edge.mean
        infm = edge.infm

        Z_ij = vecToSE2(mean)

        i, j = edge.idxs

        # [from to] : reverse order WRT markov chain

        x_j = self.nodes[np.int32(j)].pose
        x_i = self.nodes[np.int32(i)].pose

        T_i = vecToSE2(x_i)   # pose_i -> transform_i
        T_j = vecToSE2(x_j)   # pose_j -> transform_j

        R_i = T_i[:2, :2]
        R_z = Z_ij[:2, :2]

        si = np.sin(x_i[2])
        ci = np.cos(x_i[2])

        dR_i = np.array([[-si, ci], [-ci, -si]], dtype=np.float64).T
        dt_ij = np.array([x_j[:2] - x_i[:2]], dtype=np.float64).T

        """
        H_i = [[-R_z.T @ R_i.T   R_z.T @ dR_i.T @ dt_ij]
              [       0                    -1         ]]
             
        H_j = [[R_z.T @ R_i.T   0]
              [       0       -1]]        
        """

        H_i = np.vstack((np.hstack((np.dot(-R_z.T, R_i.T), np.dot(np.dot(R_z.T, dR_i.T), dt_ij))), [0, 0, -1]))
        H_j = np.vstack((np.hstack((np.dot(R_z.T, R_i.T), np.zeros((2, 1), dtype=np.float64))), [0, 0, 1]))

        e = SE2ToVec(np.dot(np.dot(inv(Z_ij), inv(T_i)), T_j))

        A_i = np.dot(np.sqrt(infm), H_i)
        A_j = np.dot(np.sqrt(infm), H_j)
        b = -np.dot(np.sqrt(infm), e)

        self.A_frame[(3 * (idx_edge+1)):(3 * (idx_edge+2)), (3 * i):(3 * (i + 1))] = A_i
        self.A_frame[(3 * (idx_edge+1)):(3 * (idx_edge+2)), (3 * j):(3 * (j + 1))] = A_j
        self.b_frame[(3 * (idx_edge+1)):(3 * (idx_edge+2)), 0] = b

    def solveFullChol(self):

        """
        (1) Function
            - solve linear system using cholesky decomposition

        """

        Lambda = np.dot(self.A.T, self.A)
        b = np.dot(self.A.T, self.b)
        R = np.linalg.cholesky(Lambda).T
        d = np.dot(inv(R.T), b)
        dx = np.dot(inv(R), d)

        dx[np.isnan(dx)] = 0
        for i in range(len(self.nodes)):
            self.nodes[i].pose += dx[i*3:(i+1)*3, 0]

    def solveFullQR(self):

        """
        (1) Function
            - solve linear system using QR decomposition

        """

        Q, R = qr(self.A)
        d = np.dot(Q.T, self.b)

        dx = np.dot(inv(R), d)

        dx[np.isnan(dx)] = 0
        for i in range(len(self.nodes)):
            self.nodes[i].pose += dx[i*3:(i+1)*3, 0]


    def optimize(self, iter_num, mode='qr'):

        for iter in range(iter_num):
            nb_edges = len(self.edges)

            self.A_frame = np.zeros(shape=(3 * (self.max_edge + 1), 3 * (self.max_node)))
            self.A_frame[:3, :3] = np.eye(3, dtype=np.float64)  # information matrix corresponding to factor graph
            self.b_frame = np.zeros(shape=(3 * (self.max_edge + 1), 1), dtype=np.float64)  # information vector

            # Re linearization

            for i in range(nb_edges):
                self.linearize(i)

            self.A = self.A_frame[:(len(self.edges) + 1) * 3, :len(self.nodes) * 3]
            self.b = self.b_frame[:(len(self.edges) + 1) * 3]
            if mode == 'qr':
                self.solveFullQR()

            elif mode == 'chol':
                self.solveFullChol()












