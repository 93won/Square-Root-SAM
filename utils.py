import numpy as np
import matplotlib.pyplot as plt

def getRotation(heading):
    """
    (1) Function
        - Get rotation matrix from heading(radian)
    """
    cos = np.cos(heading)
    sin = np.sin(heading)

    rotation = np.array([[cos, -sin],
                        [sin, cos]])

    return rotation

def vecToSE2(pose):
    # convert pose [x, y, heading] to SE2 format

    """
    (1) Function
        - convert pose [x, y, heading] to SE2 matrix

    (2) Input
        - pose = [x, y, theta]

    (3) Output
        - SE2 matrix
    """

    x, y, heading = pose
    rotation = getRotation(heading)

    SE2 = np.eye(3)

    SE2[:2, :2] = rotation
    SE2[0, 2] = x
    SE2[1, 2] = y

    return SE2

def SE2ToVec(SE2):
    # convert SE2 matrix to pose [x, y, heading]

    """
    (1) Function
        - convert SE2 matrix to pose [x, y, heading]

    (2) Input
        - SE2 matrix

    (3) Output
        - pose = [x, y, theta]
    """

    x, y = SE2[:2, 2]
    cos = np.clip(SE2[0, 0], -1, 1)
    sin = np.clip(SE2[1, 0], -1, 1)
    heading = np.arctan2(sin, cos)

    pose = [x, y, heading]

    return pose


def readData(nfile, efile):
    """
    (1) Function
        - Read data

    (2) Input
        - nfile : file path of node data
        - efile : file path of edge data

    (3) Output
        - nodes : [[pose_0], [pose_1], ... , [pose_n]]
        - edges : [[[idx of edges observed from nodes[0], e_0, omega_o],
                    ... ,
                   [idx of edges observed from nodes[0], e_k, omega_k]],
                    ... ,
                   [[idx of edges observed from nodes[n], e_0, omega_0],
                   ... ,
                   [idx of edges observed from nodes[n], e_k, omega_k]]]
    """

    nodes = np.loadtxt(nfile, usecols=range(1, 5))[:, 1:]
    edges_aux = np.loadtxt(efile, usecols=range(1, 12))
    idxs_edge_aux = edges_aux[:, :2]
    means_edge_aux = edges_aux[:, 2:5]
    infms_edge_aux = []

    edges = [[]]

    nb_edges = edges_aux.shape[0]


    idx_ref = -1


    for i in range(nb_edges):

        infm = np.zeros((3, 3), dtype=np.float64)
        # edges[i, 5:11] ... upper-triangular block of the information matrix (inverse cov.matrix) in row-major order
        infm[0, 0] = edges_aux[i, 5]
        infm[1, 0] = infm[0, 1] = edges_aux[i, 6]
        infm[1, 1] = edges_aux[i, 7]
        infm[2, 2] = edges_aux[i, 8]
        infm[0, 2] = infm[2, 0] = edges_aux[i, 9]
        infm[1, 2] = infm[2, 1] = edges_aux[i, 10]

        infms_edge_aux.append(infm)

    for i in range(nb_edges):
        idx_max = np.int32(np.max(idxs_edge_aux[i]))
        idx_min = np.int32(np.min(idxs_edge_aux[i]))


        if idx_max != idx_ref:
            idx_ref = idx_max
            edges.append([[idx_min, means_edge_aux[i], infms_edge_aux[i]]])

        else:
            idx_min = np.int32(np.min(idxs_edge_aux[i]))
            edges[idx_ref].append([idx_min, means_edge_aux[i], infms_edge_aux[i]])


    return nodes, edges

def show_data(ax, xs, ys, hs):
    """
        (1) Function
            - Show coordinates of nodes and headings

        (2) Input
            - x-coordinates, y-coordinates, headings
        """

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)

    nb_data = xs.shape[0]

    for i in range(nb_data):
        r = 0.5
        circle = plt.Circle((xs[i], ys[i]), radius=r, color='green')

        ax.add_patch(circle)
        el = 1

        label = ax.annotate("X"+str(i), xy=(xs[i], ys[i]), fontsize=7, ha="center")
        ax.arrow(xs[i] + r*el*np.cos(hs[i]), ys[i] + r*el*np.sin(hs[i]), el*np.cos(hs[i]), el*np.sin(hs[i]),
                 fc="k", ec="k", head_width=0.5, head_length=0.5, width=0.1)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()

    plt.show()


from math import copysign, hypot

import numpy as np


def gram_schmidt_process(A):
    """Perform QR decomposition of matrix A using Gram-Schmidt process."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize empty orthogonal matrix Q.
    Q = np.empty([num_rows, num_rows])
    cnt = 0

    # Compute orthogonal matrix Q.
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
            u -= proj

        e = u / np.linalg.norm(u)
        Q[:, cnt] = e

        cnt += 1  # Increase columns counter.

    # Compute upper triangular matrix R.
    R = np.dot(Q.T, A)

    return (Q, R)


def householder_reflection(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterative over column sub-vector and
    # compute Householder matrix to zero-out lower triangular matrix entries.
    for cnt in range(num_rows - 1):
        x = R[cnt:, cnt]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -A[cnt, cnt])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_cnt = np.identity(num_rows)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_cnt, R)
        Q = np.dot(Q, Q_cnt.T)

    return (Q, R)


def givens_rotation(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if np.abs(R[row, col]) >= 1e-7:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)


def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)

def back_substitution(A, b):
    n = b.size
    x = np.zeros_like(b)

    if A[n-1, n-1] == 0:
        raise ValueError


    x[n-1] = b[n-1]/A[n-1, n-1]
    C = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += A[i, j]*x[j]

        C[i, i] = b[i] - bb
        x[i] = C[i, i]/A[i, i]

    return x


def plotGraph(graph, type):
    nb_nodes = len(graph.nodes)

    xs = []
    ys = []

    for i in range(nb_nodes):
        x = graph.nodes[i].pose[0]
        y = graph.nodes[i].pose[1]
        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys, type, markersize=0.5)
    plt.show()