#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network tools.
"""

import math
import numpy as np
from numpy import linalg as la
from functools import partialmethod

ran = np.random.default_rng() # random generator

from tvopt import sets, utils


#%% NETWORK CLASSES

class Network():
    """
    Representation of an undirected network.
    
    The class implements an undirected network defined from the adjacency
    matrix. The class provides methods for different communication protocols,
    such as node-to-node and broadcast.
    
    Transmissions are implemented via the `buffer` attribute of the network:
    the sender stores the packet to be transmitted in the `buffer` dictionary,
    specifying the recipient, which can then access the packet.
    
    By convention, the nodes in the network are indexed from :math:`0` to
    :math:`N-1`, where :math:`N` is the total number of nodes.
    
    Attributes
    ----------
    adj_mat : ndarray
        The adjacency matrix of the network.
    N : ndarray
        The number of nodes in the network.
    weights : ndarray
        The consensus weight matrix, if not specified in the constructor
        this is the Metropolis-Hastings weight matrix.
    neighbors : list
        A list whose :math:`i`-th element is a list of node :math:`i`'s
        neighors.
    degrees : list
        The number of neighbors of each node.
    buffer : dict
        The dictionary used for node-to-node transmissions.
    """
    
    def __init__(self, adj_mat, weights=None):
        """
        Class constructor.

        Parameters
        ----------
        adj_mat : ndarray
            The adjacency matrix describing the connectivity pattern of the
            network.
        weights : ndarray, optional
            The consensus matrix used by the network. If the argument is not
            specified, then a Metropolis-Hastings weight matrix is generated.
        """
        
        # store network attributes
        self.adj_mat, self.N = adj_mat, adj_mat.shape[0]
        
        # consensus matrix
        self.weights = weights if weights is not None else metropolis_hastings(self.adj_mat)
        
        # list of neighbors for each node
        self.neighbors = [list(np.where(self.adj_mat[i,:])[0]) for i in range(self.N)]
        self.degrees = [len(n) for n in self.neighbors]
        
        # buffer for node-to-node transmissions
        self.buffer = {}
    
    def send(self, sender, receiver, packet):
        """
        Node-to-node transmission (sender phase).
        
        This method simulates a node-to-node transmission by storing the packet
        to be communicated in the `buffer`. In particular, if :math:`i` is the
        sender and :math:`j` the receiver, then the packet is introduced in
        `buffer` with keyword :math:`(j,i)`.
        
        Note that older information (if any) in the `buffer` is overwritten 
        whenever `send` is called.

        Parameters
        ----------
        sender : int
            The index of the transmitting node.
        receiver : int
            The index of the recipient.
        packet : array_like
            The packet to ne communicated.
        """
        
        self.buffer[receiver, sender] = packet
    
    def receive(self, receiver, sender, default=0, destructive=True):
        """
        Node-to-node transmission (receiver phase).
        
        This method simulates the reception of a packet previously transmitted
        using the `send` method. In patricular, the method accesses the 
        packet in the `buffer` dictionary. If the packet is not present,
        a default value is returned.
        
        Reads from the `buffer` can be destructive, meaning that the packet
        is read and removed, which is the default, or not.

        Parameters
        ----------
        receiver : int
            The index of the recipient.
        sender : int
            The index of the transmitting node.
        default : array_like, optional
            The value returned when a packet from `sender` to `receiver` is not
            found in the `buffer`.
        destructive : bool, optional
            Specifies if the packet should be removed from the `buffer` after
            being read (which is the default) or not.

        Returns
        -------
        array_like
            The packet or a default value.
        """
        
        return self.buffer.pop((receiver, sender), default) if destructive \
               else self.buffer.get((receiver, sender), default)
    
    def broadcast(self, sender, packet):
        """
        Broadcast transmission.
        
        This method implements a broadcast transmission in which a node sends
        the same packet to all its neighbors. The packet is also transmitted
        to the node itself. The method is implemented using the `send` method.

        Parameters
        ----------
        sender : int
            The index of the transmitting node.
        packet : array_like
            The packet to ne communicated.
        """
        
        for r in self.neighbors[sender]+[sender]: self.send(sender, r, packet)
    
    def consensus(self, x, weights=None):
        """
        Consensus mixing.
        
        This method implements a consensus step over the network, mixing the 
        given nodes' states using the weight matrix of the network or a 
        different one.

        Parameters
        ----------
        x : array_like
            The nodes' local states in an array with the last dimension 
            indexing the nodes.
        weights : ndarray, optional
            The consensus weight matrix to be used instead of the one created
            at initialization.

        Returns
        -------
        y : ndarray
            The local states after a consensus step.
        """
        
        # consensus matrix
        weights = weights if weights is not None else self.weights
        
        # broadcast the states
        for i in range(self.N): self.broadcast(i, x[...,i])
        
        # perform consensus
        y = np.empty(x.shape)
        for i in range(self.N):
            y[...,i] = sum([weights[i,j]*self.receive(i, j) for j in self.neighbors[i]+[i]])
                    
        return y
    
    def max_consensus(self, x):
        """
        Max-consensus.
        
        This method implements a step of max-consensus, where each node selects
        the (element-wise) maximum between the packets received from its 
        neighbors and its own state. See [#]_ for a reference on 
        max-consensus.

        Parameters
        ----------
        x : array_like
            The nodes' local states in an array with the last dimension 
            indexing the nodes.

        Returns
        -------
        x : ndarray
            The local states after a max-consensus step.
        
        References
        ----------
        .. [#] F. Iutzeler, P. Ciblat, and J. Jakubowicz, "Analysis of 
               Max-Consensus Algorithms in Wireless Channels," IEEE 
               Transactions on Signal Processing, vol. 60, no. 11, pp. 
               6103–6107, Nov. 2012.
        """
        
        # broadcast the states
        for i in range(self.N): self.broadcast(i, x[...,i])
        
        # perform max consensus
        for i in range(self.N): x[...,i] = np.max([self.receive(i, j, -np.inf) for j in self.neighbors[i]+[i]], axis=0)
        
        return x


# -------- EXAMPLES

class NoisyNetwork(Network):
    """
    Network with Gaussian communication noise.
    
    Representation of a connected, undirected network, whose communication
    protocol is subject to additive white Gaussian noise. The network's
    transmission methods add normal noise to all packets (unless they are sent
    from a node to itself).
    """
    
    def __init__(self, adj_mat, noise_var, weights=None):
        """
        Class constructor from adjacency matrix.
        
        Parameters
        ----------
        adj_mat : ndarray
            Adjacency matrix of the given graph.
        n : int
            The size of local states.
        noise_var : float
            The white Gaussian noise variance.
        weights : ndarray, optional
            The edge weights of the graph. If None, Metropolis-Hasting weights
            are generated by default.
        
        Raises
        ------
        ValueError.
        """   
        
        # call the super-class constructor
        super().__init__(adj_mat, weights)        
        self.std = math.sqrt(noise_var)

    def send(self, sender, receiver, packet):
        
        # generate noise
        noise = self.std*ran.standard_normal() if sender != receiver else 0
        # send noisy packet
        self.buffer[receiver, sender] = packet + noise

class QuantizedNetwork(Network):
    """
    Network with quantized communications.
    
    Representation of a connected, undirected network, whose communications
    are quantized. The network's transmission methods quantize all packets 
    (unless they are sent from a node to itself).
    """
    
    def __init__(self, adj_mat, step, thresholds=None, weights=None):
        """
        Class constructor from adjacency matrix.
        
        This object simulates a network that uses a quantized communication
        protocol. The quantization is uniform with the specified step, and
        optionally saturates to the given lower and upper thresholds.
        
        Parameters
        ----------
        adj_mat : ndarray
            Adjacency matrix of the given graph.
        n : int
            The size of local states.
        step : float
            The step of the quantizer.
        thresholds : list, optional
            The lower and upper saturation thresholds of the quantizer.
        weights : ndarray, optional
            The edge weights of the graph. If None, Metropolis-Hasting weights
            are generated by default.
        """   
        
        # call the super-class constructor
        super().__init__(adj_mat, weights)
        
        # store the quantizer parameters (their validity is implicitly checked 
        # when calling utils.uniform_quantizer)
        self.step, self.thresholds = step, thresholds
        
    def send(self, sender, receiver, packet):
        
        # generate noise
        q = utils.uniform_quantizer(packet, self.step, self.thresholds) if sender != receiver else packet
        # send quantized packet
        self.buffer[receiver, sender] = q

class LossyNetwork(Network):
    """
    Network with random communication failures.
    
    Representation of a connected, undirected network, whose communication
    protocol is subject to packet losses. Packet sent from a node to another
    may be lost with a certain probability.
    """
    
    def __init__(self, adj_mat, loss_prob, weights=None): 
        
        # call the super-class constructor
        super().__init__(adj_mat, weights)
        
        # store the communication noise variance
        if loss_prob < 0 or loss_prob >= 1:
            raise ValueError("The packet loss probability should be in [0,1).")
        
        self.loss_prob = loss_prob
    
    def send(self, sender, receiver, packet):
        # the packet always arrives if the sender==receiver, or it arrives w.p. 1-loss_prob
        
        # try to send the packet
        if sender == receiver or ran.random() > self.loss_prob: self.buffer[receiver, sender] = packet


# -------- EXAMPLES

class DynamicNetwork(Network):
    """
    Time-varying network.
    
    This class creates a time-varying network from a list of network objects,
    and possibly a sampling time that specifies how often the network changes.
    """
    
    def __init__(self, nets, t_s=1):
        """
        Class constructor.

        Parameters
        ----------
        nets : list
            A list of the networks over time.
        t_s : float, optional
            The time after which the network changes to the next configuration.
        """
            
        # create time domain
        self.time = sets.T(t_s, t_min=0, t_max=len(nets)*t_s)
        # store networks
        self.nets, self.N = nets, len(nets)
    
    def _evaluate(self, method, t, *args, **kwargs):
        
        return getattr(self.nets[self.time.check_input(t)], method)(*args, **kwargs)
    
    send = partialmethod(_evaluate, "send")
    broadcast = partialmethod(_evaluate, "broadcast")
    consensus = partialmethod(_evaluate, "consensus")
    max_consensus = partialmethod(_evaluate, "max_consensus")

    def sample(self, t):
        """
        Sample the dynamic network.
        
        This method returns the network object that is active at time `t`.

        Parameters
        ----------
        t : float
            The time when the network should be sampled.

        Returns
        -------
        Network
            The sampled network.
        """
        
        return self.nets[self.time.check_input(t)]


#%% GRAPH GENERATION

def random_graph(N, radius):
    """
    Generate a random geometric graph.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    radius : float
        Radius of each node's neighborhood, must be in :math:`[0,1)`.
    
    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    
    Raises
    ------
    ValueError.
    
    Notes
    -----
    The function recursively generates random positions for the nodes on the
    :math:`[0,1] \\times [0,1]` square, and then builds the graph by setting 
    as neighbors each pair of nodes within a distance no larger than `radius`. 
    The process is repeated until the result is a connected graph. For this 
    reason, combinations of small `N` and `radius` *can yield 
    exceedingly long computation times*. If the computation does not succeed
    after `2500` iterations, an error is raised.
    """
    
    # check validity of arguments
    if N <= 0:
        raise ValueError("The number of nodes must be positive.")   
    if radius < 0 or radius > 1:
        raise ValueError("Radius must be a number in [0,1).")

    # (try to) cast N as integer
    N = int(N)
    
    # stopping conditions
    connected, max_iter = False, 2500
    
    i = 0
    while not connected and i < max_iter:
        
        # generate nodes' positions
        pos = ran.random((N, 2)) # random positions on the plane
        
        dist_x = np.repeat(pos[:,0], N).reshape((N,N)) \
               - np.repeat(pos[:,0], N).reshape((N,N)).T
        dist_y = np.repeat(pos[:,1], N).reshape((N,N)) \
               - np.repeat(pos[:,1], N).reshape((N,N)).T
        
        dist = np.square(dist_x) + np.square(dist_y) # nodes' distances
        
        # build adjacency matrix
        adj_mat = (dist < radius).astype(float)
        # and remove self-loops
        adj_mat = adj_mat - np.diag(np.diag(adj_mat))

        # check if graph is connected        
        connected = is_connected(adj_mat)
        i += 1
    
    # return graph
    if connected:
        return adj_mat - np.diag(np.diag(adj_mat))
    else:
        raise ValueError("Unable to generate the graph, try a larger radius.")

def erdos_renyi(N, prob):
    """
    Generate a random Erdos-Renyi graph.

    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    prob : float
        The probability of adding an edge between any two nodes.

    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    
    Raises
    ------
    ValueError.
    """
    
    # check validity of arguments
    if N <= 0:
        raise ValueError("The number of nodes must be positive.")   
    if prob <= 0 or prob > 1:
        raise ValueError("`prob` must be a number in (0,1].")

    # (try to) cast N as integer
    N = int(N)
    
    # stopping conditions
    connected, max_iter = False, 2500
    
    i = 0
    while not connected and i < max_iter:
        
        # generate the edges
        adj_mat = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i+1, N):
                                
                if ran.random() <= prob: adj_mat[i,j], adj_mat[j,i] = 1, 1

        # check if graph is connected        
        connected = is_connected(adj_mat)
        i += 1
    
    # return graph
    if connected:
        return adj_mat - np.diag(np.diag(adj_mat))
    else:
        raise ValueError("Unable to generate the graph, try a larger probability.")


def circulant_graph(N, num_conn):
    """
    Generate a circulant graph.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    num_conn : int
        Number of neighbors on each side of a node.
    
    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    
    Notes
    -----
    If `num_conn` is larger than `N` / 2 a complete graph is returned.
    """
    
    # check validity of arguments
    if N <= 0 or num_conn <= 0:
        raise ValueError("The number of nodes and connections must be positive.")   
        
    # (try to) cast N and num_conn as integers
    N = int(N)
    num_conn = int(num_conn)
    
    adj_mat = np.zeros((N,N))
    
    # create base vector
    if 2*num_conn+1 >= N:
        vec = np.hstack((0, np.ones(N-1)))
    else:
        vec = np.hstack((0, np.ones(num_conn), 
                         np.zeros(N-2*num_conn-1), np.ones(num_conn)))
    
    for i in range(N):
        
        adj_mat[i,:] = np.roll(vec, i)
            
    return adj_mat


def circle_graph(N):
    """
    Generate a circle graph.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    
    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    
    See Also
    --------
    circulant_graph : Circulant graph generator
    """
    
    return circulant_graph(N, 1)


def complete_graph(N):
    """
    Generate a complete graph.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    
    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    
    See Also
    --------
    circulant_graph : Circulant graph generator
    """
    
    return circulant_graph(N, N)
    

def star_graph(N):
    """
    Generate a star graph.
    
    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    
    Returns
    -------
    adj_mat : ndarray
        Adjacency matrix of the generated graph.
    """
    
    # check validity of arguments
    if N <= 0:
        raise ValueError("The number of nodes must be positive.")   
        
    # (try to) cast N as integer
    N = int(N)
    
    adj_mat = np.zeros((N,N))
    adj_mat[0,1:N] = 1
    adj_mat[1:N,0] = 1
            
    return adj_mat


#%% GRAPH MATRICES

def metropolis_hastings(adj_mat):
    """ 
    Compute a consensus matrix based on the Metropolis-Hastings rule.
    
    The Metropolis-Hastings rule generates a matrix :math:`W` with off-diagonal
    elements equal to:
        
        .. math:: w_{ij} = \\frac{1}{1 + \max\{ d_i, d_j \}}
        
    where :math:`i` is a node index and :math:`j \\neq i` the index of one of 
    its neighbors, and :math:`d_i`, :math:`d_j` are their respective degrees. 
    The diagonal elements are assigned as:
    
        .. math:: w_{ii} = 1 - \sum_{j \in \mathcal{N}_i} w_{ij}
    
    to guarantee double stochasticity.
    
    Parameters
    ----------
    adj_mat : ndarray
        Adjacency matrix describing the graph.
    
    Returns
    -------
    mh_mat : ndarray
        Metropolois-Hastings consensus matrix.
    """
    
    # retrieve num. of nodes
    N = adj_mat.shape[0]
    # nodes' degrees
    deg = np.sum(adj_mat,axis=0)
    
    mh_mat = np.zeros((N,N))
    
    # add the off-diagonal elements
    for i in range(N):
        for j in np.where(adj_mat[i,:])[0]:
            
            mh_mat[i,j] = 1 / (1 + np.max((deg[i],deg[j])))
    
    # add the diagonal elements
    mh_mat = mh_mat + np.eye(N) - np.diag(np.sum(mh_mat,axis=0))
    
    return mh_mat

def incidence_matrix(adj_mat, n=1):
    """
    Build the incidence matrix.
    
    The edges :math:`e = (i,j)` are ordered with :math:`i \leq j`, so that 
    in the :math:`e`-th column the :math:`i`-th element is :math:`1` and
    the :math:`j`-th is :math:`-1` (the remaining are of course :math:`0`).
    
    Parameters
    ----------
    adj_mat : ndarray
        Adjacency matrix describing the graph.
    n : int, optional
        Size of the local states.

    Returns
    -------
    incid_mat : ndarray
        The incidence matrix.
    """
    
    # return upper triangular portion [not to count edges twice]
    adj_mat = np.triu(adj_mat)

    N = adj_mat.shape[0] # num of nodes 
    num_edges = int(np.sum(adj_mat)) # num of edges
    
    incid_mat = np.zeros((num_edges,N))
    
    # index of current edge
    e = 0
    # fill in the incidence matrix
    for i in range(N):
        for j in np.where(adj_mat[i,:])[0]:
            
            incid_mat[e,i] = 1
            incid_mat[e,j] = -1
            
            e += 1
            
    return np.kron(incid_mat, np.eye(n))


#%% TOOLS

def is_connected(adj_mat):
    """
    Verify if a graph is connected.
    
    Parameters
    ----------
    adj_mat : ndarray
        Adjacency matrix describing the graph.

    Returns
    -------
    bool
        True if the graph is connected, False otherwise.
    
    Notes
    -----
    The connectedness of the graph is checked by verifying whether the
    :math:`N`-th power of the adjacency matrix plus the identity is a full 
    matrix (no zero elements), with :math:`N` the number of nodes.
    """
    
    N = adj_mat.shape[0]
    
    return np.all(la.matrix_power(adj_mat + np.eye(N), N))