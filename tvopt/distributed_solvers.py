#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distributed solvers.
"""


import numpy as np
ran = np.random.default_rng() # random generator

from tvopt import costs, solvers


#%% CONSENSUS METHODS

def average_consensus(net, x_0, num_iter=100):
    """
    Average consensus.
    
    Compute the average consensus over the network `net` with initial
    states `x_0`.

    Parameters
    ----------
    net : networks.Network
        The network describing the multi-agent system.
    x_0 : ndarray
        The initial states in a ndarray, with the last dimension indexing the 
        nodes.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    """
    
    x = x_0

    
    for l in range(num_iter): x = net.consensus(x)
    
    return x

def ratio_consensus(net, x_0, num_iter=100):
    """
    Ratio consensus.
    
    Compute the average consensus over the network `net` with initial
    states `x_0` using the ratio consensus protocol.

    Parameters
    ----------
    net : networks.Network
        The network describing the multi-agent system.
    x_0 : ndarray
        The initial states in a ndarray, with the last dimension indexing the 
        nodes.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    """
    
    x = x_0
    y, z = x, np.ones(x_0.shape) # auxiliary variables
    
    
    for l in range(num_iter):
        
        y_old, z_old = np.copy(y), np.copy(z)
        y[...], z[...] = 0, 0
        
        for i in range(net.N):
            for j in net.neighbors[i]+[i]:
                # y
                net.send(j, i, y_old[...,j]/(1+net.degrees[j]))
                y[...,i] += net.receive(i, j)
                # z
                net.send(j, i, z_old[...,j]/(1+net.degrees[j]))
                z[...,i] += net.receive(i, j)
        
        # perform consensus
        x = np.divide(y, z)
    
    return x

def gossip_consensus(net, x_0, num_iter=100, q=0.5):
    """
    Average consensus.
    
    Compute the average consensus over the network `net` with initial
    states `x_0` using the symmetric gossip protocol.

    Parameters
    ----------
    net : networks.Network
        The network describing the multi-agent system.
    x_0 : ndarray
        The initial states in a ndarray, with the last dimension indexing the 
        nodes.
    num_iter : int, optional
        The number of iterations to be performed.
    q : float, optional
        The weight used in the convex combination of the nodes that 
        communicate at each iteration.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    """
    
    x = x_0
    
    
    for l in range(num_iter):
                
        # choose nodes for gossip communication
        i = ran.integers(0, net.N)
        j = ran.choice(net.neighbors[i])
        
        # perform gossip communication
        net.send(i, j, x[...,i]), net.send(j, i, x[...,j])
        
        # update local states
        x[...,i] = q*x[...,i] + (1 - q)*net.receive(i, j)
        x[...,j] = q*x[...,j] + (1 - q)*net.receive(j, i)
    
    return x

def max_consensus(net, x_0, num_iter=100):
    """
    Max consensus.
    
    Compute the maximum of the nodes' states `x_0`.

    Parameters
    ----------
    net : networks.Network
        The network describing the multi-agent system.
    x_0 : ndarray
        The initial states in a ndarray, with the last dimension indexing the 
        nodes.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    """
    
    x = x_0
    
    
    for l in range(num_iter): x = net.max_consensus(x)
    
    return x


#%% PROXIMAL GRADIENT METHODS

def dpgm(problem, step, x_0=0, num_iter=100):
    r"""
    Distributed proximal gradient method (DPGM).
    
    This function implements the DPGM algorithm proposed in [#]_ (see also 
    [#]_ for the gradient-only version). The algorithm is characterized by 
    the following updates
    
    .. math:: \begin{align}
              \pmb{y}^\ell &= \pmb{W} \pmb{x}^\ell - \alpha \nabla f(\pmb{x}^\ell) \\
              \pmb{x}^{\ell+1} &= \operatorname{prox}_{\alpha g}(\pmb{y}^\ell)
              \end{align}
    
    for :math:`\ell = 0, 1, \ldots`. The algorithm is guaranteed to converge
    to a neighborhood of the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the costs describing the (possibly composite) problem. The 
        dictionary should contain :math:`f` and the network, and optionally
        :math:`g`.
    step : float
        The step-size.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [#] Bastianello, N., Ajalloeian, A., & Dall'Anese, E. (2020). 
           Distributed and Inexact Proximal Gradient Method for Online Convex 
           Optimization. arXiv preprint arXiv:2001.00870.
    .. [#] Yuan, K., Ling, Q., & Yin, W. (2016). On the convergence of 
           decentralized gradient descent. SIAM Journal on Optimization, 
           26(3), 1835-1854.
    """
    
    # unpack problem data
    f, g, net = problem["f"], problem.get("g", None), problem["network"]
    is_composite = g is not None
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    # run algorithm
    for l in range(num_iter):
        
        # gradient step
        y = net.consensus(x, net.weights) - step*f.gradient(x)

        # proximal step
        if is_composite: x = g.proximal(y, penalty=step)
        else: x = y
    
    return x

def pg_extra(problem, step, x_0=0, num_iter=100):
    r"""
    Proximal gradient exact first-order algorithm (PG-EXTRA).
    
    This function implements the PG-EXTRA algorithm proposed in [#]_ (see also 
    [#]_ for the gradient-only version, EXTRA). The algorithm is characterized 
    by the following updates
    
    .. math:: \begin{align}
              \pmb{y}^\ell &= \pmb{y}^{\ell-1} + \pmb{W} \pmb{x}^\ell
                           - \tilde{\pmb{W}} \pmb{x}^{\ell-1} 
                           - \alpha (\nabla f(\pmb{x}^\ell) - \nabla f(\pmb{x}^{\ell-1})) \\
              \pmb{x}^{\ell+1} &= \operatorname{prox}_{\alpha g}(\pmb{y}^\ell)
              \end{align}
    
    for :math:`\ell = 0, 1, \ldots`, where
    :math:`\tilde{\pmb{W}} = (\pmb{I} + \pmb{W}) /2`. The 
    algorithm is guaranteed to converge to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the costs describing the (possibly composite) problem. The 
        dictionary should contain :math:`f` and the network, and optionally
        :math:`g`.
    step : float
        The step-size.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [#] Shi, W., Ling, Q., Wu, G., & Yin, W. (2015). A proximal gradient 
            algorithm for decentralized composite optimization. IEEE 
            Transactions on Signal Processing, 63(22), 6013-6023.
    .. [#] Shi, W., Ling, Q., Wu, G., & Yin, W. (2015). Extra: An exact 
            first-order algorithm for decentralized consensus optimization. 
            SIAM Journal on Optimization, 25(2), 944-966.
    """
    
    # unpack problem data
    f, g, net = problem["f"], problem.get("g", None), problem["network"]
    is_composite = g is not None
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    x_old = x
    
    # initialization
    x_cons = net.consensus(x, net.weights)
    grad = f.gradient(x)
    y = x_cons - step*grad
    
    if is_composite: x = g.proximal(y, penalty=step)
    else: x = y
    
    
    # run algorithm
    for l in range(1,num_iter):
        
        # store old consensus step and gradient evaluation
        x_cons_old, grad_old = np.copy(x_cons), np.copy(grad)
        
        # perform consensus step and gradient evaluation
        x_cons = net.consensus(x, net.weights)
        grad = f.gradient(x)
        
        # gradient tracking step
        y = y + x_cons - 0.5*(x_old + x_cons_old) - step*(grad - grad_old)
        x_old = x
        
        # proximal step
        if is_composite: x = g.proximal(y, penalty=step)
        else: x = y
    
    return x

def nids(problem, step, x_0=0, num_iter=100):
    r"""
    Network InDependent Step-size algorithm (NIDS).
    
    This function implements the NIDS algorithm proposed in [#]_. The algorithm
    is characterized by the following updates
    
    .. math:: \begin{align}
              \pmb{y}^\ell &= \pmb{y}^{\ell-1} - \pmb{x}^\ell
              - \tilde{\pmb{W}} (2 \pmb{x}^\ell - \pmb{x}^{\ell-1}
              - \operatorname{diag}(\pmb{\alpha}) (\nabla f(\pmb{x}^\ell) 
              - \nabla f(\pmb{x}^{\ell-1}))) \\
              \pmb{x}^{\ell+1} &= \operatorname{prox}_{\pmb{\alpha} g}(\pmb{y}^\ell)
              \end{align}
    
    for :math:`\ell = 0, 1, \ldots`, where :math:`\pmb{\alpha}` is a column
    vector containing the independent step-sizes of the nodes, and
    
    .. math:: \tilde{\pmb{W}} = \pmb{I} 
              + c \operatorname{diag}(\pmb{\alpha}) (\pmb{W} - \pmb{I})
    
    with :math:`c = 0.5 / \max_i \{ \alpha_i \}`. The algorithm is guaranteed
    to converge to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the costs describing the (possibly composite) problem. The 
        dictionary should contain :math:`f` and the network, and optionally
        :math:`g`.
    step : float or list
        A common step-size or a list of local step-sizes, one for each node.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [#] Li, Z., Shi, W., & Yan, M. (2019). A decentralized proximal-gradient
            method with network independent step-sizes and separated convergence
            rates. IEEE Transactions on Signal Processing, 67(17), 4494-4506.
    """

    # unpack problem data
    f, g, net = problem["f"], problem.get("g", None), problem["network"]
    is_composite = g is not None
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    x_old = x
    
    # uncoordinated step-sizes
    step = step if isinstance(step, list) else [step for _ in range(net.N)]
    
    # compute consensus matrix
    common_step = 0.5 / np.max(step)
    
    weights = np.eye(net.N) + common_step*np.diag(step).dot(net.weights - np.eye(net.N))
    
    # initialization
    grad = f.gradient(x)
    y = x - np.stack([step[i]*grad[...,i] for i in range(net.N)], axis=-1)
    
    if is_composite: x = g.proximal(y, penalty=step)
    else: x = y
    
    
    # run algorithm
    for l in range(1,num_iter):
        
        # store old gradient and compute new one
        grad_old = np.copy(grad)
        grad = f.gradient(x)
        
        # gradient tracking step
        y = y - x + net.consensus(2*x - x_old \
          - np.stack([step[i]*(grad[...,i]-grad_old[...,i]) for i in range(net.N)], axis=-1), weights=weights)
        x_old = x
        
        # proximal step
        if is_composite: x = g.proximal(y, penalty=step)
        else: x = y
    
    return x

def prox_ed(problem, step, x_0=0, num_iter=100):
    r"""
    Proximal exact diffusion (Prox-ED).
    
    This function implements the Prox-ED algorithm [#]_. The algorithm is 
    characterized by the following updates
    
    .. math:: \begin{align}
              \pmb{y}^\ell &= \pmb{x}^\ell - \alpha \nabla f(\pmb{x}^\ell) \\
              \pmb{u}^\ell &= \pmb{z}^{\ell-1} + \pmb{y}^\ell - \pmb{y}^{\ell-1} \\
              \pmb{z}^\ell &= \tilde{\pmb{W}} \pmb{u}^\ell \\
              \pmb{x}^{\ell+1} &= \operatorname{prox}_{\alpha g}(\pmb{z}^\ell)
              \end{align}
    
    for :math:`\ell = 0, 1, \ldots`, where
    :math:`\tilde{\pmb{W}} = (\pmb{I} + \pmb{W}) /2`. The 
    algorithm is guaranteed to converge to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the costs describing the (possibly composite) problem. The 
        dictionary should contain :math:`f` and the network, and optionally
        :math:`g`.
    step : float
        The step-size.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [#] S. A. Alghunaim, E. Ryu, K. Yuan, and A. H. Sayed, "Decentralized 
           Proximal Gradient Algorithms with Linear Convergence Rates," IEEE 
           Transactions on Automatic Control, 2020.
    """
    
    # unpack problem data
    f, g, net = problem["f"], problem.get("g", None), problem["network"]
    is_composite = g is not None
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    # consensus matrix
    weights = 0.5*(np.eye(net.N) + net.weights)
    
    # initialization
    y = x - step*f.gradient(x)
    u = np.copy(y)
    z = net.consensus(u, weights=weights)
    
    if is_composite: x = g.proximal(z, penalty=step)
    else: x = z
    
    
    # run algorithm
    for l in range(1,num_iter):
        
        # store old y variables
        y_old = np.copy(y)
        
        # gradient updates
        y = x - step*f.gradient(x)
        u = z + y - y_old
        
        # communication step
        z = net.consensus(u, weights=weights)
        
        # proximal step
        if is_composite: x = g.proximal(z, penalty=step)
        else: x = z
    
    return x

def prox_aac(problem, step, x_0=0, num_iter=100, consensus_steps=[True, True]):
    r"""
    Proximal adapt-and-combine (Prox-AAC).
    
    This function implements the Prox-AAC algorithm (see [1]_ for the gradient
    only version). The algorithm is characterized by the 
    following updates
    
    .. math:: \pmb{z}^\ell = \pmb{W}_1 \pmb{x}^\ell
    
    .. math:: \pmb{y}^\ell = 
                    \pmb{z}^\ell - \alpha \nabla f(\pmb{z}^\ell)

    .. math:: \pmb{x}^{\ell+1} = 
                      \operatorname{prox}_{\alpha g}(\pmb{W}_2 \pmb{y}^\ell)
    
    for :math:`\ell = 0, 1, \ldots`, where :math:`\pmb{W}_1` and
    :math:`\pmb{W}_2` are doubly stochastic matrices (or the identity). 
    
    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the costs describing the (possibly composite) problem. The 
        dictionary should contain :math:`f` and the network, and optionally
        :math:`g`.
    step : float or list
        A common step-size or a list of local step-sizes, one for each node.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.
    consensus_steps : list
        A list specifying which consensus steps to perform; the list must have
        two elements that can be interpreted as bools.
    
    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [1] Chen, J., & Sayed, A. H. (2013). Distributed Pareto optimization via
            diffusion strategies. IEEE Journal of Selected Topics in Signal 
            Processing, 7(2), 205-220.
    """    
    
    # unpack problem data
    f, g, net = problem["f"], problem.get("g", None), problem["network"]
    is_composite = g is not None
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    # uncoordinated step-sizes
    step = step if isinstance(step, list) else [step for _ in range(net.N)]
    
    
    # run algorithm
    for l in range(num_iter):
        
        # first communication step
        if consensus_steps[0]: z = net.consensus(x)
        else: z = x
        
        # gradient step
        grad = f.gradient(z)
        y = z - np.stack([step[i]*grad[...,i] for i in range(net.N)], axis=-1)

        # second communication step
        if consensus_steps[1]: u = net.consensus(y)
        else: u = y
        
        # proximal step
        if is_composite: x = g.proximal(u, penalty=step)
        else: x = u
    
    return x


#%% GRADIENT METHODS

def aug_dgm(problem, step, x_0=0, num_iter=100):
    r"""
    Augmented distributed gradient method (Aug-DGM).
    
    This function implements the Aug-DGM algorithm (see [#]_). The algorithm is
    characterized by the following updates
    
    .. math:: \begin{align}
              \pmb{y}^\ell &= \pmb{W} \left( \pmb{y}^{\ell-1} 
                            + \nabla f(\pmb{x}^\ell) - \nabla f(\pmb{x}^{\ell-1}) \right) \\
              \pmb{x}^{\ell+1} &= \pmb{W} \left( \pmb{x}^\ell - \pmb{A} \pmb{y}^\ell \right)
              \end{align}
    
    for :math:`\ell = 0, 1, \ldots` where :math:`\pmb{A}` is a diagonal matrix 
    of uncoordinated step-sizes. The algorithm is guaranteed to converge 
    to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the cost describing the problem.
    step : float
        A common step-size or a list of local step-sizes, one for each node.
    x_0 : ndarray, optional
        The initial states of the nodes. This can be either an ndarray
        of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components of the states.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    
    References
    ----------
    .. [#] J. Xu, S. Zhu, Y. C. Soh, and L. Xie, "Augmented distributed 
           gradient methods for multi-agent optimization under uncoordinated 
           constant stepsizes," in 2015 54th IEEE Conference on Decision and 
           Control (CDC), Osaka, Japan, Dec. 2015, pp. 2055–2060.
    """

    # unpack problem data
    f, net = problem["f"], problem["network"]
    
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    # uncoordinated step-sizes
    step = step if isinstance(step, list) else [step for _ in range(net.N)]
    
    # initialization
    grad = f.gradient(x)
    y = grad
    x = net.consensus(x - np.stack([step[i]*y[...,i] \
               for i in range(net.N)], axis=-1), net.weights)
    
    
    # run algorithm
    for l in range(1,num_iter):
        
        # old and new gradient evaluations
        grad_old = np.copy(grad)
        grad = f.gradient(x)
        
        # gradient tracking step
        y = net.consensus(y + grad - grad_old, net.weights)
        x = net.consensus(x - np.stack([step[i]*y[...,i] \
                     for i in range(net.N)], axis=-1), net.weights)
    
    return x


#%% DUAL-BASED METHODS

def dual_ascent(problem, step, w_0=0, num_iter=100):
    r"""
    Distributed dual ascent a.k.a. dual decomposition (DD).
    
    This function implements the DD algorithm [#]_. The algorithm is 
    characterized by the following updates
    
    .. math:: \pmb{x}^\ell = \operatorname{arg\,min}_{\pmb{x}} \left\{
        f(\pmb{x}) - \langle (\pmb{I} - \pmb{W}) \pmb{x}, \pmb{w}^\ell 
        \rangle\right\}
    
    .. math:: \pmb{w}^{\ell+1} = \pmb{w}^\ell 
                                - \alpha (\pmb{I} - \pmb{W}) \pmb{x}^\ell
    
    for :math:`\ell = 0, 1, \ldots`, where :math:`\pmb{w}` is the vector of
    Lagrange multipliers. The algorithm is guaranteed to converge to the 
    optimal solution.

    Parameters
    ----------
    A dictionary containing the network describing the multi-agent system
        and the cost describing the problem.
    step : float
        The step-size.
    w_0 : ndarray, optional
        The initial value of the dual nodes' states. This can be either an
        ndarray of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    w : ndarray
        The dual variables of the nodes after `num_iter` iterations.
    
    References
    ----------
    .. [#] Simonetto, A. (2018). Dual Prediction–Correction Methods for
            Linearly Constrained Time-Varying Convex Programs. IEEE Transactions
            on Automatic Control, 64(8), 3355-3361.
    """
    
    # unpack problem data
    f, net = problem["f"], problem["network"]
    
    x = np.zeros(f.dom.shape)
    w = np.zeros(f.dom.shape) # dual variables
    w[...] = w_0

    
    # run algorithm
    for l in range(num_iter):
        
        # dual communication step
        w_comm = w - net.consensus(w, net.weights)
        
        # primal update
        for i in range(net.N):
            x[...,i] = solvers.newton({"f":f.costs[i]+costs.Linear(-w_comm[...,i])}, x_0=x[...,i], tol=1e-5)
        
        # primal communication step
        x_comm = x - net.consensus(x, net.weights)
        
        # dual update
        w = w - step*x_comm
    
    return x, w

def admm(problem, penalty, rel, w_0=0, num_iter=100):
    r"""
    Distributed relaxed alternating direction method of multipliers (ADMM).
    
    This function implements the distributed ADMM, see [#]_ and references 
    therein. The algorithm is characterized by the following updates
    
    .. math:: x_i^\ell = \operatorname{prox}_{f_i / (\rho d_i)}
                                    ([\pmb{A}^\top z^\ell]_i / (\rho d_i))
    
    .. math:: z_{ij}^{\ell+1} = (1-\alpha) z_{ij}^\ell - \alpha z_{ji}^\ell 
                              + 2 \alpha \rho x_j^\ell
    
    for :math:`\ell = 0, 1, \ldots`, where :math:`d_i` is node :math:`i`'s 
    degree, :math:`\rho` and :math:`\alpha` are the penalty and relaxation
    parameters, and :math:`\pmb{A}` is the arc incidence matrix. The algorithm 
    is guaranteed to converge to the optimal solution.

    Parameters
    ----------
    problem : dict
        A dictionary containing the network describing the multi-agent system
        and the cost describing the problem.
    penalty : float
        The penalty parameter :math:`\rho` of the algorithm (convergence is
        guaranteed for any positive value).
    rel : float
        The relaxation parameter :math:`\alpha` of the algorithm (convergence 
        is guaranteed for values in :math:`(0,1)`).
    w_0 : ndarray, optional
        The initial value of the dual nodes' states. This can be either an
        ndarray of suitable size with the last dimension indexing the nodes, or
        a scalar. If it is a scalar then the same initial value is used for
        all components.
    num_iter : int, optional
        The number of iterations to be performed.

    Returns
    -------
    x : ndarray
        The nodes' states after `num_iter` iterations.
    w : ndarray
        The dual variables of the nodes after `num_iter` iterations.
    
    References
    ----------
    .. [#] N. Bastianello, R. Carli, L. Schenato, and M. Todescato, 
           "Asynchronous Distributed Optimization over Lossy Networks via 
           Relaxed ADMM: Stability and Linear Convergence," IEEE Transactions 
           on Automatic Control.
    """
    
    # unpack problem data
    f, net = problem["f"], problem["network"]
    
    x = np.zeros(f.dom.shape)
    
    # parameters for local proximal evaluations
    penalty_i = [1 / (penalty*net.degrees[i]) for i in range(net.N)]
    
    # initialize arc variables
    z = {}
    for i in range(net.N):
        for j in net.neighbors[i]:
            z[i,j] = np.zeros(f.dom.shape[:-1])
    
    
    for l in range(num_iter):
        
        for i in range(net.N):
            
            # primal update
            x[...,i] = f.proximal(sum([z[i,j] for j in net.neighbors[i]]) * penalty_i[i], penalty=penalty_i[i], i=i)
                
            # communication step
            for j in net.neighbors[i]: net.send(i, j, -z[i,j] + 2*penalty*x[...,i])
        
        # update auxiliary variables
        for i in range(net.N):
            for j in net.neighbors[i]:
                
                z[i,j] = (1 - rel)*z[i,j] + rel*net.receive(i, j)
    
    
    return x, z