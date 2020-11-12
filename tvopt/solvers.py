#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solvers.
"""

import numpy as np
from numpy import linalg as la

from tvopt import utils, costs


#%% FIRST ORDER METHODS

def subgradient(problem, x_0=0, num_iter=100, tol=None):
    r"""
    Sub-gradient method.
    
    This function implements the sub-gradient method
    
        .. math:: \pmb{x}^{\ell+1} = \pmb{x}^\ell - \alpha^\ell \tilde{\nabla} f(\pmb{x}^\ell)
    
    where :math:`\tilde{\nabla} f(\pmb{x}^\ell) \in \partial f(\pmb{x}^\ell)`
    is a sub-differential and :math:`\alpha^\ell = 1 / (\ell + 1)`.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """

    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        x_old = x
        
        # choose step-size
        step = 1 / (l+1)
        
        # sub-gradient step
        x = x - step*f.gradient(x)
            
        if stop(x, x_old, tol): break
    
    return x

def gradient(problem, step, x_0=0, num_iter=100, tol=None):
    r"""
    Gradient method.
    
    This function implements the gradient method
    
        .. math:: \pmb{x}^{\ell+1} = \pmb{x}^\ell - \alpha \nabla f(\pmb{x}^\ell)
    
    for a given step-size :math:`\alpha > 0`.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`.
    step : float
        The algorithm's step-size.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
    
    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        x_old = x
        
        # gradient step
        x = x - step*f.gradient(x)
        
        if stop(x, x_old, tol): break
    
    return x

def backtracking_gradient(problem, r=0.2, c=0.5, x_0=0, num_iter=100, tol=None):
    r"""
    Gradient method with backtracking line search.
    
    This function implements the gradient method
    
        .. math:: \pmb{x}^{\ell+1} = \pmb{x}^\ell - \alpha^\ell \nabla f(\pmb{x}^\ell)
    
    where :math:`\alpha^\ell` is chosen via a backtracking line search. In 
    particular, at each iteration we start with :math:`\alpha^\ell = 1` and,
    while
    
        .. math:: f(\pmb{x}^\ell - \alpha^\ell \nabla f(\pmb{x}^\ell)) >
                  f(\pmb{x}^\ell) - c \alpha^\ell \| \nabla f(\pmb{x}^\ell) \|^2
    
    we set :math:`\alpha^\ell = r \alpha^\ell` until a suitable step is found.
    
    Note that the backtracking line search does not stop until a suitable
    step-size  si found; this means that large `r` parameters may result in
    big computation times.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`.
    r : float, optional
        The value by which a candidate step-size is multiplied if it does not
        satisfy the descent condition. `r` should be in :math:`(0,1)`.
    c : float, optional
        The parameter defining the descent condition that a candidate step
        must satisfy.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
        
    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        # candidate descent direction
        grad = f.gradient(x)
        
        # backtracking line search
        step, end = 1, False
        
        x_old, f_old = x, f.function(x)
        while not end:
            
            # candidate point and cost evaluation
            x = x_old - step*grad
            f_new = f.function(x)
            
            if f_new > f_old - c*step*la.norm(grad)**2: step *= r
            else: end = True
            
        if stop(x, x_old, tol): break
    
    return x

def ppa(problem, penalty, x_0=0, num_iter=100, tol=None):
    r"""
    Proximal point algorithm (PPA).
    
    This function implements the proximal point algorithm
    
        .. math:: \pmb{x}^{\ell+1} = \operatorname{prox}_{\rho f}(\pmb{x}^\ell)
    
    where :math:`\rho > 0` is the penalty parameter and we recall that
    
        .. math:: \operatorname{prox}_{\rho f}(\pmb{x}) =
                  \operatorname{arg\,min}_{\pmb{y}} \left\{ f(\pmb{y}) + \frac{1}{2\rho} \| \pmb{y} - \pmb{x} \|^2 \right\}.
    
    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`.
    penalty : float
        The penalty parameter for the proximal evaluation.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
                
    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        # proximal step
        x_old = x
        x = f.proximal(x, penalty)
        
        if stop(x, x_old, tol): break
    
    return x


# -------- COMPOSITE OPTIMIZATION

def fbs(problem, step, rel=1, x_0=0, num_iter=100, tol=None):
    r"""
    Forward-backward splitting (FBS).
    
    This function implements the forward-backward splitting (a.k.a. proximal 
    gradient method) to solve the composite problem
    
        .. math:: \min_{\pmb{x}} \{ f(\pmb{x}) + g(\pmb{x}) \}.
    
    The algorithm is characterized by the update:
    
        .. math:: \pmb{x}^{\ell+1} = (1-\alpha) \pmb{x}^\ell + 
                  \alpha \operatorname{prox}_{\rho g}(\pmb{x}^\ell - \rho \nabla f(\pmb{x}^\ell))
    
    where :math:`\rho > 0` is the step-size and :math:`\alpha \in (0,1]` is the
    relaxation constant.
    
    Parameters
    ----------
    problem : dict
        Problem dictionary defining the costs :math:`f` and :math:`g`.
    step : float
        The algorithm's step-size.
    rel : float, optional
        The relaxation constant.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
                
    f, g = problem["f"], problem["g"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        x_old = x
        
        # gradient and proximal steps
        x = (1-rel)*x + rel*g.proximal(x - step*f.gradient(x), step)
        
        if stop(x, x_old, tol): break
    
    return x

def prs(problem, penalty, rel=1, x_0=0, num_iter=100, tol=None):
    r"""
    Peaceman-Rachford splitting (PRS).
    
    This function implements the Peaceman-Rachford splitting to solve the 
    composite problem
    
        .. math:: \min_{\pmb{x}} \{ f(\pmb{x}) + g(\pmb{x}) \}.
    
    The algorithm is characterized by the updates:
        
        .. math:: \begin{align} \pmb{x}^\ell &= \operatorname{prox}_{\rho f}(\pmb{z}^\ell) \\
                  \pmb{y}^\ell &= \operatorname{prox}_{\rho g}(2 \pmb{x}^\ell - \pmb{z}^\ell) \\
                  \pmb{z}^{\ell+1} &= \pmb{z}^\ell + 2 \alpha (\pmb{y}^\ell - \pmb{x}^\ell)
                  \end{align}
    
    where :math:`\rho > 0` is the penalty and :math:`\alpha \in (0,1]` is the
    relaxation constant.
    
    Parameters
    ----------
    problem : dict
        Problem dictionary defining the costs :math:`f` and :math:`g`.
    penalty : float
        The algorithm's penalty parameter.
    rel : float, optional
        The relaxation constant.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
    
    f, g = problem["f"], problem["g"]

    z = np.zeros(f.dom.shape)
    z[...] = x_0
    
    
    for l in range(num_iter):
        
        z_old = z
        
        # proximal step (f)
        x = f.proximal(z, penalty)
        
        # proximal step (g) and auxiliary update
        z = z + 2*rel*(g.proximal(2*x - z, penalty) - x)
        
        if stop(z, z_old, tol): break
    
    return x, z


#%% SECOND ORDER METHODS

def newton(problem, r=0.2, c=0.5, x_0=0, num_iter=100, tol=None):
    r"""
    Newton method with backtracking line search.
    
    This function implements the Newton method
    
        .. math:: \pmb{x}^{\ell+1} = \pmb{x}^\ell - \alpha^\ell \nabla^2 f(\pmb{x}^\ell)^{-1} \nabla f(\pmb{x}^\ell)
    
    where :math:`\alpha^\ell` is chosen via a backtracking line search. In 
    particular, at each iteration we start with :math:`\alpha^\ell = 1` and,
    while
    
        .. math:: f(\pmb{x}^\ell - \alpha^\ell \nabla^2 f(\pmb{x}^\ell)^{-1} \nabla f(\pmb{x}^\ell)) >
                  f(\pmb{x}^\ell) - c \alpha^\ell \| \nabla f(\pmb{x}^\ell) \|^2
    
    we set :math:`\alpha^\ell = r \alpha^\ell` until a suitable step is found.
    
    Note that the backtracking line search does not stop until a suitable
    step-size  si found; this means that large `r` parameters may result in
    big computation times.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`.
    r : float, optional
        The value by which a candidate step-size is multiplied if it does not
        satisfy the descent condition. `r` should be in :math:`(0,1)`.
    c : float, optional
        The parameter defining the descent condition that a candidate step
        must satisfy.
    x_0 : array_like, optional
        The initial condition. This can be either an ndarray of suitable size,
        or a scalar. If it is a scalar then the same initial value is used for
        all components of :math:`\pmb{x}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        stopping condition :math:`\| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate solution after `num_iter` iterations.
    """
                
    f = problem["f"]
    x = np.zeros(f.dom.shape)
    x[...] = x_0
    
    
    for l in range(num_iter):
        
        # candidate descent direction
        grad, hess = f.gradient(x), f.hessian(x)
        d = utils.solve(hess, grad)
        
        # backtracking line search
        step, end = 1, False
        
        x_old, f_old = x, f.function(x)
        while not end:
            
            # candidate point and cost evaluation
            x = x_old - step*d
            f_new = f.function(x)
            
            if f_new > f_old - c*step*np.dot(np.transpose(grad), d): step *= r
            else: end = True
        
        if stop(x, x_old, tol): break
    
    return x


#%% DUAL-BASED METHODS

def dual_ascent(problem, penalty, w_0=0, num_iter=100, tol=None):
    r"""
    Dual ascent.
    
    This function implements the dual ascent to solve the constrained problem
    
        .. math:: \min_{\pmb{x}} f(\pmb{x}) \ \text{s.t.} \ \pmb{A} \pmb{x} = \pmb{c}.
    
    The algorithm is characterized by the updates:    
    
        .. math:: \begin{align}
                  \pmb{x}^\ell &= \operatorname{arg\,min}_{\pmb{x}} \left\{ f(\pmb{x}) - \langle \pmb{w}^\ell, \pmb{A} \pmb{x} \rangle \right\} \\
                  \pmb{w}^{\ell+1} &= \pmb{w}^\ell - \rho (\pmb{A} \pmb{x}^\ell - \pmb{c})
                  \end{align}
        
    for a given penalty :math:`\rho > 0`.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`, and the constraints :math:`A` and :math:`c`.
    penalty : float
        The algorithm's penalty.
    w_0 : array_like, optional
        The dual initial condition. This can be either an ndarray of suitable 
        size, or a scalar. If it is a scalar then the same initial value is 
        used for all components of :math:`\pmb{w}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        dual stopping condition 
        :math:`\| \pmb{w}^{\ell+1} - \pmb{w}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate primal solution after `num_iter` iterations.
    w : ndarray
        The approximate dual solution after `num_iter` iterations.
    """
    
    f, A, c = problem["f"], problem["A"], problem["c"]
    
    x = np.zeros(f.dom.shape)
    w = np.zeros((A.shape[0],1)) # dual variables
    w[...] = w_0
    
    # auxiliary function for primal update
    p = costs.Linear(- A.T.dot(w))
    
    
    for l in range(num_iter):
               
        # primal update
        x = newton({"f":f+p}, x_0=x, tol=1e-5)
        
        w_old = w
        
        # dual update
        w = w - penalty*(A.dot(x) - c)
        
        if stop(w, w_old, tol): break
        
        # update auxliary function
        p.a = - A.T.dot(w)
    
    return x, w

def mm(problem, penalty, w_0=0, num_iter=100, tol=None):
    r"""
    Method of multipliers (MM).
    
    This function implements the method of multipliers to solve the constrained
    problem
    
        .. math:: \min_{\pmb{x}} f(\pmb{x}) \ \text{s.t.} \ \pmb{A} \pmb{x} = \pmb{c}.
    
    The algorithm is characterized by the updates:    
    
        .. math:: \begin{align}
                  \pmb{x}^\ell &= \operatorname{arg\,min}_{\pmb{x}} \left\{ f(\pmb{x}) 
                  - \langle \pmb{w}^\ell, \pmb{A} \pmb{x} \rangle + \frac{\rho}{2} \| \pmb{A} \pmb{x} - \pmb{c} \|^2 \right\} \\
                  \pmb{w}^{\ell+1} &= \pmb{w}^\ell - \rho (\pmb{A} \pmb{x}^\ell - \pmb{c})
                  \end{align}
        
    for a given penalty :math:`\rho > 0`.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the cost :math:`f`, and the constraints :math:`A` and :math:`c`.
    penalty : float
        The algorithm's penalty.
    w_0 : array_like, optional
        The dual initial condition. This can be either an ndarray of suitable 
        size, or a scalar. If it is a scalar then the same initial value is 
        used for all components of :math:`\pmb{w}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        dual stopping condition 
        :math:`\| \pmb{w}^{\ell+1} - \pmb{w}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate primal solution after `num_iter` iterations.
    w : ndarray
        The approximate dual solution after `num_iter` iterations.
    """
    
    f, A, c = problem["f"], problem["A"], problem["c"]
    
    x = np.zeros(f.dom.shape)
    w = np.zeros((A.shape[0],1)) # dual variables
    w[...] = w_0
    
    # auxiliary function for primal update
    p = costs.Quadratic(penalty*A.T.dot(A), -penalty*A.T.dot(c + w/penalty))
    
    
    for l in range(num_iter):
                
        # primal update
        x = newton({"f":f+p}, x_0=x, tol=1e-5)
        
        w_old = w
                
        # dual update
        w = w - penalty*(A.dot(x) - c)
        
        if stop(w, w_old, tol): break
        
        # update auxliary function
        p.b = -penalty*A.T.dot(c + w/penalty)
    
    return x, w

def dual_fbs(problem, penalty, rel=1, w_0=0, num_iter=100, tol=None):
    r"""
    Dual forward-backward splitting.
    
    This function implements the dual FBS to solve the constrained problem
    
        .. math:: \begin{align}
                  &\min_{\pmb{x},\pmb{y}} \left\{ f(\pmb{x}) + g(\pmb{y}) \right\} \\
                  &\text{s.t.} \ \pmb{A} \pmb{x} +  \pmb{B} \pmb{y} = \pmb{c}.
                  \end{align}
    
    The algorithm is characterized by the updates:    
    
        .. math:: \begin{align}
                  \pmb{x}^\ell &= \operatorname{arg\,min}_{\pmb{x}} \left\{ f(\pmb{x}) - \langle \pmb{w}, \pmb{A} \pmb{x} \rangle \right\} \\
                  \pmb{u}^\ell &= \pmb{w}^\ell - \rho (\pmb{A} \pmb{x}^\ell - \pmb{c}) \\
                  \pmb{y}^\ell &= \operatorname{arg\,min}_{\pmb{y}} \left\{ g(\pmb{y}) 
                  - \langle \pmb{u}^\ell, \pmb{B} \pmb{y} \rangle + \frac{\rho}{2} \| \pmb{B} \pmb{y} \|^2 \right\} \\
                  \pmb{w}^{\ell+1} &= (1-\alpha) \pmb{w}^\ell + \alpha (\pmb{u}^\ell - \rho \pmb{B} \pmb{y}^\ell)
                  \end{align}
        
    for a given penalty :math:`\rho > 0` and :math:`\alpha \in (0,1]` is the
    relaxation constant.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the costs :math:`f` and :math:`g`, and the constraints
        :math:`A`, :math:`B` and :math:`c`.
    penalty : float
        The algorithm's penalty.
    rel : float, optional
        The relaxation constant.
    w_0 : array_like, optional
        The dual initial condition. This can be either an ndarray of suitable 
        size, or a scalar. If it is a scalar then the same initial value is 
        used for all components of :math:`\pmb{w}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        dual stopping condition 
        :math:`\| \pmb{w}^{\ell+1} - \pmb{w}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate primal solution :math:`\pmb{x}` after `num_iter` iterations.
    y : ndarray
        The approximate primal solution :math:`\pmb{y}` after `num_iter` iterations.
    w : ndarray
        The approximate dual solution after `num_iter` iterations.
    """
    
    f, g, A, B, c = problem["f"], problem["g"], problem["A"], problem["B"], problem["c"]
    
    x, y = np.zeros(f.dom.shape), np.zeros(g.dom.shape)
    w = np.zeros((A.shape[0],1)) # dual variables
    w[...] = w_0
    
    # auxiliary functions for primal updates
    p_x = costs.Linear(- A.T.dot(w))
    p_y = costs.Quadratic(penalty*B.T.dot(B), np.zeros(g.dom.shape))
    
    
    for l in range(num_iter):
                
        # primal update (x)
        x = newton({"f":f+p_x}, x_0=x, tol=1e-5)
        
        # auxiliary update
        u = w - penalty*(A.dot(x) - c)
        
        # update auxiliary function (y)
        p_y.b = -penalty*B.T.dot(u/penalty)
        
        # primal update (y)
        y = newton({"f":g+p_y}, x_0=y, tol=1e-5)
        
        w_old = w        
        
        # dual update
        w = (1-rel)*w + rel*(u - penalty*B.dot(y))
        
        if stop(w, w_old, tol): break
        
        # update auxliary function (x)
        p_x.a = - A.T.dot(w)
    
    return x, y, w

def admm(problem, penalty, rel=1, w_0=0, num_iter=100, tol=None):
    r"""
    Alternating direction method of multipliers (ADMM).
    
    This function implements the ADMM to solve the constrained problem
    
        .. math:: \begin{align}
                  &\min_{\pmb{x},\pmb{y}} \left\{ f(\pmb{x}) + g(\pmb{y}) \right\} \\
                  &\text{s.t.} \ \pmb{A} \pmb{x} +  \pmb{B} \pmb{y} = \pmb{c}.
                  \end{align}
    
    The algorithm is characterized by the updates:    
    
        .. math:: \begin{align}
                  \pmb{x}^\ell &= \operatorname{arg\,min}_{\pmb{x}} \left\{ f(\pmb{x}) 
                  - \langle \pmb{z}^\ell, \pmb{A} \pmb{x} \rangle + \frac{\rho}{2} \| \pmb{A} \pmb{x} - \pmb{c} \|^2 \right\} \\
                  \pmb{w}^\ell &= \pmb{z}^\ell - \rho (\pmb{A} \pmb{x}^\ell - \pmb{c}) \\
                  \pmb{y}^\ell &= \operatorname{arg\,min}_{\pmb{y}} \left\{ g(\pmb{y}) 
                  - \langle 2 \pmb{w}^\ell - \pmb{z}^\ell, \pmb{B} \pmb{y} \rangle + \frac{\rho}{2} \| \pmb{B} \pmb{y} \|^2 \right\} \\
                  \pmb{u}^\ell &= 2 \pmb{w}^\ell - \pmb{z}^\ell - \rho \pmb{B} \pmb{y}^\ell \\
                  \pmb{z}^{\ell+1} &= \pmb{z}^\ell + 2 \alpha (\pmb{u}^\ell - \pmb{w}^\ell)
                  \end{align}
        
    for a given penalty :math:`\rho > 0` and :math:`\alpha \in (0,1]` is the
    relaxation constant.

    Parameters
    ----------
    problem : dict
        Problem dictionary defining the costs :math:`f` and :math:`g`, and the constraints
        :math:`A`, :math:`B` and :math:`c`.
    penalty : float
        The algorithm's penalty.
    rel : float, optional
        The relaxation constant.
    w_0 : array_like, optional
        The dual initial condition. This can be either an ndarray of suitable 
        size, or a scalar. If it is a scalar then the same initial value is 
        used for all components of :math:`\pmb{w}`.
    num_iter : int, optional
        The number of iterations to be performed.
    tol : float, optional
        If given, this argument specifies the tolerance :math:`t` in the 
        dual stopping condition 
        :math:`\| \pmb{w}^{\ell+1} - \pmb{w}^\ell \| \leq t`.

    Returns
    -------
    x : ndarray
        The approximate primal solution :math:`\pmb{x}` after `num_iter` iterations.
    y : ndarray
        The approximate primal solution :math:`\pmb{y}` after `num_iter` iterations.
    w : ndarray
        The approximate dual solution after `num_iter` iterations.
    """
    
    f, g, A, B, c = problem["f"], problem["g"], problem["A"], problem["B"], problem["c"]
    
    x, y = np.zeros(f.dom.shape), np.zeros(g.dom.shape)
    z = np.zeros((A.shape[0],1)) # auxiliary variables
    z[...] = w_0
    
    # auxiliary functions for primal updates
    p_x = costs.Quadratic(penalty*A.T.dot(A), -penalty*A.T.dot(c + z/penalty))
    p_y = costs.Quadratic(penalty*B.T.dot(B), np.zeros(g.dom.shape))
    
    
    for l in range(num_iter):
                
        # primal update (x)
        x = newton({"f":f+p_x}, x_0=x, tol=1e-5)
        
        # dual update
        w = z - penalty*(A.dot(x) - c)
        
        # update auxiliary function (y)
        p_y.b = -penalty*B.T.dot((2*w - z)/penalty)
        
        # primal update (y)
        y = newton({"f":g+p_y}, x_0=y, tol=1e-5)
    
        z_old = z
        
        # auxiliary update
        z = (1-2*rel)*z + 2*rel*(w - penalty*B.dot(y))
        
        if stop(z, z_old, tol): break
        
        # update auxliary function (x)
        p_x.b = -penalty*A.T.dot(c + z/penalty)
    
    return x, y, z


#%% UTILITY FUNCTIONS

def stop(x, x_old, tol=None):
    """
    Stopping condition.
    
    This function checks the stopping condition
    
        .. math:: \| \pmb{x}^{\ell+1} - \pmb{x}^\ell \| \leq t
    
    if `t` is specified.

    Parameters
    ----------
    x : ndarray
        The current iterate.
    x_old : ndarray
        The previous iterate.
    tol : float, optional
        The tolerance in the stopping condition.

    Returns
    -------
    bool
        True if `tol` is given and the stopping condition is verified, False
        otherwise.
    """

    return tol is not None and la.norm(x - x_old) <= tol