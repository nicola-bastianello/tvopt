#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility tools.
"""


import sys
import math
import numbers
import numpy as np
from numpy import linalg as la
from numpy.random import default_rng
from scipy.linalg import block_diag

ran = default_rng() # random generator


#%% LINEAR ALGEBRA

def square_norm(x):
    """
    Compute the square norm of the given vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The square norm.
    
    Notes
    -----
    The function reshapes `x` to a column vector, so it does not correctly
    handle n-dimensional arrays. For n-dim arrays use `numpy.linalg.norm`.
    """
    
    x = np.reshape(x, (-1,1))
    
    return x.T.dot(x).item()

def norm(x):
    """
    Compute the norm of the given vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The square norm.
    
    See Also
    --------
    square_norm : Square norm
    
    Notes
    -----
    The function reshapes `x` to a column vector, so it does not correctly
    handle n-dimensional arrays. For n-dim arrays use `numpy.linalg.norm`.
    """
    
    return math.sqrt(square_norm(x))

def normalize(x):
    """
    Normalize a vector to unit vector.

    Parameters
    ----------
    x : array_like
        The vector array.

    Returns
    -------
    ndarray
        The normalized vector.
    
    Notes
    -----
    The function reshapes `x` to a column vector, so it does not correctly
    handle n-dimensional arrays. For n-dim arrays use `numpy.linalg.norm`.
    """
    
    return x / norm(x)

def is_square(mat):
    """
    Check if the matrix is 2-D and square.
    
    Parameters
    ----------
    mat : ndarray
        The given matrix.

    Returns
    -------
    bool
        True if the matrix is 2-D and square, False otherwise.
    """
    
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]

def is_stochastic(mat, row=True, col=True):
    """
    Verify if a given matrix is row, column or doubly stochastic.
    
    Parameters
    ----------
    mat : ndarray
        The given matrix.
    row : bool, optional
        Check for row stochasticity, default True.
    col : bool, optional
        Check for column stochasticity, default True.

    Returns
    -------
    bool
        True if the matrix is stochastic (row, column or doubly, as specified
        by the arguments).
    
    Raises
    ------
    ValueError
        If neither `row` nor `col` are True.
    """
    
    if not row and not col:
        raise ValueError("At least one of row and col must be True.")
    
    # verify the matrix is non-negative
    if np.any(mat < 0):
        return False
    
    # verify row stochasticity
    ones = np.ones((mat.shape[0],1))
    is_row = np.allclose(mat.dot(ones) - ones, np.zeros((mat.shape[0],1)))
    
    # verify column stochasticity
    ones = np.ones((mat.shape[1],1))
    is_col = np.allclose(mat.T.dot(ones) - ones, np.zeros((mat.shape[1],1)))
    
    if row and col: return is_row and is_col
    elif row and not col: return is_row
    else: return is_col

def orthonormal_matrix(dim):
    """
    Generate a random orthonormal matrix.
    
    This function generates uniformly distributed random orthonormal matrices
    using Householder reflections (see Section 7 of `this paper
    <https://arxiv.org/pdf/math-ph/0609050.pdf>`_).
    
    Parameters
    ----------
    dim : int
        Size of the matrix.

    Returns
    -------
    orth_mat : ndarray
        The random orthonormal matrix.
    
    Raises
    ------
    ValueError
        For invalid `dim`.
    """
    
    # check validity of dim and cast to integer
    if int(dim) < 1:
        raise ValueError("The dimension must be at least one.")
    dim = int(dim)
    
    # array for the orthonormal matrix
    orth_mat = np.eye(dim)
        
    for i in range(0,dim):
        
        rnd_vec = ran.standard_normal((dim-i,1)) # generate a random vector
        # first vector in standard basis of R^{dim-i}
        strd_1 = np.vstack((1,np.zeros((dim-i-1,1))))
        # temporary vector for computing HH reflection
        tmp_vec = (rnd_vec + np.sign(rnd_vec[0])*strd_1) \
                / la.norm(rnd_vec + np.sign(rnd_vec[0])*strd_1)
        
        # Householder reflection
        hh_refl = -np.sign(rnd_vec[0])*(np.eye(dim-i)-2*tmp_vec.dot(tmp_vec.T))
        # block diagonal with identity and Householder reflection
        hh_tilde = block_diag(np.eye(i),hh_refl)
        
        # update the orthogonal matrix
        orth_mat = orth_mat.dot(hh_tilde.T)
    
    return orth_mat

def random_matrix(eigs):
    """
    Generate a random matrix.
    
    The matrix is generated as
    
        .. math:: M = O \mathrm{diag}\{ \lambda_i \} O^\\top
    
    where :math:`O` is a random orthonormal matrix and :math:`\lambda_i` are
    the specified eigenvalues.
    
    Parameters
    ----------
    eigs : array-like
        The list of eigenvalues for the matrix.

    Returns
    -------
    ndarray
        The random positive semi-definite matrix.
    
    See Also
    --------
    orthonormal_matrix : Orthonormal matrix generator.
    """
    
    eigs = np.ravel(eigs) # flatten eigs in an array

    # generate an orthonormal matrix
    orth_mat = orthonormal_matrix(eigs.size)
    
    return orth_mat.dot(np.diag(eigs).dot(orth_mat.T))

def positive_semidefinite_matrix(dim, max_eig=None, min_eig=None):
    """
    Generate a random positive semi-definite matrix.
    
    The matrix is generated as
    
        .. math:: M = O \mathrm{diag}\{ \lambda_i \} O^\\top
    
    where :math:`O` is a random orthonormal matrix and :math:`\lambda_i` are
    random eigenvalues uniformly drawn between `min_eig` and `max_eig`. If 
    `dim` is larger than or equal to two, `min_eig` and `max_eig` are included 
    in the eigenvalues list.
    
    Parameters
    ----------
    dim : int
        Size of the matrix.
    eigs : array-like, optional
        The list of eigenvalues for the matrix; if None, the eigenvalues are
        uniformly drawn from :math:`[10^{-2}, 10^2]`.

    Returns
    -------
    ndarray
        The random positive semi-definite matrix.
    
    Raises
    ------
    ValueError.
    
    See Also
    --------
    random_matrix : Random matrix generator.
    """
    
    max_eig = 1e2 if max_eig is None else max_eig
    min_eig = 1e-2 if min_eig is None else min_eig
    
    if max_eig < 0 or min_eig < 0:
        raise ValueError("`max_eig` and `min_eig` must be non-negative.")
    
    if dim < 2:
        eigs = [max_eig]
    if dim < 3:
        eigs = [max_eig, min_eig]
    else:
        eigs = np.hstack((max_eig, (max_eig-min_eig)*ran.random(dim-2)+min_eig, min_eig))
    
    return random_matrix(eigs)

def solve(a, b):
    
    if is_scalar(a) or np.size(a) == 1: return b / a
    else: return la.solve(a, b)
    

#%% PERFORMANCE METRICS

def fpr(s, ord=2):
    """
    Fixed point residual.
    
    This function computes the fixed point residual of a signal `s`,
    that is
    
        .. math:: \{ \| s^\ell - s^{\ell-1} \|_i \}_{\ell \in \mathbb{N}}.
    
    Different norm orders can be used, that can be specified using the
    `numpy.linalg.norm` argument `ord`.

    Parameters
    ----------
    s : array_like
        The signal, with the last dimension indexing time.
    ord : optional
        Norm order, see `numpy.linalg.norm`.

    Returns
    -------
    ndarray
        The fixed point residual.
    """
    
    if isinstance(s, list): s = np.stack(s, axis=-1)
    
    return dist(s[...,:-1], s[...,1:], ord=ord)


def dist(s, r, ord=2):
    """
    Distance of a signal from a reference.
    
    This function computes the distance of a signal `s` from a reference `r`.
    The reference can be either constant or a signal itself.
    Different norm orders can be used, that can be specified using the
    `numpy.linalg.norm` argument `ord`.

    Parameters
    ----------
    s : array_like
        The signal, with the last dimension indexing time.
    r : array_like
        The reference, either a single array or a signal with the last 
        dimension indexing time.
    ord : optional
        Norm order, see `numpy.linalg.norm`.

    Raises
    ------
    ValueError
        For incompatible dimensions of signal and reference.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if isinstance(s, list): s = np.stack(s, axis=-1)
    if isinstance(r, list): r = np.stack(r, axis=-1)
    
    # repeat x_opt if it is a static quantity
    if r.shape == s.shape[:-1]:
        r = np.repeat(r[...,np.newaxis], s.shape[-1], axis=-1)
    if s.ndim == 2 and r.shape[0] == s.shape[0] and r.shape[0] != s.shape[0]:
        r = np.repeat(r, s.shape[-1], axis=-1)
    
    # compute the norm if the shapes are compatible
    if r.shape == s.shape:
        # handle scalar signals
        if s.shape[:-1] == ():
            return np.abs(s - r)
        else:
            d = np.zeros(s.shape[-1])
            for l in range(s.shape[-1]):
                d[l] = la.norm((s[...,l] - r[...,l]).flatten(), ord=ord)
            return d
    else:
        raise ValueError("Incompatible shapes of `s` and `r` {}, {}.".format(s.shape, r.shape))

def regret(f, s, r=None):
    r"""
    Cost over time or regret.
    
    This function computes the cost evaluated using `f` incurred by an 
    approximate minimizer `s`
    
        .. math:: \{ \frac{1}{\ell} \sum_{j = 1}^\ell f(s^j) \}_{\ell \in \mathbb{N}}
    
    or, if a reference `r` is specified, then the function computes the regret
    
        .. math:: \{ \frac{1}{\ell} \sum_{j = 1}^\ell f(s^j) - f(r^j) \}_{\ell \in \mathbb{N}}
    
    where `r` is either a constant array or a signal.    

    Parameters
    ----------
    f : costs.Cost
        The cost to evaluate in the signal.
    s : array_like
        The sequence of approximate minimizers.
    r : array_like, optional
        The reference, either a single array or a signal with the last 
        dimension indexing time.

    Returns
    -------
    ndarray
        The sequence of cost evaluations or regret.
    """
    
    if isinstance(s, list): s = np.stack(s, axis=-1)
    if isinstance(r, list): r = np.stack(r, axis=-1)

        
    reg = np.zeros(s.shape[-1])
    for k in range(s.shape[-1]):
        
        # dynamic cost
        if f.is_dynamic:
            t = k*f.time.t_s
            reg[k] = f.function(s[...,k], t)
            if r is not None: reg[k] -= f.function(r[...,k], t)
        # static cost
        else:
            reg[k] = f.function(s[...,k])
            if r is not None: reg[k] -= f.function(r)
    
    # compute cumulative regret
    cum_reg = np.cumsum(reg)
    return np.array([cum_reg[k]/(k+1) for k in range(s.shape[-1])])


#%% VARIOUS  

def uniform_quantizer(x, step, thresholds=None):
    """
    Function to perform uniform quantization.
    
    The function applies the uniform quantization
    
    .. math:: q(x) = \Delta \operatorname{floor}
              \left( \\frac{x}{\Delta} + \\frac{1}{2} \\right)
    
    where :math:`\Delta` is the given step. Moreover, a saturation to
    upper and lower thresholds is peformed if given as argument.

    Parameters
    ----------
    x : ndarray
        The array to be quantized.
    step : float
        The step of the quantizer.
    thresholds : list, optional
        The upper and lower saturation thresholds.
    
    Returns
    -------
    ndarray
        The quantized array.
    """
    
    # check validity of quantizer parameters
    if step < 0:
        raise ValueError("The step must be positive.")
    if thresholds is not None:
        if len(thresholds) != 2:
            raise ValueError("Both upper and lower thresholds must be given.")
        if thresholds[0] >= thresholds[1]:
            raise ValueError("Invalid thresholds.")
    
    # perform quantization using the given step
    q = step*np.floor(x/step + 0.5)
    
    # saturate to the thresholds, if given
    if thresholds is not None:
        q[np.where(q < thresholds[0])] = thresholds[0]
        q[np.where(q > thresholds[1])] = thresholds[1]
    
    return q

def is_scalar(c):
    """
    Check if scalar.
    """
    
    return isinstance(c, numbers.Number)

def soft_thresholding(x, penalty):
    r"""
    Soft-thresholding.
    
    The function computes the element-wise soft-trhesholding defined as
    
        .. math:: \operatorname{sign}(x) \max\{ |x| - \rho, 0 \}
    
    where :math:`\rho` is a positive penalty parameter.
    
    Parameters
    ----------
    x : array_like
        Where to evaluate the soft-thresholding.
    penalty : float
        The positive penalty parameter :math:`\rho`.

    Returns
    -------
    ndarray
        The soft-thresolding of `x`.
    """
    
    return np.multiply(np.sign(x), np.maximum(np.absolute(x) - penalty, 0))

def bisection_method(f, a, b, tol=1e-5):
    """
    Minimize using the bisection method.
    
    This function minimizes a function `f` using the bisection method, stopping
    when :math:`a - b \leq t` for some threshold :math:`t`.

    Parameters
    ----------
    f
        The scalar function to be minimized.
    a : float
        The lower bound of the initial interval.
    b : float
        the upper bound of the initial interval.
    tol : float, optional
        The stopping condition, defaults to 1e-5.

    Returns
    -------
    x : float
        The approximate minimizer.
    """
    
    x = (a + b) / 2
    
    stop = False
    while not stop:
        
        f_x = f(x)
        
        if f(a)*f_x < 0: b = x
        elif f(b)*f_x < 0: a = x
        
        x = (a + b) / 2
        
        stop = abs(b - a) <= tol
    
    return x

def print_progress(i, num_iter, bar_length=80, decimals=2):
    """
    Print the progresso to command line.
    
    Parameters
    ----------
    i : int
        Current iteration.
    num_iter : int
        Total number of iterations.
    bar_length : int, optional
        Length of progress bar.
    decimals : int, optional
        Decimal places of the progress percent.
    
    Notes
    -----
    Adapted from `here 
    <https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a>`_.
    """
    
    # completion percent
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (i / num_iter))
    
    # progress bar
    filled_length = int(round(bar_length * i / num_iter))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    
    # print progress
    sys.stdout.write("\rCompleted |%s| %s%s" % (bar, percents, "%")),
    if i == num_iter:
        sys.stdout.write("\n")
    sys.stdout.flush()

def initialize_trajectory(x_0, shape, num_iter):
    
    x = np.zeros(shape + (num_iter+1,))
    x[...,0] = x_0
    
    return x