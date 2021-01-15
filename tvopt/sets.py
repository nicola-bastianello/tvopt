#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Set template and examples.
"""


import numpy as np
from numpy import linalg as la

from tvopt import utils


#%% SET TEMPLATES

class Set():
    r"""
    Template for a set.
    
    This class defines a non-empty, closed, convex set in
    :math:`\mathbb{R}^{n_1 \times n_2 \times \ldots}`. These objects are 
    defined by a `contains` method (to check if an input belongs to the set)
    and a `projection` method.
    
    Sets can be translated and scaled (via the respective methods). The 
    `contains` method can also be accessed via the built-in `in` operator.
    Using `+` it is possible to intersect sets.
    
    Attributes
    ----------
    shape : tuple
        The dimensions of the underlying space.
    ndim : int
        The number of dimensions of the underlying space.
    size : int
        The product of each dimension's size.
    """
    
    def __init__(self, *dims):
        
        # scalars
        if len(dims) == 0: # or all([d == 1 for d in dims]):
            self.shape, self.ndim, self.size = (), 0, 1
        # n-dimensional vectors
        else:

            if any([d < 1 for d in dims]):
                raise ValueError("The domain dimension(s) must be positive.")
            
            self.shape = tuple([int(d) for d in dims])
            self.ndim, self.size = len([None for d in self.shape]), np.prod(self.shape)
    
    def check_input(self, x):
        """
        Check dimension of input.
        
        This method verifies if the argument `x` belong to the space underlying
        the set, possibly reshaping it. If it is not compatible or cannot be
        reshaped (using numpy's broadcasting rules), and exception is raised.

        Parameters
        ----------
        x : array_like
            The input to be checked.

        Returns
        -------
        ndarray
            The (possibly reshaped) input if it is compatible with the space.
        """
        
        if self.ndim == 0 and utils.is_scalar(x): return x
        else: return np.reshape(x, self.shape)
    
    def contains(self, x):
        """
        Check if the input belongs to the set.
        """
        
        raise NotImplementedError()
    
    def projection(self, x, *args, **kwargs):
        """
        Project the input onto the set.
        """
        
        raise NotImplementedError()
    
    def translate(self, x):
        """
        Translate the set.
        """
        
        return TranslatedSet(self, x)
    
    def scale(self, c):
        """
        Scale the set.
        """
        
        return ScaledSet(self, c)
    
    def __eq__(self, other):
        
        if not isinstance(other, Set):
            raise TypeError("Equality not implemented for 'Set' and '{}'.".format(type(other).__name__))
        
        if self is other: return True
        else: return self.shape == other.shape
    
    def __ne__(self, other):
        
        return not self.__eq__(other)
    
    def __contains__(self, x):
        
        return self.contains(x)
    
    def __add__(self, other):
        
        return IntersectionSet(self, other)
    
    def __radd__(self, other):
        
        if other == 0: return self
        else: return self.__add__(other)


# -------- CLASSES for OPERATIONS

class TranslatedSet(Set):
    r"""
    Translated set.
    
    Given a set :math:`\mathbb{S}` and a vector :math:`t`, this class defines
    
        .. math:: \{ x + t \ \forall x \in \mathbb{S} \}.
    """
                
    def __init__(self, s, t):
        """
        Class constructor.
        
        Parameters
        ----------
        s : Set
            The set to be translated.
        t : array_like
            The translation vector.
        """
        
        super().__init__(*s.shape)
        
        self.s, self.t = s, super().check_input(t)
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return (x - self.t) in self.s
    
    def projection(self, x, *args, **kwargs):
        
        return self.t + self.s.projection(x - self.t, *args, **kwargs)

class ScaledSet(Set):
    r"""
    Scaled set.
    
    Given a set :math:`\mathbb{S}` and a scalar :math:`c`, this class defines
    
        .. math:: \{ c x \ \forall x \in \mathbb{S} \}.
    """

    def __init__(self, s, c):
        """
        Class constructor.

        Parameters
        ----------
        s : Set
            The set to be scaled.
        c : float
            The scaling constant.

        Raises
        ------
        TypeError.
        ValueError.
        """
        
        if not np.isscalar(c):
            raise TypeError("Unsupported scaling by (non-scalar) '{}'.".format(type(c).__name__))
        
        if c == 0:
            raise ValueError("`c` must be non-zero.")
        
        super().__init__(*s.shape)
        self.s, self.c = s, c
    
    def contains(self, x):
        
        return (x/self.c) in self.s
    
    def projection(self, x, *args, **kwargs):
        
        return self.c*(self.s.projection(x/self.c, *args, **kwargs))

class IntersectionSet(Set):
    r"""
    Intersection of sets.
    
    Given the sets :math:`\mathbb{S}_i`, :math:`i = 1, \ldots, N` this class
    implements
    
        .. math:: \bigcap_{i = 1}^N \mathbb{S}_i.
    """
    
    def __init__(self, *sets):
        """

        Parameters
        ----------
        *sets
            The sets to be intersected.

        Raises
        ------
        TypeError.
        ValueError.
        """
        
        # check validity
        for s in sets:
            if not isinstance(s, Set):
                raise TypeError("Unsupported intersection for 'Set' and '{}'.".format(type(s).__name__))
        
        super().__init__(*sets[0].shape)
        if any([s.shape != self.shape for s in sets[1:]]):
            raise ValueError("Incompatible sets dimensions.")

        # sets and their number
        self.sets, self.N = sets, len(sets)
    
    def contains(self, x):
        # parsimonious check, stops as soon as it finds one outside
        
        # stop if x is outside one of the sets
        for i in range(self.N):
            if x not in self.sets[i]: return False
        
        return True # all the checks passed
    
    def projection(self, x, *args, **kwargs):
        """
        Projection onto the intersection.
        
        This method returns an approximate projection onto the intersection of
        sets, computed using the method of alternating projections.

        See Also
        --------
        alternating_projections : method of alternating projection
        """
        
        return alternating_projections(self.sets, self.check_input(x), **kwargs)


#%% EXAMPLES

class R(Set):
    r"""
    The underlying space.
    
    This class implements the underlying space 
    :math:`\mathbb{R}^{n_1 \times n_2 \times \ldots}`.
    """
    
    def contains(self, x):
        
        # if check succeeds x is in the space
        self.check_input(x)
        return True
    
    def projection(self, x):
        # it is the identity, with a check on x's shape

        return self.check_input(x)

class NonnegativeOrthant(Set):
    r"""
    Non-negative orthant.
    
    This class implements:
    
        .. math:: \{ x \in \mathbb{R}^n \ | \ x \geq 0 \}
    
    where :math:`x \geq 0` if :math:`x` is component-wise non-negative.
    """
    
    def __init__(self, n):
        """
        Class constructor.

        Parameters
        ----------
        n : int
            The underlying space dimension.
        """

        super().__init__(n, 1)
    
    def contains(self, x):
        
        return np.all(self.check_input(x) >= 0)
    
    def projection(self, x):
        
        x = self.check_input(x)
        
        if x in self: return x
        else: return np.maximum(x, 0)
        
class Ball(Set):
    r"""
    Ball set.
    
    This class implements:
        
        .. math:: \{ x \in \mathbb{R}^n \ | \ \| x - c \| \leq r \}
    
    for a center :math:`c` and radius :math:`r > 0`.
    """
    
    def __init__(self, center, radius):
        """
        Class constructor.

        Parameters
        ----------
        radius : float
            Radius of the ball.
        center : array_like, optional
            Center of the ball, defaults to the origin.
        
        Raises
        ------
        ValueError.
        """
        
        center = np.reshape(center, (-1, 1))
        super().__init__(*center.shape)
        
        if radius <= 0: raise ValueError("The radius must be positive.")
        
        self.radius, self.center = radius, center
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return utils.norm(x - self.center) <= self.radius
    
    def projection(self, x):
        
        x = self.check_input(x)
        
        if x in self: return x
        else: return self.center + self.radius*utils.normalize(x - self.center)

class Box(Set):
    r"""
    Box set.
    
    This class implements:
        
        .. math:: \{ x \in \mathbb{R}^n \ | \ l \leq x \leq u \}
    
    with bounds :math:`l, u` either scalar (applied element-wise) or vectors.
    """
    
    def __init__(self, l, u, n=1):
        """
        Class constructor.
        
        The lower and upper bound can be either (1) scalars, in which case they
        are applied element-wise and the argument `n` specifies the dimension
        of the underlying space, or (2) vectors, which implecitly specify the
        dimension of the space.

        Parameters
        ----------
        l : array_like
            Scalar or vector lower bound.
        u : array_like
            Scalar or vector upper bound.
        n : int, optional
            The dimension of the underlying space that needs to specified if
            scalar lower and upper bounds are used. Defaults to 1.

        Raises
        ------
        ValueError.
        """
        
        super().__init__(n, 1)
        
        # check lower bounds
        if np.isscalar(l): l = l*np.ones(self.shape)
        else: l = np.reshape(l, self.shape)
        # check upper bounds
        if np.isscalar(u): u = u*np.ones(self.shape)
        else: u = np.reshape(u, self.shape)

        if np.any(l > u):
            raise ValueError("`l` must be component-wise smaller than `u`.")
            
        # store arguments
        self.l, self.u = l, u
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return np.all(self.l <= x) and np.all(x <= self.u)
    
    def projection(self, x):
        
        x = self.check_input(x)
        
        if x in self: return x
        else: return np.maximum(np.minimum(x, self.u), self.l)

class AffineSet(Set):
    r"""
    Affine set.
    
    This class implements:
        
        .. math:: \{ x \in \mathbb{R}^n \ | \ A x = b \}
    
    for given matrix :math:`A \in \mathbb{R}^{m \times n}` and vector
    :math:`b \in \mathbb{R}^{m}`.
    """
    
    def __init__(self, A, b):
        """
        Class constructor.
        
        If :math:`m = 1` then the space is an hyper-plane and a closed form
        projection is available. Otherwise, the projection employes a
        pseudo-inverse computation for projecting.

        Parameters
        ----------
        A : array_like
            The constraint matrix.
        b : array_like
            The constraint vector.

        Notes
        -----
        If it is an hyperplane, `A` is normalized to a unit vector, hence `b` 
        is divided by the norm of `A`.
        """
        
        b = np.reshape(b, (-1,1))
        
        if b.size != A.shape[0]:
            raise ValueError("Incompatible shapes of `A` and `b`.")
        super().__init__(A.shape[1], 1)
        
        # store arguments
        self.A, self.b = A, b
        self.hyperplane = A.shape[0] == 1

        if self.hyperplane:
            norm_A = utils.norm(A)
            self.A, self.b = A/norm_A, b/norm_A
        else:
            self.A_pinv = la.pinv(self.A)
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return all(self.A.dot(x) == self.b)
    
    def projection(self, x):
        
        x = self.check_input(x)
        
        if x in self: return x
        else:
            if self.hyperplane:
                return x - (self.A.dot(x) - self.b)*self.A.T
            else:
                return x - self.A_pinv.dot(self.A.dot(x) - self.b)

class Halfspace(Set):
    r"""
    Halfspace.
    
    This class implements:
        
        .. math:: \{ x \in \mathbb{R}^n \ | \ \langle a, x \rangle = b \}
    
    for given vetor :math:`a \in \mathbb{R}^{n}` and scalar
    :math:`b \in \mathbb{R}`.
    """
    
    def __init__(self, a, b):
        """
        Class constructor.

        Parameters
        ----------
        a : array_like
            The vector :math:`a` defining the half-space.
        b : float
            The scalar :math:`b` defining the half-space.
        
        Raises
        ------
        ValueError.
        
        Notes
        -----
        The vector `a` is normalized to a unit vector, hence `b` is divided
        by the norm of `a`.
        """
        
        a = np.reshape(a, (-1,1)) # reshape to column vector
        super().__init__(a.size, 1)
        
        if not np.isscalar(b):
            raise ValueError("Coefficient `b` must be a scalar.")
        
        # store arguments
        norm_a = utils.norm(a)
        self.a, self.b = a/norm_a, b/norm_a
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return (self.a.T.dot(x) - self.b).item() <= 0
    
    def projection(self, x):
        
        x = self.check_input(x)
        
        if x in self: return x
        else: return x - (self.a.T.dot(x) - self.b)*self.a

class Ball_l1(Set):
    r"""
    :math:`\ell_1`-ball set.
    
    This class implements:
        
        .. math:: \{ x \in \mathbb{R}^n \ | \ \| x - c \|_1 \leq r \}
    
    for a center :math:`c` and radius :math:`r > 0`.
    """
    
    def __init__(self, center, radius):
        """
        Class constructor.

        Parameters
        ----------
        radius : float
            Radius of the ball.
        center : array_like, optional
            Center of the ball, defaults to the origin.
        
        Raises
        ------
        ValueError.
        """
        
        center = np.reshape(center, (-1, 1))
        super().__init__(*center.shape)
        
        if radius <= 0: raise ValueError("The radius must be positive.")
        
        self.radius, self.center = radius, center
    
    def contains(self, x):
        
        x = self.check_input(x)
        
        return la.norm(x - self.center, ord=1) <= self.radius
    
    def projection(self, x, tol=1e-5):
        
        x = self.check_input(x)
        
        if x in self: return x
        else:
            f = lambda l: la.norm(utils.soft_thresholding(x - self.center, l), ord=1) - self.radius
            l_star = utils.bisection_method(f, 0, la.norm(x, ord=1), tol=tol)
            return self.center + utils.soft_thresholding(x - self.center, l_star)


# -------- TIME DOMAIN

class T(Set):
    r"""
    Set of sampling times.
    
    This class implements the set of sampling times:
        
        .. math:: \{ t_k \geq 0, \ k \in \mathbb{N} \}
    
    with :math:`t_{k+1} - t_k = T_\mathrm{s}` for a sampling time 
    :math:`T_\mathrm{s}`.
    """
    
    def __init__(self, t_s, t_min=0, t_max=np.inf):
        """
        Class constructor.

        Parameters
        ----------
        t_s : float
            The sampling time :math:`T_\mathrm{s}`.
        t_min : float, optional
            The lower bound for the time.
        t_max : float, optional
            The upper bound for the time. `t_max` is resized so that it is an 
            exact multiple of `t_s`.

        Raises
        ------
        ValueError.
        """
        
        if t_s <= 0: raise ValueError("`t_s` must be positive.")
        if t_min < 0: raise ValueError("`t_min` must be non-negative.")
        if t_max < t_min or t_s > t_max - t_min: raise ValueError("Incompatble arguments.")
        
        self.num_samples = int((t_max - t_min) / t_s)
        self.t_s, self.t_min, self.t_max = t_s, t_min, t_s*self.num_samples+t_min
        
        super().__init__()
        
    def check_input(self, t):
        # returns a number k such that k*t_s gives t (or the closest multiple 
        # of t_s, rounding down)
        
        if t not in self:
            raise ValueError("`t` is out of bounds.")
        
        return int((t - self.t_min) / self.t_s)

    def contains(self, t):
        
        return t >= self.t_min and t <= self.t_max
    
    def projection(self, t):
        
        return self.check_input(t)
    
    def translate(self, t):
        
        return T(self.t_s, self.t_min+t, self.t_max+t)
    
    def scale(self, c):
        # the upper and lower bound are rescaled, and t_s is changed
        # so that it preserves the same num of samples
        
        t_min, t_max = c*self.t_min, c*self.t_max
        return T((t_max - t_min)/self.num_samples, c*self.t_min, t_max)
    
    def __eq__(self, other):
        
        if not isinstance(other, T):
            raise TypeError("Equality not implemented for 'T' and '{}'.".format(type(other).__name__))
        
        if self is other: return True
        else: return self.t_s == other.t_s and \
                     self.t_min == other.t_min and \
                     self.t_max == other.t_max
    
    def __ne__(self, other):
        
        return not self.__eq__(other)


#%% PROJECTION onto INTERSECTION of SETS

def alternating_projections(sets, x, tol=1e-10, num_iter=10):
    """
    Method of alternating projections.
    
    This function returns a point in the intersection of the given convex
    sets, computed using the method of alternating projections (MAP) [#]_.

    Parameters
    ----------
    x : array_like
        The starting point.
    sets : list
        The list of sets.
    tol : float, optional
        The stopping condition. If the difference between consecutive iterates
        is smaller than or equal to `tol`, then the function returns. 
        Defaults to :math:`10^{-10}`.
    num_iter : int, optional
        The maximum number of iterations of the projection algorithm. Defaults
        to :math:`1000`. This stopping condition is enacted if the algorithm
        does not reach `tol`.

    Returns
    -------
    x : ndarray
        A point in the intersection.
    
    References
    ----------
    .. [#] H. Bauschke and V. Koch, "Projection Methods: Swiss Army Knives for
           Solving Feasibility and Best Approximation Problems with 
           Halfspaces," in Contemporary Mathematics, vol. 636, S. Reich and 
           A. Zaslavski, Eds. Providence, Rhode Island: 
           American Mathematical Society, 2015, pp. 1–40.
    """
    
    
    for l in range(int(num_iter)):
        
        x_old = x
        
        for s in sets: x = s.projection(x)
            
        if la.norm(x - x_old) <= tol: break
    
    return x