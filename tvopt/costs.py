#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cost template definition and examples.
"""

import math
import numpy as np
from numpy import linalg as la
from scipy.special import binom
from functools import partial, partialmethod

from tvopt import solvers, sets, utils



#%% COST TEMPLATES

class Cost():
    r"""
    Template for a cost function.

    This class defines the template for a cost function
    
    .. math:: f : \mathbb{R}^{n_1 \times n_2 \times \ldots} \times \mathbb{R}_+
              \to \mathbb{R} \cup \{ +\infty \}
    
    which depends on the unknown
    :math:`\pmb{x} \in \mathbb{R}^{n_1 \times n_2 \times \ldots}` and,
    optionally, on the time :math:`t \in \mathbb{R}_+`.
    
    `Cost` objects support the following operations:
        
        - negation
        - sum (by another cost or with a scalar),
        - product (by another cost or with a scalar),
        - division and power with a scalar.
    
    A `Cost` object should expose, compatibly with the smoothness degree, the 
    methods `function`, `gradient`, `hessian`, `proximal`. The convention for
    these methods is that the first positional argument is :math:`\pmb{x}`,
    and only a second positional argument is allowed, for :math:`t`. Any other 
    argument should be passed as a keyword argument.
    
    If the cost is time-varying, then it should expose the methods 
    `time_derivative` and `sample`, as well; see methods' documentation for
    the default behavior.
    
    Attributes
    ----------
    dom : sets.Set
        The x domain :math:`\mathbb{R}^{n_1 \times n_2 \times \ldots}`.
    time : sets.T
        The time domain :math:`\mathbb{R}_+`. If the cost is static this is None.
    is_dynamic : bool
        Attribute to check if the cost is static or dynamic.
    smooth : int
        This attribute stores the smoothness degree of the cost, for example
        it is :math:`0` if the cost is continuous, :math:`1` if the cost
        is differentiable, *etc*. By convention it is :math:`-1` if the cost
        is discontinuous.
    prox_solver : str or None
        This attribute specifies the method (gradient or Newton) that should be
        used to compute the proximal
        
        .. math:: \text{prox}_{\rho f(\cdot; t)}(\pmb{x}) = \text{argmin}_{\pmb{y}} 
                  \left\{ f(\pmb{y};t) + \frac{1}{2 \rho} \| \pmb{y} - \pmb{x} \|^2 \right\}
        
        of the cost, if a closed form is not available. See also the auxiliary
        function `compute_proximal`.
    
    Notes
    -----
    Not all operations preserve convexity.
    """
    
    def __init__(self, dom, time=None, prox_solver=None):
        """
        Class constructor.

        Parameters
        ----------
        dom : sets.Set
            The x domain of the cost.
        time : sets.T, optional
            The time domain of the cost. The default is None, in which case the
            cost is static.
        prox_solver : str, optional
            The method for computing the proximal if it is unavailable in 
            closed form. The default is None, and see `compute_proximal` for 
            the behavior in this case.
        """
                
        self.dom, self.time = dom, time
        self.is_dynamic = time is not None
        self.smooth = -1
        self.prox_solver = prox_solver # proximal solver method
    
    def function(self, x, *args, **kwargs):
        """
        An evaluation of the cost. *Implement if needed*.

        Parameters
        ----------
        x : array_like
            The x where the cost should be evaluated.
        *args
            The time at which the cost should be evaluated. Not required if the
            cost is static.
        **kwargs
            Any other required argument.
        """
        
        raise NotImplementedError()
    
    def gradient(self, x, *args, **kwargs):
        """
        An evaluation of the cost's gradient or sub-gradient. *Implement if 
        needed*.

        Parameters
        ----------
        x : array_like
            The x where the (sub-)gradient should be evaluated.
        *args
            The time at which the (sub-)gradient should be evaluated. Not 
            required if the cost is static.
        **kwargs
            Any other required argument.
        """
        
        raise NotImplementedError()
    
    def hessian(self, x, *args, **kwargs):
        """
        An evaluation of the cost's Hessian. *Implement if needed*.

        Parameters
        ----------
        x : array_like
            The x where the Hessian should be evaluated.
        *args
            The time at which the Hessian should be evaluated. Not 
            required if the cost is static.
        **kwargs
            Any other required argument.
        """
        
        raise NotImplementedError()
    
    def proximal(self, x, *args, penalty=1, **kwargs):
        r"""
        An evaluation of the cost's proximal.
        
        If this method is not overwritten, the default behavior is to 
        recursively compute the proximal via a gradient or Newton backtracking
        algorithm. See `compute_proximal` for the function that is used
        for this purpose.

        Parameters
        ----------
        x : array_like
            The x where the proximal should be evaluated.
        *args
            The time at which the proximal should be evaluated. Not 
            required if the cost is static.
        penalty : float, optional
            The penalty parameter :math:`\rho` for the proximal evaluation.
            Defaults to :math:`1`.
        **kwargs
            Any other required argument.
        """
        
        if self.is_dynamic:
            return self.sample(args[0]).proximal(x, penalty=penalty, **kwargs)
        else:
            if "solver" not in kwargs: kwargs["solver"] = self.prox_solver
            return compute_proximal(self, x, penalty, **kwargs)
    
    def time_derivative(self, x, t, der="tx", **kwargs):
        """
        A derivative w.r.t. time of the cost.
        
        This method computes derivatives w.r.t. time of the cost, or mixed 
        derivatives w.r.t. both time and x (*e.g.* the derivative in time of 
        the gradient).
        
        If this method is not overwritten, it computes the derivative by 
        default using *backward finite differences*. See 
        `backward_finite_difference` for details.
        
        If the cost is static, :math:`0` is returned.

        Parameters
        ----------
        x : array_like
            The x where the derivative should be evaluated.
        t : float
            The time at which the derivative should be evaluated.
        der : str, optional
            A sequence of "x" and "t" that chooses which derivative should be 
            computed. For example, the default "tx" denotes the derivative
            w.r.t. time of the cost's (sub-)gradient.
        **kwargs
            Any other required argument.

        Raises
        ------
        ValueError
            If the number of "x" characters in `der` exceeds :math:`2`.

        Returns
        -------
        array_like
            The required derivative or :math:`0`.
        """
        
        if self.is_dynamic:
            
            # parse the derivative order
            der = "".join(der.split()).lower() # remove spaces, set to lower case
            # num. of derivations w.r.t. time and x
            order_t, order_x = der.count("t"), der.count("x")
            
            # signal to be derived
            if order_x == 0: s = self.function
            elif order_x == 1: s = self.gradient
            elif order_x == 2: s = self.hessian
            else: raise ValueError("Derivative of order {} w.r.t. `x` is unavailable.".format(order_x))
            
            return backward_finite_difference(partial(s, x, **kwargs), t, order=order_t, step=self.time.t_s)
        
        else:
            return 0
            
    def sample(self, t):
        """
        Sample the cost.
        
        This method returns a `SampledCost` object which exposes the same 
        methods of the cost but fixing the time argument to `t`.
        
        If the cost is static, the cost itself is returned.

        Parameters
        ----------
        t : float
            The time at which the cost should be sampled.

        Returns
        -------
        Cost
            The sampled cost or, if static, the cost itself.
        """
        
        if self.is_dynamic: return SampledCost(self, t)
        else: return self
    
    def __add__(self, other):
        
        if utils.is_scalar(other): other = Constant(self.dom, other)
        return SumCost(self, other)
    
    def __mul__(self, other):
        
        if utils.is_scalar(other): return ScaledCost(self, other)
        else: return ProductCost(self, other)
    
    def __pow__(self, other):
        
        return PowerCost(self, other)
    
    def __neg__(self):
        
        return self.__mul__(-1)

    def __radd__(self, other):
        
        return self.__add__(other)
    
    def __sub__(self, other):
        
        return self.__add__(-other)
    
    def __rsub__(self, other):
        
        return self.__neg__().__add__(other)
    
    def __rmul__(self, other):
        
        return self.__mul__(other)
    
    def __truediv__(self, other):
        
        return self.__mul__(1/other)


# -------- SPECIAL COSTS

class SeparableCost(Cost):
    r"""
    Separable cost function.

    This class defines a separable cost, that is
    
    .. math:: f(\pmb{x}; t) = \sum_{i = 1}^N f_i(x_i; t)
    
    where :math:`x_i \in \mathbb{R}^{n_1 \times n_2 \times \ldots}` for each
    :math:`i = 1, \ldots, N`. Each of the component functions :math:`f_i` can
    be either static or dynamic. This is useful for defining distributed
    optimization problems.
    
    The overall dimension of the domain is 
    :math:`n_1 \times n_2 \times \ldots \times N`, meaning that the last 
    dimension indexes the components.
    
    The class exposes the same methods as any `Cost`, with the difference that
    the keyword argument `i` can be used to evaluate only a single component.
    If all components are evaluated, an ndarray is returned with the last
    dimension indexing the components.

    The class has the `Cost` attributes, with the following additions or
    differences.
    
    Attributes
    ----------
    costs : list
        The component costs.
    N : int
        The number of components.
    is_dynamic : bool
        True if at least one component is dynamic.
    smooth : int
        This is the minimum of the smoothness degrees of all components.
    """
    
    def __init__(self, costs):
        """
        Class constructor.

        Parameters
        ----------
        costs : list
            The component costs.
        """
        
        # check if there are dynamic costs
        times = [c.time for c in costs if c.is_dynamic]
        time = times[0] if len(times) > 0 else None
        
        super().__init__(sets.R(*costs[0].dom.shape, len(costs)), time)
        self.costs, self.N = costs, len(costs)
        
        self.smooth = min([c.smooth for c in costs])
    
    def _evaluate(self, method, x, *args, i=None, **kwargs):
        """
        Auxiliary method for components' evaluation.
        
        This method performs an evaluation of the components' function,
        (sub-)gradient or Hessian methods. If the keyword argument `i` is 
        specified, then only the corresponding cost is evaluated.

        Parameters
        ----------
        method : str
            The method to be evaluated.
        x : array_like
            The x where the cost(s) should be evaluated.
        *args
            The time at which the cost(s) should be evaluated. Not required if 
            the cost is static.
        i : int, optional
            If specified, only the corresponding component cost :math:`f_i` is
            evaluated.
        **kwargs
            Any other required argument.

        Returns
        -------
        ndarray
            If `i` is specified, the evaluation of the i-th component, 
            otherwise an ndarray stacking the components evaluations along
            the last dimension.
        """
        
        if i is not None:
            return getattr(self.costs[i], method)(x, *args, **kwargs) if self.costs[i].is_dynamic \
                   else getattr(self.costs[i], method)(x, **kwargs)
        else:
            return np.stack([getattr(c, method)(x[...,i], *args, **kwargs) if c.is_dynamic \
                   else getattr(c, method)(x[...,i], **kwargs) for c, i in zip(self.costs, range(self.N))], axis=-1)  
    
    function = partialmethod(_evaluate, "function")
    gradient = partialmethod(_evaluate, "gradient")
    hessian = partialmethod(_evaluate, "hessian")
    
    def proximal(self, x, *args, penalty=1, i=None, **kwargs):
        """
        An evaluation of the cost(s) proximal(s).
        
        This is the same as calling _evaluate with "proximal", with the 
        difference that is customized to handle the penalty parameter. In 
        particular, the penalty can either be a scalar, in which case the same
        penalty is used for all components, or a list of component-wise 
        penalties.
        """

        if i is None and not isinstance(penalty, list):
            penalty = [penalty for _ in range(self.N)]
        
        # compute the proximals
        if i is not None:
            return self.costs[i].proximal(x, *args, penalty=penalty, **kwargs) if self.costs[i].is_dynamic \
                   else self.costs[i].proximal(x, penalty=penalty, **kwargs)
        else:
            return np.stack([c.proximal(x[...,i], *args, penalty=penalty[i], **kwargs) if c.is_dynamic \
                   else c.proximal(x[...,i], penalty=penalty[i], **kwargs) for c, i in zip(self.costs, range(self.N))], axis=-1) 

class DiscreteDynamicCost(Cost):
    r"""
    Dynamic cost from a sequence of static costs.
    
    This class creates a dynamic cost from a list of static costs. That is, 
    given a sampling time :math:`T_\mathrm{s}`, the cost at time 
    :math:`t_k = k T_\mathrm{s}` is:
    
    .. math:: f(\pmb{x}; t_k) = f_k(\pmb{x})
    
    with :math:`f_k` the k-th static cost in the list.
    """
    
    def __init__(self, costs, t_s=1):
        """
        Class constructor.
        
        Creates the dynamic cost, optionally using the specified sampling
        time `t_s`.

        Parameters
        ----------
        costs : list
            The sequence of static costs.
        t_s : float, optional
            The sampling time, that is, the time that elapses between two
            consecutive costs. The default is 1.
        
        Notes
        -----
        The costs are implicitly assumed to have the same domain and to be 
        static.
        """
        
        # create time domain
        time = sets.T(t_s, t_min=0, t_max=len(costs)*t_s)
        
        super().__init__(costs[0].dom, time)
        self.costs, self.N = costs, len(costs)
        
        self.smooth = min([c.smooth for c in costs])
    
    def _evaluate(self, method, x, t, **kwargs):
        r"""
        Auxiliary method for costs evaluation.
        
        This method evaluates the required method at the given time, by calling
        the method of the k-th function, with :math:`t_k` being the closest
        sampling time to `t`.

        Parameters
        ----------
        method : str
            The method to be evaluated.
        x : array_like
            The x where the cost should be evaluated.
        t : float
            The time at which the cost should be evaluated.
        **kwargs
            Any other required argument.

        Returns
        -------
        ndarray
            The evaluation of the cost.
        """
        
        return getattr(self.costs[self.time.check_input(t)], method)(x, **kwargs)
    
    function = partialmethod(_evaluate, "function")
    gradient = partialmethod(_evaluate, "gradient")
    hessian = partialmethod(_evaluate, "hessian")
    proximal = partialmethod(_evaluate, "proximal")

    def sample(self, t):
        """
        Sample the cost.
        
        The difference with the default `Cost` method is that it returns a
        cost in the list rather than a `SampledCost`.

        Parameters
        ----------
        t : float
            The time at which the cost should be sampled.

        Returns
        -------
        Cost
            The closest cost in the list.
        """
        
        return self.costs[self.time.check_input(t)]


# -------- AUXILIARY CLASSES

class Constant(Cost):
    """
    Constant cost.
    
    This class defines a constant, whose value is stored in the attribute `c`.
    The `gradient` and `hessian` methods return 0, while the proximal acts
    as an identity.
    
    Attributes
    ----------
    dom : sets.Set
        The given cost domain, for compatibility with other costs.
    c : float
        The constant value.
    smooth : int
        The smoothness degree, set to 2.
    """
    
    def __init__(self, dom, c):
        """
        Class constructor.

        Parameters
        ----------
        dom : sets.Set
            The cost domain.
        c : float
            The constant value.
        """
        
        super().__init__(dom)
        self.c = np.array(c).item() # this allows to account for 0-dim or 1 x 1 ndarrays
        
        self.smooth = 2
    
    def function(self, *args, **kwargs):
        """
        An evaluation of the cost.
        
        Returns the costant value.
        """

        return self.c
    
    def gradient(self, *args, **kwargs):
        """
        An evaluation of the cost's gradient.
        
        Returns 0.
        """

        return 0
    
    def hessian(self, *args, **kwargs):
        """
        An evaluation of the cost's Hessian.
        
        Returns 0.
        """
        
        return 0
    
    def proximal(self, x, *args, **kwargs):
        """
        An evaluation of the cost's proximal.
        
        Acts as the identity, returning `x`.
        """
        
        return x

class SumCost(Cost):
    r"""
    Sum of costs.
    
    This class defines a cost as the sum of an arbitrary number of costs. That
    is, given the costs 
    :math:`f_i : \mathbb{R}^n \times \mathbb{R}_+ \to \mathbb{R} \cup \{ +\infty \}`
    with :math:`i = 1, \ldots, N`, the class defines:
        
        .. math:: f(\pmb{x}; t) = \sum_{i = 1}^N f_i(\pmb{x}; t)
    
    The `function`, `gradient` and `hessian` are defined from the components' 
    methods using the sum rule, while the proximal by default is computed 
    recursively.
    """
    
    def __init__(self, *costs):
        """
        Class constructor.
        
        Define the sum of the `costs`, checking if any of them is dynamic.

        Parameters
        ----------
        *costs : Cost
            An arbitrary number of costs to be summed.
        """
        
        # check if dynamic
        times = [c.time for c in costs if c.is_dynamic]
        time = times[0] if len(times) > 0 else None

        super().__init__(costs[0].dom, time)
        self.costs, self.N = costs, len(costs)
        
        self.smooth = min([c.smooth for c in costs])
    
    def _evaluate(self, method, x, *args, **kwargs):
        """
        Auxiliary method for cost evaluation.
        
        An evaluation of the sum of costs `function`, `gradient` or `hessian`.

        Parameters
        ----------
        method : str
            The method to be evaluated.
        x : array_like
            The x where the cost should be evaluated.
        *args
            The time at which the cost(s) should be evaluated. Not required if 
            the cost is static.
        **kwargs
            Any other required argument.

        Returns
        -------
        ndarray
            The evaluation of the sum cost.
        """
        
        return sum([getattr(c, method)(x, *args, **kwargs) if c.is_dynamic \
                    else getattr(c, method)(x, **kwargs) for c in self.costs])
    
    function = partialmethod(_evaluate, "function")
    gradient = partialmethod(_evaluate, "gradient")
    hessian = partialmethod(_evaluate, "hessian")
        
class ScaledCost(Cost):
    r"""
    Scaled cost.
    
    This class defines a cost scaled by a constant. That is, given the cost
    :math:`f : \mathbb{R}^n \times \mathbb{R}_+ \to \mathbb{R} \cup \{ +\infty \}`
    and :math:`c \in \mathbb{R}` it defines:
        
        .. math:: g(\pmb{x}; t) = c f(\pmb{x}; t).
    
    The class is used for the product and division by a constant.
    """
    
    def __new__(cls, cost, s):
        """
        Object instantiation that handles zero scaling constants.
        """
        
        if s == 0: return Constant(cost.dom, 0)
        else: return object.__new__(cls)
        
    def __init__(self, cost, s):
        """
        Class constructor.

        Parameters
        ----------
        cost : Cost
            The cost to be scaled.
        s : float
            The scaling constant.
        """

        # check if dynamic
        time = cost.time if cost.is_dynamic else None
        
        super().__init__(cost.dom, time)
        self.cost, self.s = cost, np.array(s).item()
        
        self.smooth = cost.smooth
    
    def _evaluate(self, method, *args, **kwargs):
        """
        Auxiliary method for cost evaluation.
        
        An evaluation of the scaled cost's `function`, `gradient`, `hessian` or
        `proximal`.

        Parameters
        ----------
        method : str
            The method to be evaluated.
        x : array_like
            The x where the cost should be evaluated.
        *args
            The time at which the cost(s) should be evaluated. Not required if 
            the cost is static.
        **kwargs
            Any other required argument.

        Returns
        -------
        ndarray
            The evaluation of the scaled cost.
        """
        
        # handle the proximal
        if method == "proximal":
            penalty = kwargs.pop("penalty")/self.s
            return getattr(self.cost, method)(*args, penalty=penalty, **kwargs)
        
        return self.s*getattr(self.cost, method)(*args, **kwargs)
    
    function = partialmethod(_evaluate, "function")
    gradient = partialmethod(_evaluate, "gradient")
    hessian = partialmethod(_evaluate, "hessian")
    proximal = partialmethod(_evaluate, "proximal")

class ProductCost(Cost):
    """
    Product of two costs.
    
    This class defines a cost from the product of two given costs. Derivatives
    are computed using the chain rule.
    """
        
    def __init__(self, c_1, c_2):
        """
        Class constructor

        Parameters
        ----------
        c_1 : Cost
            One of the costs to be multiplied.
        c_2 : Cost
            One of the costs to be multiplied.
        """

        # check if dynamic
        times = [c.time for c in (c_1, c_2) if c.is_dynamic]
        time = times[0] if len(times) > 0 else None
        
        super().__init__(c_1.dom, time)
        self.costs = (c_1, c_2)
        
        self.smooth = min(c_1.smooth, c_2.smooth)
    
    def function(self, x, *args, **kwargs):
        """
        An evaluation of the product cost.
        """
        
        f_1, f_2 = [c.function(x, *args, **kwargs) if c.is_dynamic else c.function(x, **kwargs) for c in self.costs]
        
        return f_1*f_2
    
    def gradient(self, x, *args, **kwargs):
        """
        An evaluation of the product cost (sub-)gradient.
        """
        
        f_1, f_2 = [c.function(x, *args, **kwargs) if c.is_dynamic else c.function(x, **kwargs) for c in self.costs]
        g_1, g_2 = [c.gradient(x, *args, **kwargs) if c.is_dynamic else c.gradient(x, **kwargs) for c in self.costs]
        
        return f_2*g_1 + f_1*g_2
    
    def hessian(self, x, *args, **kwargs):
        """
        An evaluation of the product cost Hessian.
        """
        
        f_1, f_2 = [c.function(x, *args, **kwargs) if c.is_dynamic else c.function(x, **kwargs) for c in self.costs]
        g_1, g_2 = [c.gradient(x, *args, **kwargs) if c.is_dynamic else c.gradient(x, **kwargs) for c in self.costs]
        h_1, h_2 = [c.hessian(x, *args, **kwargs) if c.is_dynamic else c.hessian(x, **kwargs) for c in self.costs]
        
        return f_2*h_1 + f_1*h_2 + g_1.dot(g_2.T) + g_2.dot(g_1.T)

class PowerCost(Cost):
    """
    Power cost.
    
    This class defines a cost as the given power of a cost. It is used for 
    implementing the `*` operation.
    """
    # class for costs elevated to some power
    # if p = 0 then a constant cost equal to 1 is returned
    
    def __new__(cls, cost, p):
        """
        Object instantiation that handles zero powers.
        """
        
        if p == 0: return Constant(cost.dom, 1)
        else: return object.__new__(cls)
            
    def __init__(self, cost, p):
        """
        Class constructor.

        Parameters
        ----------
        cost : Cost
            The given cost.
        p : float
            The power for the cost.
        """
        
        p = np.array(p).item()
        
        # check if dynamic
        time = cost.time if cost.is_dynamic else None
        
        super().__init__(cost.dom, time)
        self.cost, self.p = cost, p
        
        self.smooth = cost.smooth
    
    def function(self, *args, **kwargs):
        """
        An evaluation of the power cost.
        """
        
        return self.cost.function(*args, **kwargs)**self.p
    
    def gradient(self, *args, **kwargs):
        """
        An evaluation of the power cost (sub-)gradient.
        """
        
        return self.p*self.cost.function(*args, **kwargs)**(self.p-1) * self.cost.gradient(*args, **kwargs)
    
    def hessian(self, *args, **kwargs):
        """
        An evaluation of the power cost Hessian.
        """
        
        f, g = self.cost.function(*args, **kwargs), self.cost.gradient(*args, **kwargs)
        
        return self.p*(self.p-1)* f**(self.p-2) * np.dot(g, np.transpose(g)) + \
               self.p * f**(self.p-1) * self.cost.hessian(*args, **kwargs)

class SampledCost(Cost):
    """
    Sampled cost.
    
    This class defines a *static* cost by sampling a *dynamic* cost at a given
    time.
    """
                
    def __init__(self, cost, t):
        """
        Class constructor.

        Parameters
        ----------
        cost : Cost
            The dynamic cost to be sampled.
        t : float
            The sampling time.
        """

        super().__init__(cost.dom)
        self.cost, self.t = cost, t
        
        self.smooth = cost.smooth
    
    def _evaluate(self, method, x, **kwargs):
        """
        Auxiliary method for cost evaluation.
        
        An evaluation of the sampled cost's `function`, `gradient`, or 
        `hessian`. Notice that the method does not require positional arguments
        after `x` since the cost is static.

        Parameters
        ----------
        method : str
            The method to be evaluated.
        x : array_like
            The x where the cost should be evaluated.
        **kwargs
            Any other required argument.

        Returns
        -------
        ndarray
            The evaluation of the sampled cost.
        """
        
        return getattr(self.cost, method)(x, self.t, **kwargs)
    
    function = partialmethod(_evaluate, "function")
    gradient = partialmethod(_evaluate, "gradient")
    hessian = partialmethod(_evaluate, "hessian")
    
    def proximal(self, x, penalty=1, **kwargs):
        """
        An evaluation of the cost's proximal.
        """
        
        return super().proximal(x, penalty=penalty, **kwargs)


#%% EXAMPLES: STATIC

# -------- SCALAR COSTS

class AbsoluteValue(Cost):
    """
    Scalar absolute value function.
    """
    
    def __init__(self, weight=1):
        
        super().__init__(sets.R())

        self.weight, self.smooth = weight, 0
    
    def function(self, x):
        
        return self.weight*abs(float(x))
    
    def gradient(self, x):
        
        return self.weight*math.copysign(1, float(x))
    
    def proximal(self, x, penalty=1):
        
        x = float(x)
        
        return math.copysign(1, x)*max(abs(x) - self.weight*penalty, 0)

class Quadratic_1D(Cost):
    """
    Scalar quadratic cost.
    
    The cost is defined as
    
        .. math:: f(x) = a x^2 / 2 + b x + c.
    """
    
    def __init__(self, a, b, c=0):
        
        super().__init__(sets.R())
        
        self.a, self.b, self.c = np.array(a).item(), np.array(b).item(), np.array(c).item()
        self.smooth = 2
    
    def function(self, x):
        
        return np.array(0.5*self.a*x**2 + self.b*x + self.c).item()
    
    def gradient(self, x):
        
        return self.a*float(x) + self.b
    
    def hessian(self, x=None): # x argument is only for compatibility
        
        return self.a
    
    def proximal(self, x, penalty=1):
        
        return (float(x) - penalty*self.b) / (1 + penalty*self.a)

class Huber_1D(Cost):
    r"""
    Huber loss.
    
    The cost is defined as
    
        .. math:: f(x) = \begin{cases} x^2 / 2 & \text{if} \ |x| \leq \theta \\
                                       \theta (|x| - \theta / 2) & \text{otherwise} \end{cases}
    
    where :math:`\theta > 0` is a given threshold.
    """
    
    def __init__(self, threshold):

        super().__init__(sets.R())
        self.threshold, self.smooth = threshold, 2
    
    def function(self, x):
        
        x = float(x)
        n = abs(x)
        
        if n <= self.threshold: return x**2/2
        else: return self.threshold*(n - self.threshold/2)
    
    def gradient(self, x):
        
        x = float(x)
        n = utils.norm(x)
        
        if n <= self.threshold: return x
        else: return self.threshold*x/n
    
    def hessian(self, x):
                
        if abs(x) <= self.threshold: return 1
        else: return 0
    
    def proximal(self, x, penalty=1):
        
        x = float(x)
        n = abs(x)
        
        if n <= self.threshold*(penalty+1): return x / (penalty+1)
        else: return (1 - penalty*self.threshold/n)*x

class Logistic(Cost):
    r"""
    Logistic function.
    
    The function is defined as
    
        .. math:: f(x) = \log\left( 1 + \exp(x) \right).
    """
    
    def __init__(self):
        
        super().__init__(sets.R())
        self.smooth = 2
    
    def function(self, x):

        return math.log(1 + math.exp(float(x)))
    
    def gradient(self, x):
        
        x = float(x)
        
        return math.exp(x) / (1 + math.exp(x))
    
    def hessian(self, x):
        
        x = float(x)
        
        return math.exp(x) / (1 + math.exp(x))**2


# -------- NORMS

class Norm_1(Cost):
    r"""
    Class for :math:`\ell_1` norm.
    
    The function is defined as
    
        .. math:: f(\pmb{x}) = w \| \pmb{x} \|_1
    
    for :math:`\pmb{x} \in \mathbb{R}^n` and :math:`w > 0`.
    """
    
    def __new__(cls, n=1, weight=1):
        
        if n == 1: return AbsoluteValue(weight=weight)
        else: return object.__new__(cls)
        
    def __init__(self, n=1, weight=1):
        r"""
        Constructor of the cost.
        
        Parameters
        ----------
        n : int
            Size of the unknown :math:`x`.
        weight : float, optional
            A weight multiplying the norm. Defaults to :math:`1`.
        """
        
        # check arguments validity
        super().__init__(sets.R(n, 1))
        
        self.weight = weight
        self.smooth = 0
    
    def function(self, x):
                
        return self.weight*la.norm(self.dom.check_input(x), ord=1)
    
    def gradient(self, x):
                
        return self.weight*np.sign(self.dom.check_input(x))
    
    def proximal(self, x, penalty=1):
        """
        Proximal evaluation of :math:`\ell_1` norm, a.k.a. soft-thresholding.
        
        See Also
        --------
        utils.soft_thresholding
        """
        
        return utils.soft_thresholding(self.dom.check_input(x), self.weight*penalty)

class Norm_2(Cost):
    """
    Square :math:`2`-norm.
    """
    
    def __new__(cls, n=1, weight=1):
        
        return Quadratic(weight*np.eye(n), np.zeros((n, 1)))

class Norm_inf(Cost):
    r"""
    Class for :math:`\ell_\infty` norm.
    """
    
    def __new__(cls, n=1, weight=1):
        
        if n == 1: return AbsoluteValue(weight=weight)
        else: return object.__new__(cls)
    
    def __init__(self, n=1, weight=1):
        
        # check arguments validity
        super().__init__(sets.R(n, 1))
        
        self.weight = weight
        # l1 ball useful for computing the proximal
        self.l1_ball = sets.Ball_l1(np.zeros(self.dom.shape), 1)
    
    def function(self, x):
                
        return self.weight*la.norm(self.dom.check_input(x), ord=np.inf)
    
    def proximal(self, x, penalty=1, tol=1e-5):
        r"""
        Proximal evaluation of :math:`\ell_\infty` norm.
        
        See [#]_ for the formula.
        
        References
        ----------
        .. [#] A. Beck, First-Order Methods in Optimization. Philadelphia, PA: 
            Society for Industrial and Applied Mathematics, 2017.
        """
        
        # check arguments validity
        x = self.dom.check_input(x)
        
        if np.any(penalty <= 0):
            raise ValueError("The penalty parameter(s) must be positive.")
        
        # compute the proximal
        return x - penalty*self.l1_ball.projection(x / penalty, tol=tol)


# -------- INDICATOR FUNCTIONS

class Indicator(Cost):
    r"""
    Indicator function of a given set.
    
    This objects implements the indicator function of a given `Set` object. 
    That is, given the set :math:`\mathbb{S}` we define:
        
        .. math:: f(\pmb{x}) = \begin{cases} 0 & \text{if} \ \pmb{x} \in \mathbb{S} \\
                                             +\infty & \text{otherwise}. \end{cases}
    
    The proximal operator of the cost is the projection onto the set.
    """
    
    def __init__(self, s):
        
        super().__init__(s)
    
    def function(self, x):
        
        if x in self.dom: return 0
        else: return np.inf
    
    def projection(self, x, **kwargs):
        
        return self.dom.projection(x, **kwargs)
    
    def proximal(self, x, *args, penalty=None, **kwargs):
        
        return self.dom.projection(x, **kwargs)
    
    def __add__(self, other):
        # if other is an indicator, return the indicator of the intersection
        # otherwise normal addition rules apply
        
        if isinstance(other, Indicator): return Indicator(self.dom + other.dom)
        else: return super().__add__(other)


# -------- SMOOTH COSTS

class Linear(Cost):
    r"""
    Linear cost.
    
    The function is defined as
        
            .. math:: f(x) = \langle \pmb{x}, \pmb{b} \rangle + c.
    """
    
    def __new__(cls, a, b=0):
        
        n = np.size(a)
        
        return Quadratic(np.zeros((n, n)), a, b)    

class Quadratic(Cost):
    r"""
    Quadratic cost.
    
    The function is defined as
        
            .. math:: f(x) = \frac{1}{2} \pmb{x}^\top \pmb{A} \pmb{x} + \langle \pmb{x}, \pmb{b} \rangle + c
        
    with the given matrix :math:`\pmb{A} \in \mathbb{R}^{n \times n}` and
    vector :math:`\pmb{b} \in \mathbb{R}^n`.
    """
    
    def __new__(cls, A, b, c=0):
        
        if np.size(A) == 1: return Quadratic_1D(A, b, c)
        else: return object.__new__(cls)
    
    def __init__(self, A, b, c=0):
        
        A = np.array(A)
        super().__init__(sets.R(A.shape[0], 1))
        A = A.reshape((A.shape[0], A.shape[0])) # check it is square
        
        # check b and c (reshaping if need be)
        b = self.dom.check_input(b)
        c = np.array(c).item()
        
        self.A, self.b, self.c = A, b, c
        self.smooth = 2
    
    def function(self, x):
        
        x = self.dom.check_input(x)
        
        return np.array(0.5*x.T.dot(self.A.dot(x)) + self.b.T.dot(x) + self.c).item()
    
    def gradient(self, x):
        
        x = self.dom.check_input(x)
        
        return self.A.dot(x) + self.b
    
    def hessian(self, x=None): # x argument is only for compatibility
        
        return self.A
    
    def proximal(self, x, penalty=1):
        
        x = self.dom.check_input(x)
        
        if self.dom.size == 1: return (x - penalty*self.b) / (1 + penalty*self.A)
        else: return la.solve(np.eye(self.dom.size) + penalty*self.A, x - penalty*self.b)

class Huber(Cost):
    r"""
    Vector Huber loss.
    
    The cost is defined as
    
        .. math:: f(\pmb{x}) = \begin{cases} \|\pmb{x}\|^2 / 2 & \text{if} \ \|\pmb{x}\| \leq \theta \\
                                       \theta (\|\pmb{x}\| - \theta / 2) & \text{otherwise} \end{cases}
    
    where :math:`\theta > 0` is a given threshold.
    """
    
    def __new__(cls, n, threshold):
        
        if n == 1: return Huber_1D(threshold)
        else: return object.__new__(cls)
    
    def __init__(self, n, threshold):
        
        if threshold <= 0: raise ValueError("The threshold must be positive.")
        
        super().__init__(sets.R(n, 1))
        self.threshold = threshold
        self.smooth = 2
    
    def function(self, x):
        
        x = self.dom.check_input(x)
        n = utils.norm(x)
        
        if n <= self.threshold: return n**2/2
        else: return self.threshold*(n - self.threshold/2)
    
    def gradient(self, x):
        
        x = self.dom.check_input(x)
        n = utils.norm(x)
        
        if n <= self.threshold: return x
        else: return self.threshold*x/n
    
    def hessian(self, x):
        
        x = self.dom.check_input(x)
        
        if utils.norm(x) <= self.threshold: return np.eye(self.dom.size)
        else: return np.zeros((self.dom.size, self.dom.size))
    
    def proximal(self, x, penalty=1):
        
        x = self.dom.check_input(x)
        n = utils.norm(x)
        
        if n <= self.threshold*(penalty+1): return x / (penalty+1)
        else: return (1 - penalty*self.threshold/n)*x


# -------- REGRESSION

class LinearRegression(Cost):
    r"""
    Cost for linear regression.
    
    The cost is defined as
    
        .. math:: f(\pmb{x}) = \frac{1}{2} \| \pmb{A} \pmb{x} - \pmb{b} \|^2.
    """
    
    def __new__(cls, A, b):
        
        if np.size(A) == 1: return Quadratic_1D(A**2, -A*b, 0.5*b**2)
        else: return Quadratic(A.T.dot(A), -A.T.dot(b), 0.5*b.T.dot(b))
    
    def __getnewargs__(self):
        return self.A, self.b

class RobustLinearRegression(Cost):
    r"""
    Cost for robust linear regression.
    
    Let :math:`h : \mathbb{R} \to \mathbb{R}` be the Huber loss, then thecost 
    is defined as:
    
        .. math:: f(\pmb{x}) = \sum_{i = 1}^m h(a_i \pmb{x} - b_i)
    
    where :math:`a_i \in \mathbb{R}^{1 \times n}` are the rows of the data 
    matrix :math:`\pmb{A} \in \mathbb{R}^{m \times n}`, and :math:`b_i` the 
    elements of the data vector :math:`\pmb{b}`.
    """
    
    def __init__(self, A, b, threshold):
        # each row in A and element of b represent a data point
        
        super().__init__(sets.R(A.shape[1], 1))
        
        self.m = A.shape[0] # domain dim. and num. data points
        # store data as lists
        self.A = [A[d,].reshape(self.dom.shape).T for d in range(self.m)]
        self.b = [np.array(b[d]).item() for d in range(self.m)]
        
        self.huber_fn = Huber(1, threshold) # Huber function
        self.smooth = 2
    
    def function(self, x):
        
        x = self.dom.check_input(x)
        
        return np.sum([self.huber_fn.function(self.A[d].dot(x) - self.b[d]) for d in range(self.m)])
    
    def gradient(self, x):
        
        x = self.dom.check_input(x)
        
        g = np.zeros(self.dom.shape)
        for d in range(self.m):
            
            g += self.huber_fn.gradient(self.A[d].dot(x) - self.b[d])*self.A[d].T

        return g
    
    def hessian(self, x):
        
        x = self.dom.check_input(x)
                
        h = np.zeros(self.dom.shape[:-1] + self.dom.shape[:-1])
        for d in range(self.m):
            
            h += self.huber_fn.hessian(self.A[d].dot(x) - self.b[d])*self.A[d].T.dot(self.A[d])
        
        return h


#%% EXAMPLES: DYNAMIC

class DynamicExample_1D(Cost):
    r"""
    Scalar benchmark dynamic cost.
    
    The dynamic cost was propposed in [#]_ and is defined as:
    
        .. math:: f(x; t) = \frac{1}{2} (x - \cos(\omega t))^2 + \kappa \log(1 + \exp(\mu x))
    
    with default parameters :math:`\omega = 0.02 \pi`, :math:`\kappa = 7.5`
    and :math:`\mu = 1.75`.
    
    .. [#] A. Simonetto, A. Mokhtari, A. Koppel, G. Leus, and A. Ribeiro, 
           "A Class of Prediction-Correction Methods for Time-Varying 
           Convex Optimization," IEEE Transactions on Signal Processing, 
           vol. 64, no. 17, pp. 4576–4591, Sep. 2016.
    """
    
    def __init__(self, t_s, t_max, omega=0.02*math.pi, kappa=7.5, mu=1.75):
        
        super().__init__(sets.R(), sets.T(t_s, t_max=t_max))
        
        self.omega, self.kappa, self.mu = omega, kappa, mu
        self.smooth = 2
    
    def function(self, x, t):
                
        return 0.5*(x - math.cos(self.omega*t))**2 + self.kappa*math.log(1 + math.exp(self.mu*x))
    
    def gradient(self, x, t):
                
        return x - math.cos(self.omega*t) + self.kappa*self.mu*math.exp(self.mu*x) / (1 + math.exp(self.mu*x))

    def hessian(self, x, t=None):
                
        return 1 + self.kappa*(self.mu**2)*math.exp(self.mu*x) / (1 + math.exp(self.mu*x))**2

    def time_derivative(self, x, t, der="tx"):
        
        # parse the derivative order
        der = ''.join(der.split()).lower()
                
        # time der. gradient
        if der == "tx" or der == 'xt': return self.omega*math.sin(self.omega*t)
        else: return super().time_derivative(self.dom.check_input(x), t, der=der)
    
    def approximate_time_derivative(self, x, t, der="tx"):
        
        return super().time_derivative(x, t, der=der)

class DynamicExample_2D(Cost):
    r"""
    Bi-dimensional benchmark dynamic cost.
    
    The dynamic cost was proposed in [#]_ and is defined as:
    
        .. math:: f(\pmb{x}; t) = \frac{1}{2} (x_1 - \exp(\cos(t)))^2 + \frac{1}{2} (x_2 - x_1 \tanh(t))^2
    
    where we used the notation :math:`\pmb{x} = [x_1, x_2]^\top`.
    
    .. [#] Y. Zhang, Z. Qi, B. Qiu, M. Yang, and M. Xiao, "Zeroing Neural 
           Dynamics and Models for Various Time-Varying Problems Solving 
           with ZLSF Models as Minimization-Type and Euler-Type Special 
           Cases [Research Frontier]," IEEE Computational Intelligence 
           Magazine, vol. 14, no. 3, pp. 52–60, Aug. 2019.
    """
    
    def __init__(self, t_s, t_max):
        
        super().__init__(sets.R(2, 1), sets.T(t_s, t_max=t_max))
        self.smooth = 2
    
    def function(self, x, t):
        
        x = x.flatten()
                
        return 0.5*(x[0] - math.exp(math.cos(t)))**2 + 0.5*(x[1] - x[0]*math.tanh(t))**2
    
    def gradient(self, x, t):
        
        x = x.flatten()
        th = math.tanh(t)
        
        return np.array([[x[0] - math.exp(math.cos(t)) - th*(x[1] - x[0]*th)],
                         [x[1] - x[0]*th]])

    def hessian(self, x=None, t=None):
        
        th = math.tanh(t)
        
        return np.array([[1 + th**2, -th], [-th, 1]])

    def time_derivative(self, x, t, der="tx"):

        # parse the derivative order
        der = ''.join(der.split()).lower()
                
        # time der. gradient
        if der == "tx" or der == 'xt':
            x = x.flatten()
            return np.array([[math.exp(math.cos(t))*math.sin(t) - (x[1] - 2*x[0]*math.tanh(t))/math.cosh(t)**2], 
                             [-x[0]/math.cosh(t)**2]]).reshape((-1,1))
        else:
            return super().time_derivative(x, t, der=der)
    
    def approximate_time_derivative(self, x, t, der="tx"):
        
        return super().time_derivative(x, t, der=der)


#%% UTILITY FUNCTIONS

def compute_proximal(f, x, penalty, solver=None, **kwargs):
    """
    Compute the proximal of a cost.
    
    This function (approximately) computes the proximal of a given cost if 
    there is no closed form solution. The function uses either a Newton method 
    or a gradient method, both with backtracking line search.

    Parameters
    ----------
    f : Cost
        The static cost whose proximal is required.
    x : array_like
        Where the proximal has to be evaluated.
    penalty : float
        The penalty of the proximal.
    solver : str, optional
        The method to use for computing the proximal, Newton or gradient. If 
        not specified, Newton is used for twice differentiable function, 
        gradient otherwise.
    **kwargs : dict
        Parameters for the Newton or gradient method.

    Returns
    -------
    y : ndarray
        The proximal.
    
    See Also
    --------
    solvers.backtracking_gradient
    solvers.newton
    """
    
    # generate cost of the proximal problem
    problem = {"f":f + Quadratic(np.eye(f.dom.size)/penalty, -x/penalty, utils.square_norm(x)/(2*penalty))}
    
    if solver is None and f.smooth >= 2: solver = "newton"
    
    if solver == "newton" or solver == "n":
        y = solvers.newton(problem, **kwargs)
    else:
        y = solvers.backtracking_gradient(problem, **kwargs)
    
    return y

def backward_finite_difference(signal, t, order=1, step=1):
    r"""
    Compute the backward finite difference of a signal.
    
    This function computes an approximate derivative of a given signal using
    backward finite differences. Given the signal :math:`s(t)`, it computes:
    
        .. math:: s^o(t) = \sum_{i = 0}^o (-1)^i {o \choose i} s(t - i T_s) / T_s^o
    
    where :math:`o \in \mathbb{N}` is the derivative order and :math:`T_s` is 
    the sampling time, see [#]_ for more details.
    
    Notice that if samples before :math:`t = 0` are required, they are set to 
    zero.

    Parameters
    ----------
    signal
        A function of a single scalar argument that represents the signal.
    t : float
        The time where the derivative should be evaluated.
    order : int, optional
        The derivative order, defaults to 1.
    step : float, optional
        The sampling time, defaults to 1.

    Raises
    ------
    ValueError
        For invalid `order` or `step` arguments.

    Returns
    -------
    ndarray
        The approximate derivative.
    
    References
    ----------
    .. [#] A. Quarteroni, R. Sacco, and F. Saleri, Numerical mathematics, 2nd 
           ed. Berlin; New York: Springer, 2007.
    """
        
    if order < 1: raise ValueError("`order` must be a positive integer.")
    if step <= 0: raise ValueError("`step` must be positive.")
    
    terms = []
    for o in range(order+1):
        v = t - o*step # next time
        c = (-1)**o * binom(order, o)
        
        # data point
        if v < 0: terms.append(c*signal(0))
        else: terms.append(c*signal(v))
        
    return sum(terms) / step**order