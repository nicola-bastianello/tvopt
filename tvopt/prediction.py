#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cost prediction tools.
"""

import numpy as np

from tvopt import costs


#%% PREDICTION TEMPLATE

class Prediction(costs.Cost):
    """
    Prediction of a dynamic cost.
    
    This class creates a cost object that predicts a given dynamic function.
    The object stores a dynamic cost and a predicted cost, which can be 
    modified using new information through the method `update`.
    """
    
    def __init__(self, cost):
        
        super().__init__(cost.dom)
        self.cost, self.t_s = cost, cost.time.t_s
        
        # variable to store the last time the prediction was updated
        self.t_k = None
    
    def update(self, t, *args, **kwargs):
        """
        Update the current prediction.
        
        This method updates the current prediction by building a new predicted
        cost using the samples observed up to time `t`. By default this method 
        samples the dynamic cost, and should be overwritten when implementing 
        a custom prediction strategy.

        Parameters
        ----------
        t : float
            The time of the last sampled cost.
        """
        
        if self.t_k != t:
            self.prediction = self.cost.sample(t)
            self.t_k = t

    def function(self, x, **kwargs):
        
        return self.prediction.function(x, **kwargs)
    
    def gradient(self, x, **kwargs):
        
        return self.prediction.gradient(x, **kwargs)
    
    def hessian(self, x, **kwargs):
        
        return self.prediction.hessian(x, **kwargs)
    
    def proximal(self, x, penalty=1, **kwargs):
        
        return self.prediction.proximal(x, penalty=penalty, **kwargs)


#%% EXAMPLES

class ExtrapolationPrediction(Prediction):
    r"""
    Extrapolation-based prediction.
    
    This prediction strategy, proposed in [#]_, predicts the cost at time
    :math:`t_{k+1}` as:
    
        .. math:: \hat{f}(\pmb{x};t_{k+1}) = \sum_{i = 1}^I \ell_i f(\pmb{x}; t_{k - i + 1})
    
    where :math:`I \in \mathbb{N}` denotes the order, that is, the number of
    past functions to use, and with coefficients:
        
        .. math:: \ell_i = \prod_{1 \leq j \leq I, \ j \neq i} \frac{j}{j - i}.
        
    .. [#] N. Bastianello, A. Simonetto, and R. Carli, "Primal and Dual 
           Prediction-Correction Methods for Time-Varying Convex Optimization,"
           arXiv:2004.11709 [cs, math], Oct. 2020. Available:
           http://arxiv.org/abs/2004.11709.
    """
    
    def __init__(self, cost, order=2):
        
        super().__init__(cost)
        self.smooth = 2
        
        # extrapolation data
        self.order, self.range = order, list(range(1, order+1))
        self.coeffs = {i : np.prod([j / (j - i) for j in self.range[:i-1] + self.range[i:]]) for i in self.range}
    
    def update(self, t):
        
        # update if the t_k is different from the last used
        if self.t_k != t:
            
            if t < self.order*self.t_s: self.prediction = self.cost.sample(t)
            else: self.prediction = sum([self.coeffs[i]*self.cost.sample(t-(i-1)*self.t_s) for i in self.range])
            
            self.t_k = t

class TaylorPrediction(Prediction):
    r"""
    Taylor expansion-based prediction.
    
    This prediction strategy, proposed in [#]_ and see also [#]_, predicts 
    the cost at time :math:`t_{k+1}` using its Taylor expansion around 
    :math:`t_k` and a given :math:`\pmb{x}_k`:
        
        .. math:: \begin{split} \hat{f}(\pmb{x}; t_{k+1}) &= f(\pmb{x}_k;t_k) + \langle \nabla_x f(\pmb{x}_k;t_k), \pmb{x} - \pmb{x}_k \rangle
                                            + T_s \nabla_t f(\pmb{x}_k;t_k) + (T_s^2 / 2) \nabla_{tt} f(\pmb{x}_k;t_k) +\\
                                            &+ T_s \langle \nabla_{tx} f(\pmb{x}_k;t_k), \pmb{x} - \pmb{x}_k \rangle 
                                            + \frac{1}{2} (\pmb{x} - \pmb{x}_k)^\top \nabla_{xx} f(\pmb{x}_k;t_k) (\pmb{x} - \pmb{x}_k)
                                            \end{split}
    
    where :math:`T_s` is the sampling time.
        
    
    References
    ----------
    .. [#] A. Simonetto, A. Mokhtari, A. Koppel, G. Leus, and A. Ribeiro, 
           "A Class of Prediction-Correction Methods for Time-Varying 
           Convex Optimization," IEEE Transactions on Signal Processing, 
           vol. 64, no. 17, pp. 4576–4591, Sep. 2016.
    .. [#] N. Bastianello, A. Simonetto, and R. Carli, "Primal and Dual 
           Prediction-Correction Methods for Time-Varying Convex Optimization,"
           arXiv:2004.11709 [cs, math], Oct. 2020. Available:
           http://arxiv.org/abs/2004.11709.
    """
    
    def __init__(self, cost):
        
        super().__init__(cost)
        self.smooth = 2
        
        # variables for where the last prediction was centered
        self.x_k = None
    
    def update(self, t, x, gradient_only=True, **kwargs):
        
        # update if the x_k or t_k are different from the last used
        if np.any(self.x_k != x) or self.t_k != t:
                        
            g, h = self.cost.gradient(x, t, **kwargs), self.cost.hessian(x, t, **kwargs)
            d = self.cost.time_derivative(x, t, der="xt", **kwargs) \
              + self.cost.time_derivative(x, t, der="tx", **kwargs)
            
            # linear and constant terms of the prediction function
            b = g - np.dot(h, x) + 0.5*self.t_s*d
            
            if gradient_only:
                c = 0
            else:
                c = self.cost.function(x, t, **kwargs) - \
                    (0.5*self.t_s*d + g).T.dot(x) + 0.5*x.T.dot(h.dot(x)) + \
                    0.5*self.t_s**2*self.cost.time_derivative(x, t, der="tt", **kwargs) + \
                    self.t_s*self.cost.time_derivative(x, t, der="t", **kwargs)
            
            # store new prediction
            self.prediction = costs.Quadratic(h, b, c)
            self.x_k, self.t_k = x, t