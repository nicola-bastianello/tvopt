#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online linear regression.
"""

import numpy as np
from numpy import linalg as la
from numpy. random import default_rng
import matplotlib.pyplot as plt

ran = default_rng()


from tvopt import utils, costs, solvers



#%% PROBLEM FORMULATION

# ------ problem data
n = 5 # signal dimension

t_s = 0.1 # sampling time (in seconds)
t_max = 5 # duration (in seconds)

# ------ signal
# sparsity: choose the n/2 components of the signal that will be zero
sp = ran.choice(range(n), size=n//2, replace=False)

# generate the signal
x_gt = np.zeros((n, int(t_max/t_s)))
for i in range(n):
    if i not in sp:
        x_gt[i,...] = 5*np.sin(np.arange(0, t_max, t_s)/10 + np.pi*ran.random())

# ------ cost functions
A = utils.random_matrix((100-1)*ran.random(n)+1) # observation matrix (static)
# noisy signal observations
b = [A.dot(x_gt[...,[k]]) + ran.standard_normal((n,1)) for k in range(int(t_max/t_s))]
# linear regression
f = costs.DiscreteDynamicCost([costs.LinearRegression(A, b[k]) for k in range(int(t_max/t_s))], t_s=t_s)

# sparsity promoting ell_1 norm
g = costs.Norm_1(n, weight=2)

# ------ solver parameters
# smoothness and strong convexity moduli
eigs = la.eigvalsh(A.T.dot(A))
L, mu = np.max(eigs), np.min(eigs)
# parameters for best convergence rates of FBS and PRS
step = 2 / (L+mu)
penalty = 1 / np.sqrt(L*mu)

# num. iteration applied to each sampled problem
num_iter = 5

# initial condition
x_0 = 10*ran.standard_normal((n, 1))
    

#%% SOLVE

# initialize trajectory
x_fbs = utils.initialize_trajectory(x_0, f.dom.shape, f.time.num_samples)
x_prs = utils.initialize_trajectory(x_0, f.dom.shape, f.time.num_samples)
z = np.zeros(f.dom.shape)


# (approximately) solve the problem
for k in range(f.time.num_samples):
    
    # sample the problem
    p = {"f":f.sample(k*t_s), "g":g}
    
    # apply the FBS solver
    x_fbs[...,k+1] = solvers.fbs(p, step, x_0=x_fbs[...,k], num_iter=num_iter)
    
    # apply the PRS solver
    x_prs[...,k+1], z = solvers.prs(p, penalty, x_0=z, num_iter=num_iter)
    
    
    utils.print_progress(k+1, f.time.num_samples)


#%% PLOT RESULTS

plt.rc("text", usetex=True), plt.rc("font", family="serif")
fontsize = 18

time = np.arange(0, t_max, t_s)


plt.figure()

plt.semilogy(time, utils.fpr(x_fbs), label="FBS")
plt.semilogy(time, utils.fpr(x_prs), label="PRS")

plt.legend(fontsize=fontsize-3)
plt.xlabel("Time [s]", fontsize=fontsize)
plt.ylabel("Fixed point residual", fontsize=fontsize)

plt.grid()

plt.show()
# plt.savefig("online-regression.pdf", bbox_inches="tight")