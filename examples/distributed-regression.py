#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distributed, online linear regression.
"""

import numpy as np
from numpy. random import default_rng
import matplotlib.pyplot as plt

ran = default_rng()


from tvopt import utils, costs, networks, distributed_solvers


#%% GENERATE PROBLEM (PRIMAL SOLVERS)

# -------- create the network
N = 15

adj_mat = networks.random_graph(N, 0.1)
net = networks.Network(adj_mat)

# -------- create ground truth signal
t_s = 0.1
t_max = 10

omega = 2
s = np.cos(omega*np.arange(0, t_max, t_s))

sd = 1e-5 # noise standard deviation

# -------- create nodes' costs
A = 2*ran.random(N)
f_k = []
for k in range(int(t_max/t_s)):    
    f_i = []
    for i in range(N):
        
        b = A[i]*s[k] + sd*ran.standard_normal() # noisy observation
        f_i.append(costs.LinearRegression(A[i], b))    
    f_k.append(costs.SeparableCost(f_i))

f = costs.DiscreteDynamicCost(f_k, t_s=t_s)

# -------- solver parameters
step = 0.2

x0 = 100*ran.standard_normal(N)

num_iter = 5


#%% TEST PRIMAL SOLVERS

# -------- DGD
x = np.zeros((N, f.time.num_samples+1))
x[...,0] = x0

for k in range(f.time.num_samples):
    
    problem = {"f":f.sample(t_s*k), "network":net}
    
    x[...,k+1] = distributed_solvers.dpgm(problem, step, x_0=x[...,k], num_iter=num_iter)
    
    
    utils.print_progress(k+1, f.time.num_samples)

# -------- EXTRA
y = np.zeros((N, f.time.num_samples+1))
x[...,0] = x0

for k in range(f.time.num_samples):
    
    problem = {"f":f.sample(t_s*k), "network":net}
    
    y[...,k+1] = distributed_solvers.pg_extra(problem, step, x_0=y[...,k], num_iter=num_iter)
    
    
    utils.print_progress(k+1, f.time.num_samples)


#%% PLOT RESULTS

plt.rc("text", usetex=True), plt.rc("font", family="serif")
fontsize = 18

time = np.arange(0, t_max, t_s)
opt = np.vstack([s.reshape((1,-1)) for _ in range(N)]) # ground truth


plt.figure()
        
plt.semilogy(time, utils.dist(x[...,1:], opt), label="DGD", marker="s", markevery=[1])
plt.semilogy(time, utils.dist(y[...,1:], opt), label="EXTRA", marker="v", markevery=[1])

plt.legend(ncol=2, fontsize=fontsize-3)
plt.xlabel("Time [s]", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)

plt.grid()

plt.show()
# plt.savefig("distributed-regression.pdf", bbox_inches="tight")