#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic average consensus.
"""

import numpy as np
from numpy import linalg as la
from numpy.random import default_rng
import matplotlib.pyplot as plt

ran = default_rng()


from tvopt import networks


#%% PROBLEM FORMULATION

# -------- create the network
N = 25 # num. of nodes

adj_mat = networks.random_graph(N, 0.5)
net = networks.Network(adj_mat)

# -------- dynamic signals
# (piecewise constant signals - they change 10 times and are constant for 10
# consecutive sampling times)
r = []
for i in range(10):
    r_k = 2*ran.standard_normal((1, 1, N))
    for i in range(10): r.append(r_k)

# -------- dynamic average
r_avg = [np.mean(r_k, axis=-1) for r_k in r]


#%% DYNAMIC CONSENSUS

x = np.zeros(r[0].shape + (len(r)+1,)) # states trajectory
x[...,0] = r[0]


for k in range(1,len(r)):
    
    x[...,k] = net.consensus(x[...,k-1]) + r[k] - r[k-1]

# -------- compute distance from average
err = [la.norm(x[...,k] - np.stack([r_avg[k] for _ in range(N)])) for k in range(len(r))]


#%% PLOT RESULTS

plt.rc("text", usetex=True), plt.rc("font", family="serif")
fontsize = 18


plt.figure()

plt.plot(err)

plt.xlabel("Iteration num.", fontsize=fontsize)
plt.ylabel("Distance from consensus", fontsize=fontsize)

plt.grid()

plt.show()
# plt.savefig("dynamic-consensus.pdf", bbox_inches="tight")