#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction-correction online optimization.
"""

import numpy as np
import matplotlib.pyplot as plt

from tvopt import costs, prediction, solvers, utils


#%% PROBLEM & PARAMETERS

# -------- problem setup
t_s, t_max = 0.1, 1e4 # sampling time and time horizon

# cost function
f = costs.DynamicExample_1D(t_s, t_max)

# -------- compute optimal trajectory
x_opt = np.zeros(f.time.num_samples)

print("Computing optimal solution ...")

for k in range(f.time.num_samples):
    
    x = x_opt[k-1] if k > 0 else 0
    
    x_opt[k] = solvers.newton({"f":f.sample(t_s*k)}, x_0=x, num_iter=1000, tol=1e-15)
    
    utils.print_progress(k+1, f.time.num_samples)

# -------- parameters
step = 0.15 # solver step-size

# num. of prediction and correction steps
num_pred, num_corr = 5, 5


#%% PREDICTION-CORRECTION METHODS

# -------- correction-only
print("Correction-only method ...")

x_co = np.zeros(f.time.num_samples+1)
x_hat = 0

for k in range(f.time.num_samples):
    
    # correction
    x_co[k+1] = solvers.gradient({"f":f.sample(t_s*k)}, x_0=x_co[k], step=step, num_iter=num_pred)


# -------- extrapolation-based
print("Extrapolation-based method ...")

x_ex = np.zeros(f.time.num_samples+1)
x_hat = 0

# cost prediction
f_hat = prediction.ExtrapolationPrediction(f, order=2)

for k in range(f.time.num_samples):
    
    # correction
    x_ex[k+1] = solvers.gradient({"f":f.sample(t_s*k)}, x_0=x_hat, step=step, num_iter=num_corr)
    
    # prediction
    f_hat.update(t_s*k)
    x_hat = solvers.gradient({"f":f_hat}, x_0=x_ex[k+1], step=step, num_iter=num_pred)


# -------- Taylor-based
print("Taylor expansion-based method ...")

x_ty = np.zeros(f.time.num_samples+1)
x_hat = 0

# cost prediction
f_hat = prediction.TaylorPrediction(f)

for k in range(f.time.num_samples):
    
    # correction
    x_ty[k+1] = solvers.gradient({"f":f.sample(t_s*k)}, x_0=x_hat, step=step, num_iter=num_corr)
    
    # prediction
    f_hat.update(t_s*k, x_hat)
    x_hat = solvers.gradient({"f":f_hat}, x_0=x_ty[k+1], step=step, num_iter=num_pred)


#%% PLOT RESULTS

plt.rc("text", usetex=True), plt.rc("font", family="serif")
fontsize = 18

time = t_s*np.arange(f.time.num_samples)


plt.figure()

plt.loglog(time, np.abs(x_co[1:]-x_opt), label="Correction-only", marker="s", markevery=[1])

plt.loglog(time, np.abs(x_ex[1:]-x_opt), label="Extrapolation", marker="v", markevery=[1])

plt.loglog(time, np.abs(x_ty[1:]-x_opt), label="Taylor", marker="D", markevery=[1])


plt.legend(ncol=2, fontsize=fontsize-3)
plt.xlabel("Time [s]", fontsize=fontsize)
plt.ylabel("Tracking error", fontsize=fontsize)

plt.grid()

plt.show()
# plt.savefig("prediction-correction.pdf", bbox_inches="tight")