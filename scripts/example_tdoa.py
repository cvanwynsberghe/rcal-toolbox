# -*- coding: utf-8 -*-
"""
Created on Aug 20 2018

@author: Charles Vanwynsberghe

Simple simulation example for robust TDOA calibration
"""
import numpy as np
import matplotlib.pyplot as plt

from rcbox.rtdoa import ROno, denoise_tdoa_soft
from rcbox.utils import make_TDOA

np.random.seed(1234)  # fix random number generator

# %% set up variables

N = 30  # number of sources
M = 20  # number of microphones
K = 20  # number of outlier per TDOA matrix (sparsity with K << M*(M-1)/2)
sig_o = 10  # standard deviation of outlier amplitudes
reg_l1 = 0.5

# %% generate geometry and TDOA

X = np.random.randn(M, 3)
S = 2*np.random.randn(N, 3)

tau = make_TDOA(S, X, c0=1.)

# generate initial guess
r0 = X + 0.5*np.random.randn(*X.shape)
s0 = S + 0.5*np.random.randn(*S.shape)

# add outliers to tau
o = np.zeros_like(tau)
idx = np.triu_indices(X.shape[0], k=1)

for n in range(o.shape[0]):
    a = np.arange(idx[0].size)
    np.random.shuffle(a)
    for k in range(K):
        idx_k = (idx[0][a[k]], idx[1][a[k]])
        o[n][idx_k] = sig_o*np.random.randn()

    o[n, ...] = o[n, ...] - o[n, ...].T

tau_noisy = tau + o

# %% compare TDOA solvers

fig = plt.figure(1, figsize=(14, 4))
fig.suptitle("TDOA example")

ax_0 = fig.add_subplot(131, projection='3d')
ax_1 = fig.add_subplot(132, projection='3d')
ax_2 = fig.add_subplot(133, projection='3d')

# Case 1: baseline
ono = ROno(tau=tau_noisy, c0=1., fast=True)
ono.Init(r0=r0, s0=s0)
ono.Run(np.inf, param_converge=1e-6, verbose=0)
ono.Plot_result(r_gt=X, ax=ax_0)  # , show_init=True, show_iters=True)
ax_0.set_title("TDOA - baseline\n")

# Case2: outlier-aware denoising + baseline
tau_denoised = np.zeros_like(tau_noisy)
for n in range(N):
    tau_denoised[n, ...], _, _ = denoise_tdoa_soft(tau_noisy[n, ...], reg_l1,
                                                   eps=1e-6, tmax=200)

dono = ROno(tau=tau_denoised, c0=1., fast=True)
dono.Init(r0=r0, s0=s0)
dono.Run(np.inf, param_converge=1e-6, verbose=0)
dono.Plot_result(r_gt=X, ax=ax_1)  # , show_init=True, show_iters=True)
ax_1.set_title("TDOA - denoising + baseline\n")

# case 3: outlier-aware R-Ono
rono = ROno(tau=tau_noisy, c0=1., fast=True)
rono.Init(r0=r0, s0=s0)
rono.Run(reg_l1, param_converge=1e-6, verbose=0)
rono.Plot_result(r_gt=X, ax=ax_2)  # , show_init=True, show_iters=True)
ax_2.set_title("TDOA - outlier-aware\n")

plt.savefig("example_tdoa_geometries.pdf")
