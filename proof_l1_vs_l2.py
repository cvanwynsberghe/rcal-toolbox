# -*- coding: utf-8 -*-
"""
Created on Aug 20 2018

@author: Charles Vanwynsberghe

Simple simulation example for robust TDOA calibration
"""
import numpy as np
import matplotlib.pyplot as plt

from rcbox.rtdoa import denoise_tdoa_soft, denoise_tdoa_hard
from rcbox.utils import make_TDOA

np.random.seed(1234)  # fix random number generator

# %% set up variables

N = 2  # number of sources
M = 100  # number of microphones
K = 100  # number of outlier per TDOA matrix (sparsity with K << M*(M-1)/2)
sig_o = 10  # standard deviation of outlier amplitudes


# %% generate geometry and TDOA

X = np.zeros((M, 3))
X[:, 0] = np.linspace(0, 10, M)
S = np.zeros((N, 3))
S[:, 0] = [5, 10]
S[:, 1] = 5

tau = make_TDOA(S, X, c0=340.)

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


e = np.triu(0.2*np.random.randn(*tau.shape[1:3]), k=1)
e = e - e.T

tau_noisy = tau + o

tau_noisy[0] = tau_noisy[0] + e

# %% compare TDOA solvers

reg_l1 = np.geomspace(40, 0.2, 600)
o_s_0 = np.zeros_like(reg_l1)
for n, reg_l1_ in enumerate(reg_l1):
    _, o_s, _ = denoise_tdoa_soft(tau_noisy[0, ...], reg_l1_,
                                  eps=1e-6, tmax=200, checks=0)
    o_s_0[n] = (o_s != 0).sum()/2

k = np.arange(1, 150, 5)
o_h_0 = np.zeros_like(k)
for n, k_ in enumerate(k):
    _, o_h, _ = denoise_tdoa_hard(tau_noisy[0, ...], k_,
                                  eps=1e-6, tmax=200, checks=0)
    o_h_0[n] = (o_h != 0).sum()/2


# %%
plt.figure(1, figsize=(6, 3))

plt.suptitle(r"""
             TDOA predenoising : soft (left) & hard (right) thresholding
             $M = 100$ ; 100 outliers (2% of outliers)
             """, linespacing=1)

plt.subplot(121)
plt.plot(reg_l1, o_s_0)
plt.axhline(K, color="grey", linestyle="dashed")
plt.xlabel(r"$\ell_1$ regulizer $\lambda$")
plt.ylim(0, 400)
plt.xlim(-0.0, 15)
plt.subplot(122)
plt.plot(k, o_h_0)
plt.axhline(K, color="grey", linestyle="dashed")
plt.xlabel(r"$\ell_0$ regulizer $K$")
plt.axis("auto")

plt.subplots_adjust(top=0.80, bottom=0.2)
#plt.tight_layout()

plt.savefig("TDOA_l1_l0_difference_simulated.pdf")


#%%
import h5py

f5 = h5py.File("/home/vanwynch/rep/these_revision/SelfCalibration/"
               "TDOA/Robust_tdoa/LFV_tune_param.h5")

tau_exp = f5["TDOA"][0]

reg_l1 = np.geomspace(0.03, 0.0002, 50)
o_s_0 = np.zeros_like(reg_l1)
for n, reg_l1_ in enumerate(reg_l1):
    _, o_s, _ = denoise_tdoa_soft(tau_exp, reg_l1_,
                                  eps=1e-6, tmax=200, checks=0)
    o_s_0[n] = (o_s != 0).sum()/2

k = np.arange(1, 2500, 100)
o_h_0 = np.zeros_like(k)
for n, k_ in enumerate(k):
    _, o_h, _ = denoise_tdoa_hard(tau_exp, k_, eps=1e-6, tmax=200, checks=0)
    o_h_0[n] = (o_h != 0).sum()/2
    
plt.figure(2, figsize=(6, 3))

plt.suptitle(r"""
             TDOA predenoising : soft (left) & hard (right) thresholding
             $M = 256$ ; measured data
             """, linespacing=1)

plt.subplot(121)
plt.plot(reg_l1, o_s_0)
#plt.axhline(K, color="grey", linestyle="dashed")
plt.xlabel(r"$\ell_1$ regulizer $\lambda$")
plt.ylim(-50, 2500)
plt.xlim(-0.0005, 0.025)
plt.subplot(122)
plt.plot(k, o_h_0)
#plt.axhline(K, color="grey", linestyle="dashed")
plt.xlabel(r"$\ell_0$ regulizer $K$")
plt.axis("auto")

plt.subplots_adjust(top=0.80, bottom=0.2)
#plt.tight_layout()

plt.savefig("TDOA_l1_l0_difference_experimental.pdf")
