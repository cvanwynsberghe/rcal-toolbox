# -*- coding: utf-8 -*-
"""
Created on Aug 20 2018

@author: Charles Vanwynsberghe

Simple simulation example for robust TOA calibration by RMDU
"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rcbox.rmds import RMDU
from rcbox.utils import make_TOA

np.random.seed(2121)

# %% set up data

Xr = np.random.randn(37, 3)
Xs = 2*np.random.randn(10, 3)
X_all = np.concatenate((Xr, Xs), axis=0)

c0 = 1.

toa = make_TOA(Xs, Xr, c0=1.)

# add outliers to toa
K = 50

# outlier values follow Rayleigh distribution with mode = toa ground truth
toa_noisy_table = np.random.rayleigh(scale=toa)

mask = np.zeros((toa.size))
mask[0:K] = 1
np.random.shuffle(mask)
mask = mask.reshape(toa.shape).astype(bool)

toa_noisy = toa.copy()
toa_noisy[mask] = toa_noisy_table[mask]
o = toa_noisy - toa
e = 0.001*np.random.randn(*toa.shape)

toa_noisy += e

X_ini = X_all + 0.01*np.random.randn(*X_all.shape)

# %% find geometry from TOA matrix

# least square case (lambda = infinity)
toa_solver = RMDU(c0*toa_noisy)
toa_solver.Run(lbda=np.inf, Xinit=X_ini, itmax=15000, EpsLim=1e-6, verbose=0)
toa_solver.align(X_all)

# outlier-aware case
rtoa_solver = RMDU(c0*toa_noisy)
rtoa_solver.Run(lbda=0.15, Xinit=X_ini, itmax=15000, EpsLim=1e-6, verbose=0)
rtoa_solver.align(X_all)

# %% plot result

fig = plt.figure(1, figsize=(8, 3.5))
fig.suptitle("MDU example")

ax_ls = fig.add_subplot(121, projection='3d')
ax_ls.scatter(*toa_solver.x_aligned[0:toa_solver.Nr, :].T,
              marker='x', facecolor='RoyalBlue',
              color=(0, 0, 0, 0), s=35, linewidths=1.5, label='r estimated')

ax_ls.scatter(*toa_solver.x_aligned[toa_solver.Nr::, :].T,
              marker='x', facecolor='red',
              color=(0, 0, 0, 0), s=35, linewidths=1.5, label='r estimated')

ax_ls.scatter(*toa_solver.x_ref_centered[0:toa_solver.Nr, :].T, marker='o',
              facecolor=(0, 0, 0, 0),
              color='RoyalBlue', s=35, linewidths=1.5, label='r source')

ax_ls.scatter(*toa_solver.x_ref_centered[toa_solver.Nr::, :].T, marker='o',
              facecolor=(0, 0, 0, 0),
              color='red', s=35, linewidths=1.5, label='r source')

ax_oa = fig.add_subplot(122, projection='3d')
ax_oa.scatter(*rtoa_solver.x_aligned[0:rtoa_solver.Nr, :].T,
              marker='x', facecolor='RoyalBlue',
              color=(0, 0, 0, 0), s=35, linewidths=1.5, label='r estimated')

ax_oa.scatter(*rtoa_solver.x_aligned[rtoa_solver.Nr::, :].T,
              marker='x', facecolor='red',
              color=(0, 0, 0, 0), s=35, linewidths=1.5, label='r estimated')

ax_oa.scatter(*rtoa_solver.x_ref_centered[0:rtoa_solver.Nr, :].T, marker='o',
              facecolor=(0, 0, 0, 0),
              color='RoyalBlue', s=35, linewidths=1.5, label='r source')

ax_oa.scatter(*rtoa_solver.x_ref_centered[rtoa_solver.Nr::, :].T, marker='o',
              facecolor=(0, 0, 0, 0),
              color='red', s=35, linewidths=1.5, label='r source')

plt.savefig("example_toa_geometries.pdf")

# %% L curve choice of o
rtoa_solver.tune_lambda(np.geomspace(10, 0.001, 200), Xinit=X_ini,
                        EpsLim=10**-6, itmax=10000)

plt.figure(2)
plt.plot(rtoa_solver.lambda_sequence, rtoa_solver.k_seq)
plt.axhline(K, color='k', linestyle='dashed')
plt.axhline(rtoa_solver.Nr*rtoa_solver.Ns,
            color='grey', linestyle='dashed')

plt.xlabel(r"$\lambda$")
plt.ylabel(r"$||\mathbf{O}||_0$")

plt.savefig("example_toa_Lcurve.pdf")
