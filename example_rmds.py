# -*- coding: utf-8 -*-
"""
Created on Aug 20 2018

@author: Charles Vanwynsberghe

Simple simulation example for robust diffuse field calibration by RMDS
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
from scipy.spatial.distance import squareform

from rcbox.utils import get_D
from rcbox.rmds import RMDS

np.random.seed(3212)

# %% set up data

X = np.random.randn(16, 3)
D = get_D(X)
X_ini = X + 0.1*np.random.randn(*X.shape)

# add outliers to toa
K = 20

# outlier values follow Rayleigh distribution with mode = D ground truth
D_noisy_table = np.random.rayleigh(scale=np.triu(D))
D_noisy_table += D_noisy_table.T

mask = np.zeros((int((D.size - D.shape[0])/2),))
mask[0:K] = 1
np.random.shuffle(mask)
mask = squareform(mask).astype(bool)

D_noisy = D.copy()
D_noisy[mask] = D_noisy_table[mask]
o = D_noisy - D
e = np.triu(0.01*np.random.randn(*D.shape), k=1)
e = e + e.T

D_noisy = D_noisy + e

# %% find geometry from euclidean distance matrix

# least square case (lambda = infinity)
solver_ls = RMDS(D_noisy)
solver_ls.Run(lbda=np.inf, Xinit=X_ini, EpsLim=10**-6, itmax=10000)
solver_ls.align(X)

# outlier-aware case
solver_oa = RMDS(D_noisy)
solver_oa.Run(lbda=0.2, Xinit=X_ini, EpsLim=10**-6, itmax=10000)
solver_oa.align(X)

# %% plot result

fig = plt.figure(1, figsize=(8, 3.5))
fig.suptitle("MDS example")

ax_ls = fig.add_subplot(121, projection='3d')
ax_ls.set_title("least-square (smacof)")

ax_ls.scatter(*solver_ls.x_aligned.T, marker='x', facecolor='RoyalBlue',
              color='none', s=35, linewidths=1.5, label='r estimated')

ax_ls.scatter(*solver_ls.x_ref_centered.T, marker='o', facecolor='none',
              color='RoyalBlue', s=35, linewidths=1.5, label='r source')

ax_oa = fig.add_subplot(122, projection='3d')
ax_oa.set_title("outlier-aware")

ax_oa.scatter(*solver_oa.x_aligned.T, marker='x', facecolor='RoyalBlue',
              color='none', s=35, linewidths=1.5, label='r estimated')

ax_oa.scatter(*solver_oa.x_ref_centered.T, marker='o', facecolor='none',
              color='RoyalBlue', s=35, linewidths=1.5, label='r source')

plt.savefig("example_rmds_geometries.pdf")

# %% L-curve choice for lambda

solver_oa.tune_lambda(np.geomspace(1, 0.01, 200), Xinit=X_ini,
                      EpsLim=10**-6, itmax=10000)

plt.figure()
plt.plot(solver_oa.lambda_sequence, 2*solver_oa.k_seq)
plt.axhline(2*K, color='k', linestyle='dashed')
plt.axhline(solver_oa.Nr*(solver_oa.Nr-1), color='grey', linestyle='dashed')

plt.xlabel(r"$\lambda$")
plt.ylabel(r"$||\mathbf{O}||_0$")

plt.savefig("example_rmds_Lcurve.pdf")
