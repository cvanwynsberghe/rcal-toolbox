#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Feb 23 2016

@author: Charles Vanwynsberghe

Unit Tests for rmds functions implemented in cython
Use: py.test test_filename.py
"""


import numpy as np
from numpy.testing import assert_allclose

from rcbox.utils import get_D, make_TOA
from rcbox.rmds import rmdsw, RMDS, RMDU
from rcbox import rmds_cythoned

np.random.seed(1)

X = np.random.randn(16, 3)
D = get_D(X)
X_ini = X + 0.1*np.random.randn(*X.shape)

# add some outliers
D[(1, 3, 7), (2, 4, 8)] = [5., 7., 3.]
D[(2, 4, 8), (1, 3, 7)] = [5., 7., 3.]


def test_rmds_py_cy_function():
    """
    Compare rmdsw python function and rmds_cythoned function.

    """
    X_py, O_py, Eps_py = rmdsw(D, lbda=0.5, Ndim=3, W=None,
                               Xinit=X_ini, Maxit=10,
                               EpsLim=10**-6,
                               EpsType="Forero2012", verbose=0)

    X_cy, O_cy, Eps_cy = rmds_cythoned.rmdsw(D, lbda=0.5, Ndim=3, W=None,
                                             Xinit=X_ini, Maxit=10,
                                             EpsLim=10**-6,
                                             EpsType="Forero2012", verbose=0)

    assert_allclose(X_py, X_cy)
    assert_allclose(O_py, O_cy)
    assert_allclose(Eps_py, Eps_cy)


def test_rmds_class():
    """ Compare RMDS class and rmds_cythoned function. """

    X_cy, O_cy, Eps_cy = rmds_cythoned.rmdsw(D, lbda=0.5, Ndim=3, W=None,
                                             Xinit=X_ini, Maxit=10,
                                             EpsLim=10**-6,
                                             EpsType="Forero2012", verbose=0)

    solver = RMDS(D)
    solver.Run(lbda=0.5, Xinit=X_ini, EpsLim=10**-6, itmax=10)

    assert_allclose(X_cy, solver.X)
    assert_allclose(Eps_cy, solver.Eps)
    assert_allclose(O_cy, solver.O)


def test_rmdu_fast_inverse():
    """ Compare RMDU Lpinv compuation fast/slow. """

    Xr = np.random.randn(21, 3)
    Xs = 2*np.random.randn(17, 3)
    toa = make_TOA(Xs, Xr, c0=1.)
    slow_solver = RMDU(toa, fast_Linv=False)
    fast_solver = RMDU(toa, fast_Linv=True)

    assert_allclose(slow_solver.Lpinv, fast_solver.Lpinv, atol=1e-2)
