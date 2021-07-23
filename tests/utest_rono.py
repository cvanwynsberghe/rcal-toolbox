# -*- coding: utf-8 -*-
"""
Created on Sep 21 2015

@author: Charles Vanwynsberghe

Unit Tests for tdoa functions implemented in C
Use: py.test test_filename.py
"""
import numpy as np
from numpy.testing import assert_allclose

from ..rtdoa import ROno
from ..utils import make_TDOA

np.random.seed(1)

X = np.random.randn(12, 3)
S = 2*np.random.randn(7, 3)
tau = make_TDOA(S, X)

r0 = X + 0.3*np.random.randn(*X.shape)
s0 = S + 0.3*np.random.randn(*S.shape)


def test_rono_py_cy():
    """
    Compare python and cython implementations

    """
    # python implementation
    rono_py = ROno(tau=tau, c0=340., fast=False)
    rono_py.Init(r0=r0, s0=s0)
    rono_py.Run(0.01, itmax=10, verbose=0)

    # C implementation
    rono_cy = ROno(tau=tau, c0=340., fast=True)
    rono_cy.Init(r0=r0, s0=s0)
    rono_cy.Run(0.01, itmax=10, verbose=0)

    assert_allclose(rono_cy.r_iters, rono_py.r_iters)
    assert_allclose(rono_cy.s_iters, rono_py.s_iters)
    assert_allclose(rono_cy.o, rono_py.o)
