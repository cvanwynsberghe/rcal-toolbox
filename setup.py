#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


extensions = [
    Extension("rmds_cythoned", ["rcbox/rmds_cythoned.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"]),
    Extension("rtdoa_cythoned", ["rcbox/rtdoa_cythoned.pyx"], extra_compile_args = ["-ffast-math"]),
    #, extra_compile_args = ["-ffast-math"])
]
 
setup(
    setup_cfg=True,
    cmdclass = {'build_ext':build_ext},
    ext_modules = cythonize(extensions),
    ext_package='rcbox',
    include_dirs = [numpy.get_include()],
    test_suite = 'tests',
)

# Run this line for compilation:
# python setup.py build_ext --inplace

# Run this line to clean generated binaries and C
# rm -r *.c *.so build

