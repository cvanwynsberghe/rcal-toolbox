from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
 
extensions = [
    Extension("rmds_cythoned", ["rmds_cythoned.pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"]),
    Extension("rtdoa_cythoned", ["rtdoa_cythoned.pyx"], extra_compile_args = ["-ffast-math"]),
    #, extra_compile_args = ["-ffast-math"])
]
 
setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = cythonize(extensions),
    include_dirs = [numpy.get_include()] #Include directory not hard-wired
)

# Run this line for compilation:
# python setup.py build_ext --inplace

# Run this line to clean generated binaries and C
# rm -r *.c *.so build

