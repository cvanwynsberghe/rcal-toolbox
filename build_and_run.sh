#!/bin/bash

# build binaries
cd rcbox
python setup.py build_ext --inplace
cd ..

# run unit tests
py.test rcbox/unit_tests/utest_*

# run script examples & make figures
python example_rmds.py
python example_toa.py
python example_tdoa.py

