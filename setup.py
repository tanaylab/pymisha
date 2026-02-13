import glob
import os

import numpy
from setuptools import Extension, setup

# Get source files relative to setup.py location
here = os.path.abspath(os.path.dirname(__file__))
src_files = sorted(glob.glob(os.path.join(here, 'src', '*.cpp')))

setup(
    ext_modules=[
        Extension('_pymisha',
            sources=src_files,
            include_dirs=[numpy.get_include(), os.path.join(here, 'src')],
            extra_compile_args=[
                '-std=c++17',
                '-O2',
                '-Wno-unused-function',
                '-Wno-switch',
                '-Wno-strict-aliasing',
                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
                '-DPYTHON'
            ],
        ),
    ],
    zip_safe=False,
)
