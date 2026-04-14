import glob
import os

import numpy
from setuptools import Extension, setup

src_files = sorted(glob.glob('src/*.cpp'))

compile_args = [
    '-std=c++17',
    '-Wno-unused-function',
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPYTHON',
]
link_args = []

if os.environ.get('PYMISHA_COVERAGE'):
    compile_args += ['-O0', '--coverage', '-fprofile-arcs', '-ftest-coverage']
    link_args += ['--coverage']
else:
    compile_args += ['-O2']

setup(
    ext_modules=[
        Extension('_pymisha',
            sources=src_files,
            include_dirs=[numpy.get_include(), 'src'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ],
    zip_safe=False,
)
