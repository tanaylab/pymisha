import glob

import numpy
from setuptools import Extension, setup

# Get source files
src_files = glob.glob('src/*.cpp')

setup(
    name='pymisha',
    version='0.1.0',
    description='Python wrapper for misha - Genomic Data Analysis Toolkit',
    author='Aviezer Lifshitz',
    author_email='aviezerl@weizmann.ac.il',
    url='https://github.com/tanaylab/pymisha',
    install_requires=['pandas', 'numpy'],
    py_modules=['pymisha'],
    packages=['pymisha'],
    ext_modules=[
        Extension('_pymisha',
            sources=src_files,
            include_dirs=[numpy.get_include(), 'src'],
            extra_compile_args=[
                '-std=c++17',
                '-Wno-unused-function',
                '-Wno-switch',
                '-Wno-strict-aliasing',
                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
                '-DPYTHON'
            ],
        ),
    ],
    python_requires='>=3.10',
    zip_safe=False
)
