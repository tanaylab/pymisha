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
elif os.environ.get('PYMISHA_UBSAN'):
    # PYMISHA_UBSAN=1 pip install -e . --no-build-isolation
    # UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 python -m pytest tests/... -n0
    #
    # UBSan's runtime links into a shared object cleanly, so pytest just works -
    # unlike ASan, whose runtime must be first in link order and therefore needs
    # LD_PRELOAD against a CPython that was not built with it.
    #
    # -UNDEBUG matters as much as the sanitizer: the default build passes
    # -DNDEBUG, so every assert() in the shared core - including the size checks
    # in QuadTreeReader.h - is a no-op in CI, in the wheels and in conda.
    compile_args += ['-O1', '-g', '-UNDEBUG', '-fno-omit-frame-pointer',
                     '-fsanitize=undefined', '-fsanitize=float-divide-by-zero']
    link_args += ['-fsanitize=undefined']
elif os.environ.get('PYMISHA_ASSERTS'):
    # PYMISHA_ASSERTS=1 pip install -e . --no-build-isolation
    #
    # The half of the sanitizer story that needs no sanitizer runtime, so it runs
    # anywhere: turn the asserts back on. The normal build passes -DNDEBUG, which
    # makes every assert() in the shared core a no-op - including the size checks
    # in QuadTreeReader.h - in CI, in the wheels and in conda alike.
    # _GLIBCXX_ASSERTIONS adds libstdc++'s own bounds checks on vector/string.
    compile_args += ['-O1', '-g', '-UNDEBUG', '-D_GLIBCXX_ASSERTIONS']
else:
    compile_args += ['-O2']

setup(
    ext_modules=[
        Extension('_pymisha',
            sources=src_files,
            include_dirs=[numpy.get_include(), 'src'],
            libraries=['z'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
    ],
    zip_safe=False,
)
