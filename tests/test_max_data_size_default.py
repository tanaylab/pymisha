"""The default CONFIG['max_data_size'] is R misha's default gmax.data.size
(calculate_optimal_gmax_data_size of get_system_memory, R/utils-memory.R), and a value set by the
user still applies to the Python and the C++ code."""
import os
import shutil
import subprocess
import sys
import types

import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha, _shared

# (RAM in bytes, detectCores(), calculate_optimal_gmax_data_size) from R misha 5.18.1, printed with
# %.17g, with parallel::detectCores replaced to return the cores (None: NA)
R_VALUES = [
    (8000000000, 4, 2800000000),
    (34359738368, 16, 2186528805.2363634),
    (274877906944, 128, 2161961065.851685),
    (1081349353472, 128, 8504994914.9483137),
    (16000000000, 128, 125842696.62921348),
    (1000000000000, 2, 10000000000),
    (8000000000, 1, 5600000000),
    (8000000000, None, 5600000000),
    (8000000000, 0, 5600000000),
    (4000000000, 3, 1400000000),
    (1000000007, 7, 175000001.22499999),
    (64000000000, 10, 6400000000),
    (123456789012, 96, 1289847049.3791044),
    (1099511627776, 46, 10000000000),
    (1099511627776, 47, 10000000000),
]


@pytest.mark.parametrize("sys_mem,cores,expected", R_VALUES)
def test_the_formula_gives_r_values(sys_mem, cores, expected):
    assert _shared._optimal_max_data_size(sys_mem, cores) == expected


def test_memory_and_cores_fall_back_as_r(monkeypatch):
    # another system: R's 8e9 bytes and detectCores() of NA
    monkeypatch.setattr(_shared, "_sys", types.SimpleNamespace(platform="sunos5"))
    assert (_shared._system_memory(), _shared._detect_cores()) == (8e9, None)

    # macOS: sysctl hw.memsize and hw.logicalcpu, or the fallbacks when sysctl fails
    monkeypatch.setattr(_shared, "_sys", types.SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(_shared, "_sysctl", {"hw.memsize": "17179869184\n", "hw.logicalcpu": "8\n"}.get)
    assert (_shared._system_memory(), _shared._detect_cores()) == (17179869184.0, 8)

    def fail(name):
        raise subprocess.CalledProcessError(1, ["sysctl", name])
    monkeypatch.setattr(_shared, "_sysctl", fail)
    assert (_shared._system_memory(), _shared._detect_cores()) == (8e9, None)

    # Linux without a readable /proc
    monkeypatch.setattr(_shared, "_sys", types.SimpleNamespace(platform="linux"))
    def no_proc(*args, **kwargs):
        raise OSError("no /proc")
    monkeypatch.setattr("builtins.open", no_proc)
    assert (_shared._system_memory(), _shared._detect_cores()) == (8e9, None)


@pytest.fixture(scope="module")
def fresh_default():
    """CONFIG['max_data_size'] in a new interpreter, as pymisha sets it at import."""
    root = os.path.dirname(os.path.dirname(pm.__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
    out = subprocess.run([sys.executable, "-c", "import pymisha; print(repr(pymisha.CONFIG['max_data_size']))"],
                         capture_output=True, text=True, env=env, check=True).stdout
    return float(out)


def test_the_default_is_set_at_import_from_this_host(fresh_default):
    expected = _shared._optimal_max_data_size(_shared._system_memory(), _shared._detect_cores())
    assert fresh_default == expected == _shared._LIMIT_DEFAULTS["max_data_size"]


def test_the_default_is_r_misha_default_on_this_host(fresh_default):
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript is not available")
    res = subprocess.run(["Rscript", "-e", "suppressMessages(library(misha)); "
                          "cat(sprintf('%.17g', getOption('gmax.data.size')))"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip("R misha is not installed")
    assert fresh_default == float(res.stdout)


def test_a_value_set_by_the_user_applies_to_python_and_cpp(monkeypatch):
    scope = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [20000]})   # 400 bins of dense_track
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 399)
    assert _shared._config_limit("max_data_size") == 399
    with pytest.raises(_pymisha.error, match=r"maximum allowed \(399 rows"):
        pm.gextract("dense_track", scope)
    # the Python sampler of a virtual track's quantiles
    pm.gvtrack_create("vt_max_data_size", "dense_track", func="avg")
    try:
        with pytest.warns(RuntimeWarning, match="approximate"):
            pm.gquantiles("vt_max_data_size", 0.5, scope, iterator=50)
    finally:
        pm.gvtrack_rm("vt_max_data_size")
