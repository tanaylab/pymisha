"""verror() must never return, and a failed call must not change later ones.

Two defects, one root cause: the reference count was deciding whether an error
function stopped execution.

P2 - verror() called handle_error() when s_ref_count was 0, which sets a Python
error and RETURNS. The caller then ran on with an exception set and invalid
state: pm_interv_register() reached g_pmdb->register_interv() through a null
g_pmdb and segfaulted the interpreter.

P1 - PyMisha's constructor tested check_db after s_ref_count++, so the throw
skipped the destructor and leaked the count. Every later call then skipped the
whole init block, permanently changing behaviour.

Both run in a subprocess: a regression on P2 is a SIGSEGV, which would take the
test session with it.
"""

import os
import subprocess
import sys
import tempfile

import pymisha as pm  # noqa: F401  (import check only)


def _run(body):
    return subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, timeout=120
    )


def test_verror_with_no_pymisha_on_the_stack_does_not_crash():
    """pm_interv_register() before gdb_init(): SIGSEGV before the fix."""
    r = _run(
        "import _pymisha\n"
        "try:\n"
        "    _pymisha.pm_interv_register('x')\n"
        "    print('RETURNED')\n"
        "except BaseException as e:\n"
        "    print(type(e).__name__)\n"
    )
    assert r.returncode == 0, f"interpreter died: {r.returncode}\n{r.stderr}"
    assert r.stdout.strip() == "error", r.stdout


def test_a_failed_check_db_call_does_not_change_the_next_call():
    """The leaked ref count used to make the same call behave differently after."""
    path = os.path.join(tempfile.mkdtemp(), "bad.wig")
    with open(path, "w") as fh:
        fh.write("fixedStep chrom=chr1 start=1 step=1\n1.0\nNOT_A_NUMBER\n")

    r = _run(
        "import _pymisha\n"
        f"path = {path!r}\n"
        "def probe():\n"
        "    try:\n"
        "        _pymisha.pm_parse_wig_or_bedgraph(path)\n"
        "        return 'RETURNED'\n"
        "    except BaseException as e:\n"
        "        return type(e).__name__\n"
        "print(probe())\n"
        "try:\n"
        "    _pymisha.pm_track_info('whatever')\n"
        "except BaseException:\n"
        "    pass\n"
        "print(probe())\n"
    )
    assert r.returncode == 0, r.stderr
    before, after = r.stdout.split()
    # Before the fix: SystemError, then RuntimeError.
    assert before == after == "error", r.stdout


def test_malformed_wig_raises_instead_of_returning_a_dict():
    """It used to parse past the bad line and return a dict with an error set."""
    path = os.path.join(tempfile.mkdtemp(), "bad.wig")
    with open(path, "w") as fh:
        fh.write("fixedStep chrom=chr1 start=1 step=1\n1.0\nNOT_A_NUMBER\n2.0\n")

    import _pymisha

    try:
        _pymisha.pm_parse_wig_or_bedgraph(path)
    except pm.error as exc:
        assert "Cannot parse" in str(exc)
    else:
        raise AssertionError("expected pymisha.error")
