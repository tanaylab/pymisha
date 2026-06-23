"""Per-worker test-DB path, so the suite can run under ``pytest-xdist``.

Most tests mutate a single on-disk misha DB (creating/removing tracks, calling
``pm_dbreload``) and several files reuse the same track names. That is safe
serially but not across xdist worker *processes*, which would write into the
same directory concurrently. Under xdist each worker gets its own copy of the
canonical DB; a plain serial run uses the canonical one in place.

All test modules resolve the DB root through ``TESTDB_ROOT`` here instead of
recomputing ``Path(__file__)/...`` so the redirect applies everywhere at once.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_CANONICAL = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _worker_copy(prefix: str) -> Path:
    """A per-worker copy of the canonical DB under a unique temp dir.

    Resolve symlinks (on macOS gettempdir() is /var/... -> /private/var/...);
    misha returns realpaths, so tests comparing against this path must match.
    """
    worker = os.environ["PYTEST_XDIST_WORKER"]  # e.g. "gw0"
    dst = Path(tempfile.gettempdir()) / f"{prefix}_{os.getpid()}_{worker}" / "test"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_CANONICAL, dst)
        atexit.register(shutil.rmtree, dst.parent, ignore_errors=True)
    return dst.resolve()


def _resolve() -> Path:
    if not os.environ.get("PYTEST_XDIST_WORKER"):  # unset when serial
        return _CANONICAL
    return _worker_copy("pymisha_testdb")


def _resolve_examples() -> Path:
    # gdb_init_examples()/gdb_examples_path() must NOT point at the canonical
    # (git-tracked) tree under xdist: several tests gdb_init onto the examples DB
    # and then write tracks, which would mutate the shared tree AND race with
    # another worker's gdb_init_examples copytree (file vanishes mid-copy ->
    # "No such file or directory"). Give each worker its own pristine copy.
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return _CANONICAL
    return _worker_copy("pymisha_examples")


TESTDB_ROOT = _resolve()
os.environ["PYMISHA_EXAMPLES_DB"] = str(_resolve_examples())
