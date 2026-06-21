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


def _resolve() -> Path:
    worker = os.environ.get("PYTEST_XDIST_WORKER")  # e.g. "gw0"; unset when serial
    if not worker:
        return _CANONICAL
    dst = Path(tempfile.gettempdir()) / f"pymisha_testdb_{os.getpid()}_{worker}" / "test"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_CANONICAL, dst)
        atexit.register(shutil.rmtree, dst.parent, ignore_errors=True)
    # Resolve symlinks (on macOS gettempdir() is /var/... -> /private/var/...);
    # misha returns realpaths, so tests comparing against this path must match.
    return dst.resolve()


TESTDB_ROOT = _resolve()

# gdb_init_examples() copies PYMISHA_EXAMPLES_DB into a throwaway temp DB; point
# it at the pristine canonical tree (never mutated during a run) so the copy
# can't race with a worker's own track churn (transient .trash.* files).
os.environ["PYMISHA_EXAMPLES_DB"] = str(_CANONICAL)
