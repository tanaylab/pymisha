"""Logging and warning plumbing for pymisha.

pymisha recovers from a lot of exceptions on purpose: an optional dependency
that is not installed, a probe asking "is this a COMPUTED track", a
best-effort cleanup. Every such fallback reports what it swallowed through
the standard library's ``logging`` module, so the recovered-from failure is
discoverable instead of invisible.

Following the stdlib convention for libraries, the package attaches only a
:class:`logging.NullHandler` and never configures logging itself. Nothing is
emitted until the application asks for it::

    import logging
    logging.basicConfig()
    logging.getLogger("pymisha").setLevel(logging.DEBUG)

Each module logs to its own child logger (``pymisha.tracks``,
``pymisha.intervals``, ...), so a single module can be turned up on its own.

Failures that R misha itself warns about are reported as
:class:`PymishaWarning` instead, so they are visible without configuring
logging at all.
"""

from __future__ import annotations

import logging
import os
import sys
from types import FrameType

# A NullHandler on the package logger is what keeps the library inert: it
# stops logging's "last resort" handler from printing WARNING records to
# stderr on its own. It is attached to "pymisha", never to the root logger.
logging.getLogger("pymisha").addHandler(logging.NullHandler())

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class PymishaWarning(RuntimeWarning):
    """Category for warnings pymisha issues about a failure it recovered from.

    A subclass of :class:`RuntimeWarning`, so it is shown by default and can
    be silenced on its own::

        warnings.filterwarnings("ignore", category=pymisha.PymishaWarning)
    """


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for *name*, a child of the ``pymisha`` logger.

    Call it as ``_logger = get_logger(__name__)`` at module level.
    """
    return logging.getLogger(name)


def user_stacklevel(default: int = 2) -> int:
    """``stacklevel`` for :func:`warnings.warn` that blames the caller's code.

    Walks out of the pymisha package and returns the depth of the first frame
    that belongs to someone else, so the warning points at the user's call
    rather than at whichever internal helper happened to notice the problem.
    Falls back to *default* when every frame is pymisha's own.
    """
    frame: FrameType | None = sys._getframe(1)  # the caller of user_stacklevel
    level = 1
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_PACKAGE_DIR + os.sep):
            return level
        frame = frame.f_back
        level += 1
    return default
