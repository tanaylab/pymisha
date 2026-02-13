import os

import pymisha as pm


def test_progress_callback_fds_are_not_closed():
    opened = {"fds": None}

    def progress_cb(done, total, pct):
        if opened["fds"] is None and done > 0:
            opened["fds"] = os.pipe()

    pm.gextract("dense_track", pm.gintervals_all(), iterator=10000, progress=progress_cb)
    assert opened["fds"] is not None

    rfd, wfd = opened["fds"]
    try:
        os.fstat(rfd)
        os.fstat(wfd)
    finally:
        os.close(rfd)
        os.close(wfd)
