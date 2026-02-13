"""
Run a few PyMisha examples against the bundled example database.

Usage:
    python tools/examples.py
"""

import pymisha as pm


def main():
    pm.gdb_init_examples()

    print("Tracks:", pm.gtrack_ls())

    intervals = pm.gintervals_from_strings(["1:0-100", "1:200-300"])
    windows = pm.gintervals_window("1", [500, 1000], half_width=50)

    print("Intervals:")
    print(intervals)
    print("Windows:")
    print(windows)

    print("gextract dense_track:")
    print(pm.gextract("dense_track", intervals))

    print("gscreen dense_track > 0:")
    print(pm.gscreen("dense_track > 0", intervals))


if __name__ == "__main__":
    main()
