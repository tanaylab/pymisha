# Quickstart

## Initialize DB

```python
import pymisha as pm

pm.gdb_init("/path/to/misha_db")
```

## Extract values

```python
intervals = pm.gintervals_from_strings(["chr1:0-1000", "chr1:2000-2600"])
out = pm.gextract("track1", intervals, iterator=100)
```

## Screen and summarize

```python
filtered = pm.gscreen("track1 > 0.5", intervals)
stats = pm.gsummary("track1", intervals)
```

## Thread safety

PyMisha is **not thread-safe**. All state (active database, virtual tracks, config) is process-global, so you should not call PyMisha from multiple threads or open more than one database per process. See the [README](https://github.com/tanaylab/pymisha#thread-safety) for full details.

## Example DB

```python
pm.gdb_init_examples()
print(pm.gtrack_ls())
print(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000)))
```

## Seeing what PyMisha recovered from

PyMisha falls back rather than failing in a number of places - an optional
dependency that is missing, a probe that decides a track is not of some type, a
best-effort cleanup. Each fallback reports what it caught on the `pymisha`
logger, which is silent until you configure logging:

```python
import logging
logging.basicConfig()
logging.getLogger("pymisha").setLevel(logging.DEBUG)
```

Every module logs to its own child (`pymisha.tracks`, `pymisha.intervals`, ...),
so a single one can be turned up on its own. Failures that R misha itself warns
about are raised as `pymisha.PymishaWarning` instead, so they are visible with
no configuration; silence them with
`warnings.filterwarnings("ignore", category=pm.PymishaWarning)`.
