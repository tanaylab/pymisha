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
