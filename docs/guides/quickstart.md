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

## Example DB

```python
pm.gdb_init_examples()
print(pm.gtrack_ls())
print(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000)))
```
