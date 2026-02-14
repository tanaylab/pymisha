# Database Formats and Multi-Contig Support

```python
import pymisha as pm
```

## Overview

PyMisha databases can be stored in two formats:

- **Indexed format** (default, recommended): Single unified files for sequences and tracks
- **Per-chromosome format** (legacy): Separate files for each chromosome

The indexed format provides better performance and scalability, especially for genomes with many contigs (>50 chromosomes).

### Key Features

- **Automatic format detection** -- pymisha automatically detects which format your database uses
- **Fully backward compatible** -- existing databases continue to work without modification
- **Transparent to users** -- same API for both formats
- **Migration tools** -- convert databases when convenient
- **Performance benefits** -- 4-14% faster for large-scale analyses

## Database Formats

### Indexed Format (Recommended)

The indexed format uses unified files:

**Sequence data:**

- `seq/genome.seq` -- All chromosome sequences concatenated
- `seq/genome.idx` -- Index mapping chromosome names to positions

**Track data:**

- `tracks/mytrack.track/track.dat` -- All chromosome data concatenated
- `tracks/mytrack.track/track.idx` -- Index with offset/length per chromosome

**Advantages:**

- Fewer file descriptors (important for genomes with 100+ contigs)
- Better performance for large workloads (14% faster)
- Smaller disk footprint
- Faster track creation and conversion

### Per-Chromosome Format (Legacy)

The per-chromosome format uses separate files:

**Sequence data:**

- `seq/chr1.seq`, `seq/chr2.seq`, ... -- One file per chromosome

**Track data:**

- `tracks/mytrack.track/chr1.track`, `chr2.track`, ... -- One file per chromosome

**When to use:**

- Compatibility with older misha versions (<5.3.0)
- Small genomes (<25 chromosomes) where the performance difference is negligible

## Creating Databases

### New Databases (Indexed Format)

By default, new databases use the indexed format:

```python
# Create database from FASTA file
pm.gdb_create("/path/to/mydb", "/path/to/genome.fa")

# Or download a pre-built genome
pm.gdb_create_genome("hg38", path="/path/to/install")
```

### Force Legacy Format

To create a database in legacy (per-chromosome) format:

```python
pm.gdb_create("/path/to/mydb", "/path/to/genome.fa", db_format="per-chromosome")
```

!!! note
    Unlike R misha, which uses a global option (`options(gmulticontig.indexed_format = FALSE)`),
    pymisha uses the `db_format` parameter directly in `gdb_create()`.

## Checking Database Format

Use `gdb_info()` to inspect your database:

```python
pm.gdb_init("/path/to/mydb")
info = pm.gdb_info()
print(info["format"])  # "indexed" or "per-chromosome"
```

Example output:

```python
info = pm.gdb_info()
# info["path"]             -> "/path/to/mydb"
# info["is_db"]            -> True
# info["format"]           -> "indexed"
# info["num_chromosomes"]  -> 24
# info["genome_size"]      -> 3095693983
# info["chromosomes"]      -> DataFrame with chrom and size columns
```

!!! tip
    `gdb_info()` can also inspect a database without initializing it, by passing
    a path directly:

    ```python
    info = pm.gdb_info(groot="/path/to/some_db")
    ```

## Converting Databases

### Convert Entire Database

Convert all sequences (and optionally tracks and intervals) to indexed format:

```python
pm.gdb_init("/path/to/mydb")
pm.gdb_convert_to_indexed()
```

This will:

1. Convert sequence files (`chr*.seq` -> `genome.seq` + `genome.idx`)
2. Validate conversions
3. Optionally remove old files after successful conversion

To also convert tracks and intervals in one call:

```python
pm.gdb_convert_to_indexed(
    convert_tracks=True,
    convert_intervals=True,
    remove_old_files=True,
    verbose=True,
)
```

### Convert Individual Tracks

Convert specific tracks while keeping others in legacy format:

```python
pm.gtrack_convert_to_indexed("mytrack")

# Optionally remove the old per-chromosome files:
pm.gtrack_convert_to_indexed("mytrack", remove_old=True)
```

!!! warning
    2D tracks cannot be converted to indexed format yet.

### Convert Intervals

Convert interval sets to indexed format:

```python
# 1D intervals
pm.gintervals_convert_to_indexed("myintervals")

# 2D intervals
pm.gintervals_2d_convert_to_indexed("my2dintervals")
```

## Migration Guide

### When to Migrate

**High priority (significant benefits):**

- Genomes with many contigs (>50 chromosomes)
- Large-scale analyses (10M+ bp regions frequently)
- 2D track workflows
- File descriptor limit issues

**Medium priority (moderate benefits):**

- Repeated extraction workflows
- Regular analyses on medium-sized regions (1-10M bp)

**Low priority (minimal benefits):**

- Small genomes (<25 chromosomes)
- One-off analyses
- Simple queries on small regions

### Migration Workflow

**Step 1: Backup (optional but recommended)**

```python
import shutil
shutil.copytree("/path/to/mydb", "/path/to/mydb.backup")
```

**Step 2: Check current format**

```python
pm.gdb_init("/path/to/mydb")
info = pm.gdb_info()
print(f"Current format: {info['format']}")
```

**Step 3: Convert**

```python
pm.gdb_convert_to_indexed(
    convert_tracks=True,
    convert_intervals=True,
    remove_old_files=True,
    verbose=True,
)
```

**Step 4: Verify**

```python
# Check format changed
info = pm.gdb_info()
print(f"New format: {info['format']}")

# Test a few operations
result = pm.gextract("mytrack", pm.gintervals(1, 0, 1000))
print(result.head())
```

**Step 5: Remove backup (after validation)**

```python
import shutil
shutil.rmtree("/path/to/mydb.backup")
```

## Copying Tracks Between Databases

You can freely copy tracks between databases with different formats.

### Method 1: Export and Import

```python
# Export from source database
pm.gdb_init("/path/to/source_db")
data = pm.gextract("mytrack", pm.gintervals_all(), iterator="mytrack")
data.to_csv("/tmp/mytrack.tsv", sep="\t", index=False)

# Import to target database (format auto-detected)
pm.gdb_init("/path/to/target_db")
pm.gtrack_import("mytrack", "Copied track", "/tmp/mytrack.tsv", binsize=0)
# Automatically stored in the target database's format
```

### Method 2: Batch Copy

```python
tracks = ["track1", "track2", "track3"]

for track in tracks:
    # Export
    pm.gdb_init("/path/to/source_db")
    file_path = f"/tmp/{track}.tsv"
    data = pm.gextract(track, pm.gintervals_all(), iterator=track)
    data.to_csv(file_path, sep="\t", index=False)

    # Import
    pm.gdb_init("/path/to/target_db")
    info = pm.gtrack_info(track)
    pm.gtrack_import(track, info.get("description", ""), file_path, binsize=0)

    import os
    os.remove(file_path)
```

## Performance Comparison

Based on comprehensive benchmarks comparing indexed vs. legacy formats:

### Operations Faster with Indexed Format

| Workload | Improvement |
|---|---|
| Very large workloads (10M+ bp) | ~14% faster |
| 2D track operations | ~2% faster |
| Repeated extractions | ~14% faster |
| Real workflows | ~8% faster on average |

### Operations with Similar Performance

| Workload | Difference |
|---|---|
| Single chromosome extraction | Within 5% |
| Multi-chromosome (10-22 chr) | Within 1% |
| PWM operations | Within 3% |
| Small workloads (<1M bp) | Within 10% |

### Summary

- **64% of operations are faster** with indexed format
- **93% within 5%** of legacy format (statistically equal)
- **Average: 4% faster** across all benchmarks
- **No regressions** for common use cases

## Backward Compatibility

### Fully Compatible

- Existing databases work without modification
- Existing scripts work without changes
- Same API for both formats
- Automatic format detection
- Can mix formats in the same analysis

### Example: Mixed Environment

```python
# Work with both formats in the same session
pm.gdb_init("/path/to/legacy_db")
data1 = pm.gextract("track1", pm.gintervals(1, 0, 1000))

pm.gdb_init("/path/to/indexed_db")
data2 = pm.gextract("track2", pm.gintervals(1, 0, 1000))
```

## Troubleshooting

### "File descriptor limit reached"

This occurs with many-contig genomes in legacy format.

**Solution:** Convert to indexed format.

```python
pm.gdb_convert_to_indexed(convert_tracks=True)
```

### "Track not found after copying files"

After manually copying track directories, the in-memory cache may be stale.

**Solution:** Reload the database.

```python
pm.gdb_reload()
```

### "Conversion fails with disk space error"

Conversion needs approximately 2x the track size temporarily.

**Solution:** Free disk space or convert tracks individually.

```python
# Convert one track at a time
pm.gtrack_convert_to_indexed("track1", remove_old=True)
pm.gtrack_convert_to_indexed("track2", remove_old=True)
```

## Best Practices

### For New Projects

1. Use indexed format (the default) for new databases
2. Use `gdb_create_genome()` for standard genomes (hg19, hg38, mm9, mm10, mm39)
3. Use `gdb_create()` with FASTA files for custom genomes

### For Existing Projects

1. Check format with `gdb_info()`
2. Migrate if beneficial (many contigs, large analyses)
3. No rush -- legacy format remains fully supported

## Summary

- **Indexed format is the default and recommended** for new databases
- **Legacy format remains fully supported** -- no forced migration
- **Automatic format detection** -- users do not need to think about it
- **Significant performance benefits** for large-scale analyses
- **Easy migration** with `gdb_convert_to_indexed()`
- **Fully backward compatible** -- existing code works unchanged
