<table><tr>
<td>

# PyMisha

[![PyPI](https://img.shields.io/pypi/v/pymisha.svg)](https://pypi.org/project/pymisha/)
[![CI](https://github.com/tanaylab/pymisha/actions/workflows/ci.yml/badge.svg)](https://github.com/tanaylab/pymisha/actions/workflows/ci.yml)

Python interface for [misha](https://github.com/tanaylab/misha) genomic databases. PyMisha provides full read/write access to misha track databases with C++ streaming backends for genome-scale operations.

</td>
<td>
<img src="docs/assets/logo.png" alt="PyMisha" width="300">
</td>
</tr></table>

## Features

- **1D and 2D track support:** Dense, sparse, and 2D (rectangle/point) tracks with full CRUD operations.
- **C++ streaming backends:** Extraction, summary, quantiles, distribution, lookup, segmentation, Wilcoxon tests, correlation, and sampling all stream through C++ for performance.
- **Virtual tracks:** Computed-on-the-fly track views with filtering, shifting, and 30+ aggregation functions.
- **Interval operations:** Union, intersection, difference, canonicalization, neighbors, annotation, normalization, random generation, and liftover.
- **Sequence analysis:** Extraction, k-mer counting, PWM/PSSM scoring, and Markov-chain synthesis (`gsynth`).
- **Database management:** Create, link, convert, and manage misha-compatible genomic databases.
- **R misha compatibility:** Reads and writes the same on-disk formats as R misha (123/145 R exports covered).

## Installation

```bash
pip install pymisha
```

Pre-built wheels are available for Linux (x86_64) and macOS (x86_64 and arm64), Python 3.10-3.12.

To install from source (requires a C++17 compiler and numpy):

```bash
pip install -e ".[dev]"
```

## Quick start

PyMisha ships with a built-in examples database so you can start exploring immediately -- no external data needed:

```python
import pymisha as pm

# Option 1: one-liner to load the bundled examples database
pm.gdb_init_examples()

# Option 2: equivalent explicit form
pm.gsetroot(pm.gdb_examples_path())

# List available tracks and extract data
print(pm.gtrack_ls())
print(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000)))
```

To connect to your own misha database, use `gsetroot`:

```python
import pymisha as pm

# Initialize the database
pm.gsetroot("/path/to/misha_db")

# Create intervals and extract data
intervals = pm.gintervals_from_strings(["chr1:0-1000", "chr1:2000-2600"])
out = pm.gextract("track1", intervals, iterator=100)

# Filter and summarize
filtered = pm.gscreen("track1 > 0.5", intervals)
stats = pm.gsummary("track1", intervals)
```

## Examples

Using the built-in example database:

```python
import pymisha as pm

# Quickest way to get started
pm.gdb_init_examples()

# Or equivalently, using gsetroot with the examples path
pm.gsetroot(pm.gdb_examples_path())

print(pm.gtrack_ls())
print(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000)))
```

## Creating a genome database

PyMisha ships prebuilt genome databases for common assemblies. Download and set up with a single call:

```python
import pymisha as pm

# Download a prebuilt genome (mm9, mm10, mm39, hg19, hg38)
pm.gdb_create_genome("hg38", path="/data/genomes")   # creates /data/genomes/hg38/
pm.gsetroot("/data/genomes/hg38")

pm.gchrom_sizes()  # verify it worked
```

To build a database from your own FASTA files (e.g. a custom assembly):

```python
pm.gdb_create("/data/my_genome", "genome.fa.gz", verbose=True)
pm.gsetroot("/data/my_genome")
```

See the [Creating Genome Databases](https://tanaylab.github.io/pymisha/tutorials/genomes/) tutorial for UCSC download workflows and advanced options.

## Optional dependencies

- `pyBigWig`: For BigWig import in `gtrack_import`.
- `pyreadr` + `Rscript`: For loading R-serialized big interval sets.
- `PyYAML`: For richer `gdataset_info` metadata parsing.

## Missing features

Compared to R misha, the following are not yet implemented:

- **Track Arrays:** `gtrack.array.*` and `gvtrack.array.slice`.
- **Legacy Conversion:** `gtrack.convert` (for migrating old 2D formats).

## License

MIT. See [LICENSE](LICENSE) for details.
