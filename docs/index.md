<div style="display: flex; align-items: center; gap: 2em; margin-bottom: 1em;" markdown>
<div markdown>

# PyMisha

Python interface for [misha](https://github.com/tanaylab/misha) genomic databases with C++ streaming backends.

</div>
<div>
<img src="assets/logo.png" alt="PyMisha" width="300">
</div>
</div>

---

## Features

- **1D and 2D track support** — Dense, sparse, and 2D (rectangle/point) tracks with full CRUD
- **C++ streaming backends** — Extraction, summary, quantiles, distribution, lookup, and more
- **Virtual tracks** — Computed-on-the-fly views with 30+ aggregation functions
- **Interval operations** — Union, intersection, difference, neighbors, liftover, and more
- **Sequence analysis** — K-mer counting, PWM/PSSM scoring, Markov-chain synthesis
- **R misha compatibility** — Reads and writes the same on-disk formats (123/145 R exports covered)

## Quick Start

Get started instantly with the bundled examples database:

```python
import pymisha as pm

pm.gdb_init_examples()
# or equivalently: pm.gsetroot(pm.gdb_examples_path())

print(pm.gtrack_ls())
print(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000)))
```

To connect to your own misha database:

```python
import pymisha as pm

pm.gsetroot("/path/to/misha_db")
intervals = pm.gintervals_from_strings(["chr1:0-1000", "chr1:2000-2600"])
out = pm.gextract("track1", intervals, iterator=100)
```

## Installation

```bash
pip install pymisha
```

Pre-built wheels available for Linux (x86_64) and macOS (x86_64, arm64), Python 3.10--3.12.
