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

## Thread safety

PyMisha inherits R misha's single-threaded design. Keep the following constraints in mind:

- **Not thread-safe.** All module-level state (`_GROOT`, `_UROOT`, `_VTRACKS`, `CONFIG`) is process-global and unsynchronized. Do not call PyMisha from multiple threads concurrently.
- **One database per process.** You cannot have two databases open simultaneously; `gsetroot()` replaces the active database globally.
- **`CONFIG` is global.** Changing settings like `max_processes` affects every subsequent operation in the process.
- **Multiprocessing uses `fork()`.** The C++ backend parallelizes via `fork()` with shared memory (mmap) and semaphores. This is transparent to the caller but means PyMisha should not be used inside already-forked worker processes or with `fork`-unsafe libraries.

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

## Using pymisha with an LLM agent

LLM coding agents (Claude Code, Copilot, Cursor) writing pymisha analysis code can pre-load these reference docs into context for fewer hallucinated APIs and more idiomatic recipes:

- [agent-guides/pymisha-core.md](agent-guides/pymisha-core.md) — concepts, chooser tables, everyday recipes.
- [agent-guides/pymisha-advanced.md](agent-guides/pymisha-advanced.md) — 2D/Hi-C, PWM, import/export, new genomes, gsynth.
- [agent-guides/pymisha-anti-patterns.md](agent-guides/pymisha-anti-patterns.md) — silent footguns cross-referenced from the above.
- [agent-guides/skills/importing-tracks/SKILL.md](agent-guides/skills/importing-tracks/SKILL.md) — full track-import reference.

**Drop-in prompt (no clone needed).** Paste the block below into your agent at the start of a pymisha task. It points the agent at the raw files on GitHub, so it works without a local checkout:

```
Before writing any pymisha code, fetch and read:

- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-core.md  (mandatory: concepts + everyday recipes)
- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-anti-patterns.md  (silent footguns; cross-referenced from core)
- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-advanced.md  (consult on demand: 2D/Hi-C, PWM, import/export, new genomes)

Follow the conventions in those files. When you hit a recipe with an
"Avoid:" block, treat it as a hard rule.
```

Pin to a release tag for stability by replacing `main` with any tag that contains `agent-guides/`. The `skills/importing-tracks/SKILL.md` guide listed above is load-on-demand; pull it in only when the task specifically calls for track import.

The guides mirror the equivalent set in [R misha](https://github.com/tanaylab/misha/tree/master/agent-guides) — same section numbering, same recipes, translated to the pymisha API.

## Missing features

Compared to R misha, the following are not yet implemented:

- **Track Arrays:** `gtrack.array.*` and `gvtrack.array.slice`.
- **Legacy Conversion:** `gtrack.convert` (for migrating old 2D formats).

## License

MIT. See [LICENSE](LICENSE) for details.
