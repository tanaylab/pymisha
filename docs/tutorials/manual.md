# PyMisha User Manual

**PyMisha** is designed to help users efficiently analyze genomic data from various experiments. The data must be stored in a *Genomic Database* in a specific format described in this document. This manual also covers fundamental concepts such as *track expressions*, *iterators*, and more.

## Genomic Database

A Genomic Database starts with a *root* (also frequently referred to as *GROOT*), i.e., a top directory containing certain subdirectories and files. A new database can be created using `pm.gdb_create_genome()` and `pm.gdb_create()` functions. This is the easiest way to do it -- see the "Genomes" guide for more details. One can also build a database manually by generating all the necessary components described below.

Before the data in a Genomic Database can be accessed, you must establish a connection with it by calling `pm.gdb_init()`:

```python
import pymisha as pm

pm.gdb_init("/path/to/genomic_db")
```

To use the built-in example database:

```python
pm.gdb_init_examples()
```

A valid Genomic Database should contain the following files and subdirectories:

- `chrom_sizes.txt` -- a file containing the list of chromosomes and their sizes.
- `tracks/` -- a directory that serves as a repository for all *tracks* and *interval sets*. May contain subdirectories.
- `pssms/` -- a directory containing PSSM sets (PSSM data and PSSM key files).
- `seq/` -- a directory containing full genomic sequences.

!!! note
    The `pssms/` and `seq/` directories are optional and are required only by a subset of functions in the package.

### Dataset API

PyMisha supports working with multiple data sources through the Dataset API. This is useful when you want to combine a shared read-only reference database with a user-specific database for custom tracks, or when working with multiple data sources that share the same genome.

#### Working Database vs Datasets

The Dataset API distinguishes between:

1. **Working Database**: Set via `pm.gsetroot()`, this is your primary, writable database where new tracks and intervals are created.
2. **Loaded Datasets**: Added via `pm.gdataset_load()`, these are read-only data sources that provide additional tracks and intervals.

```python
# Set your working database
pm.gsetroot("/data/my_project")

# Load additional datasets
pm.gdataset_load("/shared/ucsc_annotations")
pm.gdataset_load("/shared/encode_chipseq")

# List all sources (working db + loaded datasets)
pm.gdataset_ls()

# Get detailed information
pm.gdataset_info("/shared/ucsc_annotations")
```

**Key points:**

- All sources must have identical `chrom_sizes.txt` files (same genome assembly).
- The working database is used for `seq/` and `chrom_sizes.txt` access.
- Tracks and intervals are aggregated from all sources.
- New tracks are created in the working database only.

#### Track Resolution and Collision Handling

By default, loading a dataset with tracks that already exist will raise an error:

```python
# If working_db has "my_track" and dataset1 also has "my_track":
pm.gdataset_load("dataset1")
# Error: Cannot load dataset 'dataset1': tracks 'my_track' already exist in working database.
# Use force=True to override (working db wins).
```

Use `force=True` to allow collisions, with clear precedence rules:

```python
# Working database always wins over datasets
pm.gdataset_load("dataset1", force=True)

# For dataset-to-dataset collisions, later-loaded wins
pm.gdataset_load("dataset2", force=True)

# Check which source provides a track
pm.gtrack_dataset("my_track")
```

#### Cross-Database Track Expressions

Track expressions seamlessly combine tracks from different sources:

```python
pm.gsetroot("/data/my_project")
pm.gdataset_load("/shared/annotations")

# Extract tracks from different sources in a single call
result = pm.gextract(
    ["my_track", "annotation_track"],
    pm.gintervals(1, 0, 10000),
    iterator=100,
)

# Use track expressions across sources
normalized = pm.gextract(
    "my_track - annotation_track",
    pm.gintervals(1, 0, 10000),
    iterator=100,
)

# Access track attributes from any source
pm.gtrack_attr_get("my_track", "description")
pm.gtrack_attr_get("annotation_track", "description")
```

Virtual tracks can reference source tracks from any loaded source:

```python
# Create virtual tracks from different sources
pm.gvtrack_create("vt_signal", "my_track", "avg")
pm.gvtrack_create("vt_annotation", "annotation_track", "avg")

# Combine them in expressions
pm.gextract("vt_signal / vt_annotation", pm.gintervals(1, 0, 10000))
```

#### Querying Track and Interval Sources

Use `pm.gtrack_dataset()` and `pm.gintervals_dataset()` to find which source contains a track or interval set:

```python
# Single track -- returns source path
pm.gtrack_dataset("my_track")

# Intervals
pm.gintervals_dataset("my_intervals")
```

Filter track and interval listings by source:

```python
# List tracks matching a pattern
pm.gtrack_ls("chip_*")

# List intervals
pm.gintervals_ls()
```

#### Creating and Sharing Datasets

Use `pm.gdataset_save()` to create a dataset from selected tracks and intervals:

```python
pm.gsetroot("/data/my_project")

# Create a dataset with selected tracks and intervals
pm.gdataset_save(
    path="/shared/my_chipseq_dataset",
    description="ChIP-seq tracks for H3K4me3 and H3K27ac",
    tracks=pm.gtrack_ls("chip_*"),  # Pattern matching
    intervals=["peaks_h3k4me3", "peaks_h3k27ac"],
)
```

Get information about a dataset:

```python
pm.gdataset_info("/shared/my_chipseq_dataset")
# Returns: description, author, created date, track/interval counts, genome hash
```

#### Creating a Linked Database

Use `pm.gdb_create_linked()` to create a lightweight database that links to a parent database's `seq/` directory and `chrom_sizes.txt` file:

```python
# Create linked database with symlinks to parent's seq and chrom_sizes
pm.gdb_create_linked("~/my_db", parent="/shared/hg38")

# Use as your working database
pm.gsetroot("~/my_db")

# Load datasets from the parent
pm.gdataset_load("/shared/hg38")

# Create your own tracks
pm.gtrack_create("my_analysis", "Analysis results", "reference_track * 2")
```

This avoids duplicating large sequence files while allowing you to maintain your own tracks.

#### Unloading Datasets

Remove loaded datasets from the namespace:

```python
# Unload a dataset (tracks/intervals become unavailable)
pm.gdataset_unload("/shared/annotations")

# Safe to call even if not loaded (no error by default)
pm.gdataset_unload("/nonexistent/path")

# Error if validate=True and not loaded
pm.gdataset_unload("/nonexistent/path", validate=True)
# Error: Dataset '/nonexistent/path' is not loaded
```

When unloading datasets that had track collisions (loaded with `force=True`), shadowed tracks are restored with proper precedence: working database first, then datasets in load order.

#### Moving and Copying Tracks

Use `pm.gtrack_mv()` to rename a track or move it within the same database:

```python
# Rename a track
pm.gtrack_mv("old_name", "new_name")

# Move to a different namespace (directory)
pm.gtrack_mv("analysis.track1", "results.track1")
```

Use `pm.gtrack_copy()` to create a copy of a track:

```python
# Copy a track within the same database
pm.gtrack_copy("source_track", "copy_track")

# Copy from a loaded dataset to the working database
pm.gsetroot("/data/my_project")
pm.gdataset_load("/shared/annotations")
pm.gtrack_copy("annotation_track", "my_local_copy")  # Copy to working db
```

!!! note
    `pm.gtrack_mv()` only works within the same database. To move a track between databases, use `pm.gtrack_copy()` followed by `pm.gtrack_rm()`.

#### Backward Compatibility

Single-database usage works exactly as before:

```python
pm.gsetroot("single_database")   # Works unchanged
pm.gdb_init("single_database")   # Equivalent, also works
```

### Database Directory Structure

An example of a Genomic Database file structure:

```
hg38/                          <- Genomic Database root directory
   chrom_sizes.txt
   .ro_attributes               <- List of read-only attributes
   pssms/                       <- (optional)
      motif1.data                  <- pssm data file
      motif1.key                   <- pssm key file
      mypssm.data
      mypssm.key
   seq/                         <- (optional)
      genome.seq                   <- indexed format: single sequence file
      genome.idx                   <- indexed format: index file
      OR
      chr1.seq                     <- per-chromosome format: separate files
      chr2.seq
      chr3.seq
   tracks/
      tss.interv                   <- small intervals set = tss
      big_data.interv/             <- big intervals set (per-chromosome format)
         .meta                        <- summary of the intervals set
         chr1                         <- chrom files
         chr5
      indexed_intervals.interv/    <- big intervals set (indexed format)
         intervals.dat                <- consolidated interval data
         intervals.idx                <- index file
      rpt.track/                   <- track = rpt (per-chromosome format)
         .attributes                  <- track attributes (optional)
         chr1                         <- chrom files
         chr2
         chr3
         vars/                        <- track variables (optional)
             myresult                    <- track variable
      indexed_track.track/         <- track (indexed format)
         track.dat                    <- consolidated track data
         track.idx                    <- index file
         .attributes                  <- track attributes (optional)
         vars/                        <- track variables (optional)
      test/
         intervals1.interv            <- intervals = test.intervals1
         track1.track/                <- track = test.track1
            .attributes
            chr1
            chr2
            chr3
      savta/
         fourC.track/                 <- track = savta.fourC
            chr1
            chr2
            chr3
```

## File Formats

### `chrom_sizes.txt`

The `chrom_sizes.txt` file must be located under the root directory of the Genomic Database. It lists the chromosomes and their sizes. The chromosome name appears in the first column, the size in the second. Chromosome names should appear **without** the "chr" prefix. The two columns are separated by a tab character.

Example:

```
1    247249719
2    242951149
3    199501827
X    154913754
Y    57772954
```

### Seq Files

Genomic sequences are stored in the `seq/` directory. Two formats are supported.

#### Indexed Format (Recommended)

The indexed format uses two files:

- `genome.seq` -- single binary file containing concatenated sequences for all contigs.
- `genome.idx` -- binary index mapping contig names to positions in `genome.seq`.

This format provides better performance and scalability, especially for genomes with many contigs. It is the default format created by `pm.gdb_create()`.

**Format Specifications:**

The `genome.idx` file has the following structure:

- **Header (24 bytes)**:
    - Magic number: `"MISHAIDX"` (8 bytes)
    - Index version: `uint32_t` (4 bytes, currently 1)
    - Number of contigs: `uint32_t` (4 bytes)
    - CRC64-ECMA checksum: `uint64_t` (8 bytes)

- **Entries** (one per contig):
    - Chromosome ID: `uint32_t` (4 bytes) -- alphabetical rank (0, 1, 2, ...)
    - Name length: `uint16_t` (2 bytes)
    - Name: UTF-8 string (not null-terminated)
    - Offset in genome.seq: `uint64_t` (8 bytes)
    - Sequence length: `uint64_t` (8 bytes)
    - Reserved: `uint64_t` (8 bytes, currently zeros)

All multi-byte integers are stored in little-endian byte order.

#### Per-Chromosome Format

Each contig has a separate `.seq` file containing its genomic sequence as a contiguous string of ASCII characters. Files are named `chrXXX.seq` where `XXX` is the contig name from `chrom_sizes.txt`.

Example of a short (25 base pairs) seq file:

```
ggtgaAGccctggagattcttatta
```

### Track Files

Tracks can be stored in two formats: **per-chromosome format** or **indexed format**.

#### Per-Chromosome Format

In the traditional per-chromosome format, each chromosome's track data is stored in a separate file within a track directory (e.g., `tracks/mytrack.track/chr1`, `tracks/mytrack.track/chr2`, etc.). This format works well for genomes with a small number of chromosomes.

#### Indexed Format (Recommended for Multi-Contig Genomes)

The indexed format consolidates all chromosome data into two files:

- `track.dat` -- single binary file containing concatenated track data for all chromosomes.
- `track.idx` -- binary index mapping chromosome IDs to positions in `track.dat`.

This format dramatically reduces file descriptor usage for genomes with many contigs (e.g., draft assemblies with thousands of scaffolds) and improves performance for parallel access.

**Format Specifications:**

The `track.idx` file has the following structure:

- **Header (36 bytes)**:
    - Magic number: `"MISHATDX"` (8 bytes)
    - Index version: `uint32_t` (4 bytes, currently 1)
    - Track type: `uint32_t` (4 bytes) -- 0=dense, 1=sparse, 2=array
    - Number of contigs: `uint32_t` (4 bytes)
    - Flags: `uint64_t` (8 bytes) -- bit 0 indicates little-endian
    - CRC64-ECMA checksum: `uint64_t` (8 bytes)

- **Entries** (24 bytes per contig):
    - Chromosome ID: `uint32_t` (4 bytes)
    - Offset in track.dat: `uint64_t` (8 bytes)
    - Data length: `uint64_t` (8 bytes)
    - Reserved: `uint32_t` (4 bytes, for future use)

All multi-byte integers are stored in little-endian byte order.

**Converting to Indexed Format:**

Existing per-chromosome tracks can be converted to indexed format using `pm.gtrack_convert_to_indexed()`:

```python
# Convert a track to indexed format
pm.gtrack_convert_to_indexed("my_track")

# Check track format
info = pm.gtrack_info("my_track")
print(info["format"])  # "indexed" or "per-chromosome"
```

**Supported Track Types:** Dense (fixed-bin) and Sparse tracks (1D only) can use indexed format. 2D tracks and virtual tracks are not supported for indexed format.

**Backward Compatibility:** The indexed format is fully transparent to all PyMisha functions. Both formats can coexist in the same database and are used identically in track expressions and analysis functions.

**Validation Limits:** To prevent issues with corrupted or malicious index files, the following limits are enforced:

- **Maximum contigs per index**: 20,000,000 (applies to genome.idx, track.idx, intervals.idx)
- **Maximum contig name length**: 1,024 bytes

### PSSM Set

Each *PSSM Set* consists of two files: *PSSM key* and *PSSM data*. The files should be named `XXX.key` and `XXX.data` accordingly, where `XXX` is the name of the PSSM set. Both files must be placed into the `pssms/` directory.

#### PSSM Key

The *PSSM Key* file describes PSSMs in the following tab-separated format:

| Column | Type | Description |
|--------|------|-------------|
| ID | Integer | Unique ID (referenced in PSSM Data file) |
| Sequence | String | PSSM sequence |
| Bidirectional | `0` or `1` | If `1`, energy is calculated on the complementary strand as well |

Example:

```
0    *************ATTAAT**************    1
1    *********A*ACACACACA*****A*******    1
2    *************AAAATGGC*G**********    1
3    *************ACTGCTTG************    1
```

#### PSSM Data

The *PSSM Data* file contains probability matrices for each PSSM key in the following tab-separated format:

| Column | Type | Description |
|--------|------|-------------|
| ID | Integer | Unique ID (must appear in PSSM Key file) |
| Position | Integer | Zero-based position in the range `[0, length(PSSM sequence)-1]` |
| Probability of 'A' | Numeric | Probability of 'A' in the range `[0, 1]` |
| Probability of 'C' | Numeric | Probability of 'C' in the range `[0, 1]` |
| Probability of 'G' | Numeric | Probability of 'G' in the range `[0, 1]` |
| Probability of 'T' | Numeric | Probability of 'T' in the range `[0, 1]` |

## Intervals

### 1D Intervals

A **1D interval** (one-dimensional interval) represents a genomic section. It is defined by `(chrom, start, end)` where `start` and `end` are genomic coordinates (`start < end`). The coordinates are zero-based, meaning the chromosome starts at coordinate 0. The end coordinate marks the last coordinate in the section plus 1. To represent a point in the genome at coordinate `X`, create an interval with `start=X` and `end=X+1`.

```python
# A single 1D interval: chr1:100-200
interval = pm.gintervals(1, 100, 200)

# Multiple intervals
intervals = pm.gintervals([1, 1, 2], [0, 500, 100], [300, 800, 400])
```

### 2D Intervals

A **2D interval** (two-dimensional interval) represents a rectangle in genomic space. It is defined by `(chrom1, start1, end1, chrom2, start2, end2)`, where `start1, start2, end1, end2` are start and end coordinates that mark the limits of a rectangle.

```python
# A single 2D interval
interval_2d = pm.gintervals_2d(1, 200, 800, 1, 100, 1000)
```

### Interval Sets

Multiple intervals can be combined into a table, known as an **interval set**. This table is represented as a pandas DataFrame. For 1D intervals, the DataFrame must have columns named `chrom`, `start`, and `end`. For 2D intervals, the columns must be `chrom1`, `start1`, `end1`, `chrom2`, `start2`, and `end2`.

Additional columns can be added to intervals, and some may be used by various functions. For instance, `pm.gintervals_neighbors()` uses the `strand` column if it is present in 1D intervals. Use `pm.gintervals()` and `pm.gintervals_2d()` to create 1D and 2D intervals, respectively.

Both 1D and 2D intervals are used throughout the package. Some functions manipulate them (union, intersect, etc.), others use them to limit a function's scope, and yet others perform calculations for each interval in the set.

### Whole-Genome Intervals

To get intervals covering the entire genome, use:

```python
# All 1D intervals (whole genome)
all_1d = pm.gintervals_all()

# All 2D intervals (all chromosome pairs)
all_2d = pm.gintervals_2d_all()
```

### Serializing Intervals, Big and Small Interval Sets

Interval sets can be saved in the Genomic Database. Use `pm.gintervals_save()` and `pm.gintervals_load()` to save or load an interval set, and `pm.gintervals_update()` to update/add/delete a certain chromosome from the set.

Internally, interval sets can be stored in two different formats: **small interval set** or **big interval set**. The specific format is chosen depending on the size of the interval set. Big format is selected for larger interval sets, while smaller sets are stored in the small format.

!!! note
    Saved interval sets in the small format can be seamlessly used in all functions and track expressions without the need to explicitly load them.

```python
# 'annotations' is an interval set saved in the Genomic Database
pm.gintervals_intersect("annotations", pm.gintervals(2))
```

Big interval sets can be used in many but not all functions. A notable exception is `pm.gintervals_load()`, which allows loading only a single chromosome (or a chromosome pair for 2D cases) of a big interval set.

#### Interval Set Storage Formats

Big interval sets can be stored in two formats: **per-chromosome format** or **indexed format**.

**Per-Chromosome Format:**
In this traditional format, intervals for each chromosome (or chromosome pair for 2D intervals) are stored in separate files within an interval set directory. For 1D intervals: `myintervals.interv/chr1`, `chr2`, etc. For 2D intervals: `myintervals.interv/chr1-chr2`, `chr1-chr3`, etc.

**Indexed Format (Recommended for Multi-Contig Genomes):**
The indexed format consolidates all chromosomes into two files, dramatically reducing file descriptor usage for genomes with many contigs.

**1D Intervals** use:

- `intervals.dat` -- binary file with concatenated interval data
- `intervals.idx` -- binary index file (36-byte header + 24 bytes per contig)

The `intervals.idx` file structure:

- **Header (36 bytes)**:
    - Magic: `"MISHAI1D"` (8 bytes)
    - Version: `uint32_t` (4 bytes)
    - Number of entries: `uint32_t` (4 bytes)
    - Flags: `uint64_t` (8 bytes) -- bit 0 indicates little-endian
    - CRC64-ECMA checksum: `uint64_t` (8 bytes)
    - Reserved: `uint32_t` (4 bytes)

- **Entries** (24 bytes per contig):
    - Chromosome ID: `uint32_t` (4 bytes)
    - Offset: `uint64_t` (8 bytes)
    - Length: `uint64_t` (8 bytes)
    - Reserved: `uint32_t` (4 bytes)

**2D Intervals** use:

- `intervals2d.dat` -- binary file with concatenated pair data
- `intervals2d.idx` -- binary index file (40-byte header + 28 bytes per pair)

The `intervals2d.idx` file structure:

- **Header (40 bytes)**:
    - Magic: `"MISHAI2D"` (8 bytes)
    - Version: `uint32_t` (4 bytes)
    - Number of entries: `uint32_t` (4 bytes)
    - Flags: `uint64_t` (8 bytes)
    - CRC64-ECMA checksum: `uint64_t` (8 bytes)
    - Reserved: `uint64_t` (8 bytes)

- **Entries** (28 bytes per pair):
    - Chromosome 1 ID: `uint32_t` (4 bytes)
    - Chromosome 2 ID: `uint32_t` (4 bytes)
    - Offset: `uint64_t` (8 bytes)
    - Length: `uint64_t` (8 bytes)
    - Reserved: `uint32_t` (4 bytes)

!!! note
    Only non-empty chromosome pairs are stored in the 2D index, avoiding O(N^2) space overhead.

**Converting to Indexed Format:**

```python
# Convert 1D interval set to indexed format
pm.gintervals_convert_to_indexed("my_intervals")

# Convert 2D interval set to indexed format
pm.gintervals_2d_convert_to_indexed("my_2d_intervals")

# Convert and remove old per-chromosome files
pm.gintervals_convert_to_indexed("my_intervals", remove_old=True)
```

All multi-byte integers are stored in little-endian byte order. The indexed format is fully backward compatible with all PyMisha functions.

## Tracks

A **Track** is a data structure that binds numeric data (floating-point values) to a genomic space (a set of genomic intervals). Track data is typically accessed through **track expressions**, which are widely used by various functions in the package.

Two fundamental types of tracks exist: **1D** and **2D**.

### 1D Track

A **1D track** (one-dimensional track) maps numeric values V_0, ..., V_n to non-overlapping 1D intervals. PyMisha supports two formats of 1D tracks: **Dense** (also called **Fixed Bin**) and **Sparse**.

For a **Dense** track, the size of the genomic interval is always fixed and called the **bin size**. Numeric values are stored for all genomic intervals that cover the genome, although some values can be `NaN`. A Dense track file appears as a continuous chunk of values V_0, ..., V_n, where V_i maps to an interval `[binsize * i, binsize * (i+1))`. Dense track files do not store interval coordinates, allowing them to represent large amounts of numeric data compactly. The size of a Dense track is inversely proportional to the bin size. Random access to a value at a given coordinate has constant complexity, O(1).

**Sparse** tracks offer more flexibility. Each numeric value can map to a genomic interval of any size. The size of a Sparse track is proportional to the number of numeric values (excluding NaN). The complexity of random access to a value at a given coordinate is O(log N), where N is the number of values in the track.

| Property | Dense | Sparse |
|---|---|---|
| Optimal use case | Data covering nearly the whole genome | Data covering a limited portion of the genome |
| Values stored | Per bin (interval of a fixed size) | Per interval of an arbitrary size |
| Random access complexity | O(1) | O(log N) |
| Disk usage | 4 bytes per bin | 20 bytes per value |

1D tracks can be created with functions such as `pm.gtrack_create()`, `pm.gtrack_create_sparse()`, `pm.gtrack_create_dense()`, `pm.gtrack_import()`, `pm.gtrack_import_set()`, and more.

### 2D Track

A **2D track** (two-dimensional track) maps numeric values V_0, ..., V_n to non-overlapping 2D intervals. These are often used to represent interactions between different parts of the genome.

Typically, 2D tracks use the **Rectangles** format. A more space-efficient **Points** format also exists, with similar behavior.

Creation functions:

- Rectangles tracks: `pm.gtrack_create()`, `pm.gtrack_2d_create()`
- Points tracks: `pm.gtrack_2d_import_contacts()`

### Track as an Intervals Set

Tracks represent sets of intervals augmented with values, and can therefore replace interval sets in functions like `pm.gextract()`, `pm.gintervals_neighbors()`, and `pm.gintervals_chrom_sizes()`.

!!! warning
    **Dense** tracks cannot be used in place of interval sets.

### Track Attributes

Beyond numeric data, tracks can store metadata such as descriptions or sources. This metadata is stored as name-value pairs (attributes), where the value is a string. Tracks created using `pm.gtrack_create()`, `pm.gtrack_smooth()`, etc., automatically have `created.by`, `created.date`, and `description` attributes.

While there is no strict rule, attributes typically store short strings. For other data formats, consider using **track variables**.

Attribute management:

- Retrieval / Modification: `pm.gtrack_attr_get()`, `pm.gtrack_attr_set()`
- Bulk Actions: `pm.gtrack_attr_export()`, `pm.gtrack_attr_import()`
- Search by Pattern: `pm.gtrack_ls()`

```python
# Get a track attribute
desc = pm.gtrack_attr_get("dense_track", "description")

# Set a track attribute
pm.gtrack_attr_set("dense_track", "source", "experiment_1")

# Export attributes for multiple tracks as a DataFrame
attrs_df = pm.gtrack_attr_export()

# Search tracks by attribute values
pm.gtrack_ls("dense*")
```

Some attributes are read-only, like `created.by` and `created.date`. Use `pm.gdb_get_readonly_attrs()` and `pm.gdb_set_readonly_attrs()` to manage the read-only list.

### Track Variables

Track variables store statistics, computation results, historical data, etc., related to a track. Unlike attributes, they can store data in any format (e.g., DataFrames, arrays, arbitrary Python objects).

Variable management:

- Retrieval / Modification / Removal: `pm.gtrack_var_get()`, `pm.gtrack_var_set()`, `pm.gtrack_var_rm()`
- List variables: `pm.gtrack_var_ls()`

```python
# Set a track variable (can be any picklable Python object)
pm.gtrack_var_set("dense_track", "my_stats", {"mean": 0.5, "std": 0.1})

# Get a track variable
stats = pm.gtrack_var_get("dense_track", "my_stats")

# List all variables for a track
pm.gtrack_var_ls("dense_track")

# Remove a variable
pm.gtrack_var_rm("dense_track", "my_stats")
```

### Track Attributes vs. Track Variables

Both track attributes and variables store track metadata, but they have distinct uses:

| Property | Track Attributes | Track Variables |
|---|---|---|
| Use Case | Track metadata as short strings (e.g., description) | Arbitrary track-associated data |
| Value Type | String | Any Python object |
| Single Value Retrieval | `pm.gtrack_attr_get()` | `pm.gtrack_var_get()` |
| Single Value Modification | `pm.gtrack_attr_set()` | `pm.gtrack_var_set()` |
| Bulk Retrieval | `pm.gtrack_attr_export()` | N/A |
| Bulk Modification | `pm.gtrack_attr_import()` | N/A |
| Object Names Retrieval | `pm.gtrack_attr_export()` | `pm.gtrack_var_ls()` |
| Object Removal | `pm.gtrack_attr_set()` (with an empty string) | `pm.gtrack_var_rm()` |
| Search by Value | `pm.gtrack_ls()` | N/A |

## Track Expressions

### Introduction

*Track expression* is a key concept in PyMisha. Track expressions are widely used in various functions (`pm.gscreen()`, `pm.gextract()`, `pm.gdist()`, etc.).

A track expression is a string that closely resembles a valid Python expression. Just like any other Python expression, it may include conditions, functions, and variables defined beforehand. `"1 > 2"`, `"np.mean([1, 2, 3])"`, and `"myvar < 17"` are all valid track expressions. Unlike regular Python expressions, a track expression might also contain track names or *virtual track* names.

How does a track expression get evaluated? A track expression is accompanied by an *iterator* that determines a set of intervals the expression iterates over. For each iterator interval, the track expression is evaluated. The value of a track expression `"np.mean([1, 2, 3])"` is constant regardless of the iterator interval. However, suppose the track expression contains a track name `mytrack`, like `"mytrack * 3"` -- then the story becomes very different. The library first recognizes that `mytrack` is not a regular Python variable but rather a track name. A new variable named `mytrack` is made available in the expression's evaluation context. For each iterator interval, this variable is assigned the corresponding value of the track. This value obviously depends on the iterator interval. Once `mytrack` is assigned the corresponding value, the track expression is evaluated.

So how exactly is the value of the `mytrack` variable determined given the iterator interval? We will demonstrate by example. Suppose the track `mytrack` is in sparse format. It consists of a single chromosome with the following values:

| chrom | start | end | value |
|-------|-------|-----|-------|
| chr1  | 100   | 200 | 10    |
| chr1  | 200   | 250 | 25    |
| chr1  | 500   | 560 | 17    |
| chr1  | 600   | 700 | 44    |

What would be the value of the variable `mytrack` for a given iterator interval? The resulting value is an average of all values of track `mytrack` covered by the iterator interval. For example, if the iterator interval is `[230, 620)`, the result is an average of 25, 17, and 44. If the iterator interval is `[0, 300)`, the result is an average of 10 and 25. If the iterator interval is `[300, 400)`, the result is NaN.

The same evaluation logic applies for Dense tracks. For Rectangles (2D) tracks, the value is calculated as a *weighted* average of the values covered by the iterator interval, where the weight equals the intersection area of the iterator interval and the 2D interval containing the value.

| Track Type | Value |
|---|---|
| Dense | Average of non-NaN values covered by iterator interval |
| Sparse | Average of non-NaN values covered by iterator interval |
| Rectangles | Weighted average of non-NaN values covered by iterator interval. Each weight equals the intersection area between the iterator interval and the track interval containing the value. |

### Virtual Tracks

So far we have shown that the value of a `mytrack` variable is set to the average (or weighted average) of the track values covered by the iterator interval. But what if we do not want to average the values but rather pick the maximal or minimal value? What if we want to use the percentile of a track value rather than the value itself? And maybe we even want to alter the iterator interval itself on the fly? This is where virtual tracks become useful.

A virtual track is a set of rules that describe how the "source" (a real track, intervals, or a value-based DataFrame) should be processed, and how the iterator interval should be modified. Virtual tracks are created with `pm.gvtrack_create()`:

```python
pm.gvtrack_create("myvtrack", "dense_track")
```

This creates a new virtual track named `myvtrack`. It can be used in track expressions instead of the real track `dense_track`. In this example, `myvtrack` is just an alias of `dense_track`. But we can create a more sophisticated virtual track by specifying a "function":

```python
pm.gvtrack_create("myvtrack", "dense_track", "global.percentile")
```

In this example, when `myvtrack` is evaluated in a track expression, it will return the percentile of V_avg among all values of `dense_track`, where V_avg is an average (or weighted average) of the track values covered by the iterator interval.

Virtual tracks also allow altering the iterator interval "on the fly":

```python
pm.gvtrack_iterator("myvtrack", sshift=-100, eshift=200)
```

This expands each iterator interval by adding -100 to its `start` coordinate and 200 to its `end` coordinate.

Similarly, iterator modifiers can be defined for 2D intervals. Moreover, an iterator modifier can create a 1D interval from a 2D iterator interval by projecting one of its axes:

```python
pm.gvtrack_create("myvtrack", "dense_track")
pm.gvtrack_iterator("myvtrack", dim=2)
```

!!! tip
    Iterator modifiers transform the iterator interval only for the given virtual track. If you have two virtual tracks V0 and V1, each can have different iterator modifications applied to the same base iterator interval.

You can also use intervals as a source for a virtual track. In this case, the value of the virtual track will be some function that takes into account the "source" intervals and the current iterator interval:

```python
pm.gvtrack_create("myvtrack", "annotations", "distance")
intervs = pm.gscreen("dense_track > 0.45")
pm.gextract("myvtrack", pm.gintervals_all(), iterator=intervs)
```

In this example, `myvtrack` returns the minimal distance between intervals from the interval set `annotations` and the center of the current iterator interval from `intervs`.

#### Value-Based Tracks

In addition to database tracks and interval sets, virtual tracks can use **value-based tracks** as sources. Value-based tracks are DataFrames containing genomic intervals with associated numeric values. They function as in-memory sparse tracks without requiring track creation in the database.

To create a value-based virtual track, provide a DataFrame with columns `chrom`, `start`, `end`, and one numeric value column:

```python
import pandas as pd

# Create a DataFrame with intervals and numeric values
intervals_with_values = pd.DataFrame({
    "chrom": ["chr1", "chr1", "chr1"],
    "start": [100, 300, 500],
    "end": [200, 400, 600],
    "score": [10, 20, 30],
})

# Use as value-based sparse track
pm.gvtrack_create("myvtrack", intervals_with_values, "avg")
pm.gvtrack_create("myvtrack_max", intervals_with_values, "max")
```

Value-based tracks support all track-based summarizer functions (e.g., `avg`, `min`, `max`, `sum`, `stddev`, `quantile`, `nearest`, `exists`, `size`, `first`, `last`, `sample`, and position functions). However, they have one important restriction: **intervals must not overlap**. Value-based tracks behave like sparse tracks and require non-overlapping intervals.

!!! info
    Value-based tracks use count-based averaging (each interval contributes equally regardless of length), matching the behavior of sparse tracks.

For a full list of supported functions, see the API reference for `pm.gvtrack_create()`.

### Administrating Virtual Tracks

Virtual tracks define a set of rules for how to access and process the values of the "source" object. The connection between the virtual track and the source object is done via "soft link" -- by name, not by reference. For example, a virtual track will continue to exist until explicitly removed by `pm.gvtrack_rm()` even if the physical track it points to is deleted or renamed.

Operations such as `pm.gdb_init()` and `pm.gdir_cd()` alter the list of available tracks and interval sets. Since these objects are referenced by virtual tracks, virtual tracks are always defined in the context of the current working directory in the Genomic Database (not to be confused with the shell's current working directory). Changing the current working directory will also change the list of available virtual tracks.

Unlike regular tracks whose data is stored on disk, virtual tracks are non-persistent objects in the current Python session. They are stored internally in the `pm._VTRACKS` dictionary. You can also use `pm.gvtrack_info()` for a more convenient way to access virtual track definitions.

```python
# List all virtual tracks
pm.gvtrack_ls()

# Get info about a virtual track
pm.gvtrack_info("myvtrack")

# Remove a virtual track
pm.gvtrack_rm("myvtrack")

# Clear all virtual tracks
pm.gvtrack_clear()
```

!!! warning
    Virtual tracks are stored in Python memory and do not persist between sessions. If you need to preserve virtual track definitions, you must serialize them yourself (e.g., using `pickle` or `json` on the `pm._VTRACKS` dictionary).

### Track Expression Evaluation under Optimization

Previously we described how a track expression `"mytrack * 3"` (where `mytrack` is a track name) leads to an implicit definition of a `mytrack` variable. To simplify the explanation, we presented this variable as a scalar whose value changes with each iterator interval. In reality, the library defines `mytrack` as a NumPy array (vector), not a single scalar. The array is filled with the corresponding values of the track, then the track expression is evaluated, and the result is expected to also be an array of the same size. Working with arrays rather than single scalars reduces the number of evaluations and hence improves run-times.

The size of the array is controlled via `pm.CONFIG['eval_buf_size']`. By default it equals 1000. Altering this value (for instance setting it to 1) might significantly affect the run-time of various functions.

```python
# Change buffer size (not generally recommended)
pm.CONFIG['eval_buf_size'] = 1
```

One might wonder why we should care about `mytrack` being an array rather than a scalar. In many cases it does not matter. For example, `mytrack * 3` produces exactly the same results regardless of whether `mytrack` is an array or a scalar, because NumPy multiplies element-wise.

However, some functions accept an array but return a scalar rather than an array. For example, `np.min()`:

```python
# WRONG: np.min returns a scalar from the entire buffer
"track1 + np.min(np.column_stack([track1, track2]))"

# CORRECT: np.minimum operates element-wise
"track1 + np.minimum(track1, track2)"
```

!!! warning "Ensure your track expressions work correctly on arrays"
    The expression `track1 + np.min(...)` was probably meant to produce a sum of `track1` and the minimum between `track1` and `track2` for each iterator interval. However, since `np.min()` returns a single scalar from the entire buffer, the result would be meaningless. Use element-wise operations like `np.minimum()`, `np.maximum()`, etc.

### Iterators

So far we have discussed in detail how the track expression is evaluated given the *iterator interval*. But how are the iterator intervals controlled?

Most functions that accept track expressions have an additional parameter named `iterator`. The value of this parameter determines the iterator intervals, which is also sometimes called an *iterator policy*:

| Value | Iterator Policy Type | Example | Description |
|---|---|---|---|
| Integer | Fixed Bin | `50` | Iterator intervals advance by a fixed step starting from zero: `[0,50), [50,100), [100,150), ...` |
| Dense track name | Fixed Bin | `"dense_track"` | Use the bin size of the track as a fixed step |
| 1D intervals | 1D Intervals | `"annotations"` | Iterate over the supplied intervals. *Note: intervals are sorted and overlapping intervals are unified.* |
| Sparse track name | 1D Intervals | `"sparse_track"` | Iterate over the intervals of a sparse track |
| `[int, int]` | 2D Intervals | `[1000, 2000]` | 2D iterator intervals cover the whole 2D chromosomal space by rectangles of fixed size: Width x Height |
| 2D intervals | 2D Intervals | `pm.gintervals_2d(1, 2)` | Iterate over the supplied 2D intervals. *Note: intervals are sorted and overlapping is forbidden.* |
| Rectangles track name | 2D Intervals | `"rects_track"` | Iterate over the intervals of a Rectangles track |
| Cartesian grid iterator | 2D Intervals | `pm.giterator_cartesian_grid(...)` | Iterate over a 2D cartesian grid (see `pm.giterator_cartesian_grid()`) |
| `None` | Implicit | `None` | Implicitly determine the iterator policy from tracks in the expression. If no track names are present or two tracks determine different policies, an error is raised. |

!!! warning
    Small 2D rectangles used without a limiting scope might result in an immense number of iterator intervals.

Use `pm.giterator_intervals()` to retrieve the iterator intervals given a track expression, scope, and an iterator:

```python
# See what intervals the iterator will produce
itr_intervals = pm.giterator_intervals(
    "dense_track", pm.gintervals(1, 0, 1000), iterator=200
)
```

### Scope

Many functions that accept track expressions and iterator policies also accept an additional set of intervals that limit the **scope** of the function. This scope also limits the iterator intervals.

```python
result = pm.gextract("dense_track", pm.gintervals(2, 340, 520))
```

As one can notice, the first and last intervals in the result are truncated by the scope `[340, 520)`.

In some cases, the combination of iterator policy and scope might result in a nontrivial set of iterator intervals. Use `pm.giterator_intervals()` to retrieve the iterator intervals given a track expression, scope, and an iterator.

### Band

As explained before, the track expression iterator can be determined implicitly or through the `iterator` parameter. If the iterator intervals are 2D, an additional filter can be applied: a *band*.

A band is a pair of integers: (D1, D2). We say that a 2D iterator interval `(chrom1, x1, x2, chrom2, y1, y2)` intersects a band if and only if:

1. `chrom1 == chrom2`
2. There exist x, y such that `x1 <= x < x2` and `y1 <= y < y2` and `D1 <= x - y < D2`.

Informally, a band can be seen as the space S between two 45-degree diagonals, where D1 and D2 determine where these diagonals cross the X axis. An iterator interval represents a rectangle in 2D space and can be intersected with S. The result of the intersection can be a rectangle, a trapezoid, a triangle, a hexagon, or it can be empty if the interval does not intersect with the band.

If the intersection is non-empty, the *minimal rectangle* bounding the intersected shape replaces the original iterator interval. Otherwise, the iterator interval is skipped.

The `pm.gintervals_2d_band_intersect()` function can help illustrate this concept:

```python
import pandas as pd

intervs = pm.gintervals_2d(1, 200, 800, 1, 100, 1000)
intervs = pd.concat([
    intervs,
    pm.gintervals_2d(1, 900, 950, 1, 0, 200),
    pm.gintervals_2d(1, 0, 100, 1, 0, 400),
    pm.gintervals_2d(1, 900, 950, 2, 0, 200),
], ignore_index=True)

# Intersect with band [500, 1000)
result = pm.gintervals_2d_band_intersect(intervs, band=[500, 1000])
```

In this example:

- The first interval `(chr1, 200, 800, chr1, 100, 1000)` intersects the band and is shrunk to its minimal rectangle.
- The second interval lies entirely within the band and is returned unchanged.
- The third interval lies entirely outside the band and is eliminated.
- The fourth interval has different chromosomes and is filtered out.

The band also affects the result of 2D tracks. When an iterator interval is intersected with a band, the weighted average for the 2D track value uses the actual intersection area -- not the area of the minimal bounding rectangle:

```python
intervs = pm.gintervals_2d(1, [100, 400], [300, 490], 1, [120, 180], [200, 500])
pm.gtrack_2d_create("test2d", "test 2D track", intervs, [10, 20])

# Without band
pm.gextract(
    "test2d",
    pm.gintervals_2d_all(),
    iterator=pm.gintervals_2d(1, 0, 1000, 1, 0, 1000),
)

# With band
pm.gextract(
    "test2d",
    pm.gintervals_2d_all(),
    iterator=pm.gintervals_2d(1, 0, 1000, 1, 0, 1000),
    band=[150, 1000],
)

# Clean up
pm.gtrack_rm("test2d", force=True)
```

!!! note
    The space used in the calculation of the weighted average is the actual space of the intersection, not the space occupied by the minimal rectangles.

## Random Algorithms

Various functions in the library such as `pm.gsample()` make use of a pseudo-random number generator. Each time the function is invoked, a unique series of random numbers is generated. Hence two identical calls might produce different results. To guarantee reproducible results, set the seed before invoking the function:

```python
import numpy as np

np.random.seed(60427)
r1 = pm.gsample("dense_track", 10)
r2 = pm.gsample("dense_track", 10)  # r2 differs from r1

np.random.seed(60427)
r3 = pm.gsample("dense_track", 10)  # r3 == r1
```

## Multitasking

### Controlling the Number of Processes

To boost runtime performance, various functions in the library support multitasking mode -- parallel computation of the result by several concurrent processes.

The multitasking behavior is controlled through the `pm.CONFIG` dictionary:

```python
# View current settings
print(pm.CONFIG['multitasking'])    # True/False -- enable/disable parallelism
print(pm.CONFIG['max_processes'])   # Upper bound on worker processes
print(pm.CONFIG['min_processes'])   # Minimum workers for multitasking

# Adjust settings
pm.CONFIG['multitasking'] = True
pm.CONFIG['max_processes'] = 8
```

Multitasking can be completely switched off:

```python
pm.CONFIG['multitasking'] = False
```

### Limiting the Memory Consumption

For certain functions, multitasking might result in higher memory consumption. The `pm.CONFIG['max_data_size']` parameter controls the maximum number of rows in memory.

```python
# View current buffer size
print(pm.CONFIG['max_data_size'])   # Default: 10,000,000

# Reduce if memory is limited
pm.CONFIG['max_data_size'] = 1000000
```

Some functions such as `pm.gscreen()` or `pm.gextract()` consume memory proportional to `max_data_size` in multitasking mode.

!!! tip
    To limit memory consumption in multitasking mode, lower `max_data_size` or switch off multitasking entirely.

### Other Considerations

In multitasking mode, the return value of `pm.gquantiles()` may vary depending on the number of CPU cores. For more details, refer to the API documentation for this function.
