# Creating Genome Databases

This tutorial covers how to create a PyMisha genomic database from UCSC genome assemblies. PyMisha supports five commonly used genomes out of the box: **hg19**, **hg38**, **mm9**, **mm10**, and **mm39**.

## Quick Start: Prebuilt Genomes

The easiest way to create a database is with `pm.gdb_create_genome()`. This downloads a prebuilt, ready-to-use database archive from a hosted repository and initializes it automatically.

```python
import pymisha as pm

pm.gdb_create_genome("hg19")  # Human, GRCh37
pm.gdb_create_genome("hg38")  # Human, GRCh38
pm.gdb_create_genome("mm9")   # Mouse, NCBI37
pm.gdb_create_genome("mm10")  # Mouse, GRCm38
pm.gdb_create_genome("mm39")  # Mouse, GRCm39
```

By default the database directory is created under the current working directory with the genome name (e.g., `./hg38/`). You can control the location with the `path` parameter:

```python
pm.gdb_create_genome("hg38", path="/data/genomes")
# Creates /data/genomes/hg38/
```

!!! tip "Checksum verification"
    By default, `gdb_create_genome` verifies a SHA-256 checksum after downloading the archive. You can disable this with `verify_checksum=False` if needed, but it is recommended to keep it enabled.

After `gdb_create_genome` completes, the database is already initialized -- you can start querying immediately:

```python
pm.gdb_create_genome("hg38", path="/data/genomes")
pm.gchrom_sizes()  # Shows chromosome names and sizes
```

## Manual Database Creation from UCSC

If you need to create a database for a genome not covered by `gdb_create_genome`, or if you want to build from the latest UCSC files directly, you can use `pm.gdb_create()` with local FASTA files.

The general workflow is:

1. Download chromosome FASTA files from UCSC
2. Call `pm.gdb_create()` with the paths to those files
3. Initialize the database with `pm.gdb_init()`

!!! note "FTP vs HTTPS"
    UCSC provides genome data via both FTP and HTTPS. The examples below use HTTPS URLs (`https://hgdownload.soe.ucsc.edu/...`), which work in most environments without requiring an FTP client.

### Downloading Chromosome Files

You can download chromosome FASTA files using Python directly, or with command-line tools like `wget` or `curl`.

=== "Python"

    ```python
    import urllib.request
    from pathlib import Path

    genome = "hg38"
    base_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/chromosomes"
    chroms = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]

    download_dir = Path(f"{genome}_fasta")
    download_dir.mkdir(exist_ok=True)

    for chrom in chroms:
        url = f"{base_url}/{chrom}.fa.gz"
        dest = download_dir / f"{chrom}.fa.gz"
        if not dest.exists():
            print(f"Downloading {chrom}...")
            urllib.request.urlretrieve(url, dest)
    ```

=== "wget"

    ```bash
    GENOME=hg38
    BASE=https://hgdownload.soe.ucsc.edu/goldenPath/${GENOME}/chromosomes
    mkdir -p ${GENOME}_fasta
    for CHR in $(seq 1 22) X Y M; do
        wget -P ${GENOME}_fasta ${BASE}/chr${CHR}.fa.gz
    done
    ```

=== "curl"

    ```bash
    GENOME=hg38
    BASE=https://hgdownload.soe.ucsc.edu/goldenPath/${GENOME}/chromosomes
    mkdir -p ${GENOME}_fasta
    for CHR in $(seq 1 22) X Y M; do
        curl -o ${GENOME}_fasta/chr${CHR}.fa.gz ${BASE}/chr${CHR}.fa.gz
    done
    ```

### Building the Database

Once the FASTA files are downloaded, create the database:

```python
import pymisha as pm
from pathlib import Path

genome = "hg38"
download_dir = Path(f"{genome}_fasta")

# Collect all chromosome FASTA files
chroms = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]
fasta_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

# Create the database
pm.gdb_create(genome, fasta_files, verbose=True)

# Initialize
pm.gdb_init(genome)
```

!!! info "Database format"
    By default, `gdb_create` uses the `"indexed"` format, which stores all sequences in a single `genome.seq` file with an accompanying `genome.idx` index. This is the recommended format. You can also use `db_format="per-chromosome"` to store one `.seq` file per contig, which can then be converted later with `pm.gdb_convert_to_indexed()`.

---

## Genome-Specific Examples

Below are complete examples for each supported genome. The chromosome lists differ between human (1-22, X, Y, M) and mouse (1-19, X, Y, M).

### hg19 (Human, GRCh37)

```python
import pymisha as pm
from pathlib import Path

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes"
chroms = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]
fasta_files = [f"{base_url}/{chrom}.fa.gz" for chrom in chroms]

# Download locally first, then create
# (see download examples above)
download_dir = Path("hg19_fasta")
local_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

pm.gdb_create("hg19", local_files)
pm.gdb_init("hg19")
```

UCSC data URLs:

- Chromosomes: `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/`
- Gene annotations: `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/knownGene.txt.gz`
- Gene cross-references: `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/kgXref.txt.gz`

### hg38 (Human, GRCh38)

```python
import pymisha as pm
from pathlib import Path

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes"
chroms = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]]

download_dir = Path("hg38_fasta")
local_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

pm.gdb_create("hg38", local_files)
pm.gdb_init("hg38")
```

UCSC data URLs:

- Chromosomes: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/`
- Gene annotations: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/knownGene.txt.gz`
- Gene cross-references: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/kgXref.txt.gz`

### mm9 (Mouse, NCBI37)

```python
import pymisha as pm
from pathlib import Path

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/chromosomes"
chroms = [f"chr{c}" for c in list(range(1, 20)) + ["X", "Y", "M"]]

download_dir = Path("mm9_fasta")
local_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

pm.gdb_create("mm9", local_files)
pm.gdb_init("mm9")
```

UCSC data URLs:

- Chromosomes: `https://hgdownload.soe.ucsc.edu/goldenPath/mm9/chromosomes/`
- Gene annotations: `https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/knownGene.txt.gz`
- Gene cross-references: `https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/kgXref.txt.gz`

### mm10 (Mouse, GRCm38)

```python
import pymisha as pm
from pathlib import Path

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes"
chroms = [f"chr{c}" for c in list(range(1, 20)) + ["X", "Y", "M"]]

download_dir = Path("mm10_fasta")
local_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

pm.gdb_create("mm10", local_files)
pm.gdb_init("mm10")
```

UCSC data URLs:

- Chromosomes: `https://hgdownload.soe.ucsc.edu/goldenPath/mm10/chromosomes/`
- Gene annotations: `https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/knownGene.txt.gz`
- Gene cross-references: `https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/kgXref.txt.gz`

### mm39 (Mouse, GRCm39)

```python
import pymisha as pm
from pathlib import Path

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes"
chroms = [f"chr{c}" for c in list(range(1, 20)) + ["X", "Y", "M"]]

download_dir = Path("mm39_fasta")
local_files = [str(download_dir / f"{chrom}.fa.gz") for chrom in chroms]

pm.gdb_create("mm39", local_files)
pm.gdb_init("mm39")
```

UCSC data URLs:

- Chromosomes: `https://hgdownload.soe.ucsc.edu/goldenPath/mm39/chromosomes/`
- Gene annotations: `https://hgdownload.soe.ucsc.edu/goldenPath/mm39/database/knownGene.txt.gz`
- Gene cross-references: `https://hgdownload.soe.ucsc.edu/goldenPath/mm39/database/kgXref.txt.gz`

---

## Gene Annotations

!!! warning "Gene import not yet supported"
    The R misha `gdb.create()` function accepts `genes_file`, `annots_file`, and `annots_names` parameters to import UCSC gene annotations (e.g., knownGene) during database creation. These parameters are accepted by `pm.gdb_create()` for API compatibility but are **not yet implemented** in PyMisha. The UCSC URLs for gene annotation files are listed above for each genome for future use.

## Linked Databases

If you already have a genome database and want to create a separate workspace that shares the same sequence data, use `pm.gdb_create_linked()`:

```python
pm.gdb_create_linked("~/my_project", parent="/shared/genomes/hg38")
```

This creates a new database at `~/my_project` with a writable `tracks/` directory and symlinks to the parent's `seq/` and `chrom_sizes.txt`. This is useful for maintaining separate track collections without duplicating large sequence files.

## Converting to Indexed Format

If you have an older per-chromosome database (one `.seq` file per contig), you can convert it to the more efficient indexed format:

```python
pm.gdb_convert_to_indexed(
    groot="/path/to/mydb",
    convert_tracks=True,
    convert_intervals=True,
    remove_old_files=True,
    verbose=True,
)
```

## API Reference

| Function | Description |
|---|---|
| [`pm.gdb_create_genome()`][pymisha.gdb_create_genome] | Download and initialize a prebuilt genome |
| [`pm.gdb_create()`][pymisha.gdb_create] | Create a database from local FASTA files |
| [`pm.gdb_create_linked()`][pymisha.gdb_create_linked] | Create a linked database sharing sequence data |
| [`pm.gdb_convert_to_indexed()`][pymisha.gdb_convert_to_indexed] | Convert per-chromosome format to indexed |
| [`pm.gdb_init()`][pymisha.gdb_init] | Initialize a database connection |
