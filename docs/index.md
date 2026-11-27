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
- **Guides for LLM agents** - [Reference docs](#using-pymisha-with-an-llm-agent) for Claude Code, Copilot or Cursor writing pymisha code

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

Reading and writing figwig (`.fw`) multi-track files (`gtrack_import_figwig`, `gtrack_export_figwig`) needs the optional [figwig](https://github.com/jmschrei/figwig) package:

```bash
pip install "pymisha[figwig]"
```

## Using pymisha with an LLM agent

LLM coding agents (Claude Code, Copilot, Cursor) writing pymisha code can pre-load these guides for fewer hallucinated APIs and more idiomatic recipes:

- [pymisha-core.md](https://github.com/tanaylab/pymisha/blob/main/agent-guides/pymisha-core.md) - concepts, chooser tables, everyday recipes. Start here.
- [pymisha-advanced.md](https://github.com/tanaylab/pymisha/blob/main/agent-guides/pymisha-advanced.md) - 2D/Hi-C, PWM, import/export, new genomes, gsynth.
- [pymisha-anti-patterns.md](https://github.com/tanaylab/pymisha/blob/main/agent-guides/pymisha-anti-patterns.md) - silent footguns cross-referenced from the above.
- [skills/importing-tracks](https://github.com/tanaylab/pymisha/blob/main/agent-guides/skills/importing-tracks/SKILL.md) - track import, loaded only when the task calls for it.

**Drop-in prompt (no clone needed).** Paste this into your agent at the start of a pymisha task:

```
Before writing any pymisha code, fetch and read:

- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-core.md  (mandatory: concepts + everyday recipes)
- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-anti-patterns.md  (silent footguns; cross-referenced from core)
- https://raw.githubusercontent.com/tanaylab/pymisha/main/agent-guides/pymisha-advanced.md  (consult on demand: 2D/Hi-C, PWM, import/export, new genomes)

Follow the conventions in those files. When you hit a recipe with an
"Avoid:" block, treat it as a hard rule.
```

For Claude Code-style setups, putting `pymisha-core.md` (or all three) into the project's `CLAUDE.md` / `AGENTS.md` works too. The guides mirror [R misha's](https://github.com/tanaylab/misha/tree/master/agent-guides), translated to the pymisha API.
