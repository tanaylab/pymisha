# Roadmap

## Current status (v0.1.0)

PyMisha covers 123 of 145 R misha exports with full database interoperability. All core 1D and 2D workflows are implemented with C++ streaming backends for genome-scale performance.

## Known gaps

- **Track Arrays:** `gtrack.array.*` and `gvtrack.array.slice` are not implemented.
- **Legacy Conversion:** `gtrack.convert` (for migrating old 2D formats) is not implemented.

## Future plans

- Track array support.
- Performance improvements for Python-side virtual track fallback paths.
- Extended documentation and tutorials.
- PyPI packaging.
