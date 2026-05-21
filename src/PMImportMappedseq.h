#ifndef PM_IMPORT_MAPPEDSEQ_H
#define PM_IMPORT_MAPPEDSEQ_H

#include <Python.h>

// Entry point for SAM/tab-delimited mapped-sequence import.
// Args (positional): track_dir, file_path, pileup, binsize, cols_order,
//                    remove_dups.
// cols_order: tuple of 4 1-based ints [seq, chrom, coord, strand] or
//             None (SAM mode: 10/3/4/2 + per-line @-header skip +
//             strand from flag bit 0x10).
// Returns dict {"total": {...}, "chrom_stats": {...}}.
// Skeleton implementation only - parser + writers wired up by
// PMImportMappedseq Tasks 3-5.
PyObject *pm_import_mappedseq(PyObject *self, PyObject *args);

#endif
