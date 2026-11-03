#ifndef PM_IMPORT_MAPPEDSEQ_H
#define PM_IMPORT_MAPPEDSEQ_H

#include <Python.h>

// Entry point for SAM/BAM/tab-delimited mapped-sequence import (R's gtrackimport_mappedseq).
// Args (positional): track_dir, file_path, pileup, binsize, cols_order,
//                    remove_dups[, paired, min_mapq, max_fraglen, one_based].
// cols_order: tuple of 4 1-based ints [seq, chrom, coord, strand] or
//             None (SAM). With paired and cols_order not None the file is
//             read as fragments (chrom, start, end in columns 1-3).
// Returns dict {"total": {...}, "chrom_stats": {...}}.
PyObject *pm_import_mappedseq(PyObject *self, PyObject *args);

#endif
