// pm_compute_strands_autocorr: C++ port of R misha's GenomeComputeStrandAutocorr.cpp
// Computes strand cross-correlation from mapped-reads file.

#include "pymisha.h"
#include "BufferedFile.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cstdint>

/*
 * pm_compute_strands_autocorr(filename, chrom, chromsize, binsize, maxread,
 *                              cols_order, min_coord, max_coord)
 *
 * filename:    str — path to tab-delimited mapped reads file
 * chrom:       str — chromosome name to analyze
 * chromsize:   int — size of chromosome in bp
 * binsize:     int — bin size in bp
 * maxread:     int — max read length (controls offset range)
 * cols_order:  tuple of 4 ints — 1-based column indices for (seq, chrom, coord, strand)
 * min_coord:   int — minimum coordinate for analysis window
 * max_coord:   int — maximum coordinate for analysis window
 *
 * Returns: tuple(stats_dict, bins_df)
 *   stats_dict: {"forward_mean", "forward_stdev", "reverse_mean", "reverse_stdev"}
 *   bins_df: DataFrame with "bin" and "corr" columns
 */
PyObject *pm_compute_strands_autocorr(PyObject *self, PyObject *args)
{
    const char *filename = nullptr;
    const char *chrom = nullptr;
    long long chromsize = 0;
    int binsize = 0;
    int maxread = 400;
    PyObject *py_cols_order = nullptr;
    long long min_coord = 0;
    long long max_coord = 300000000;

    if (!PyArg_ParseTuple(args, "ssLi|iOLL",
                          &filename, &chrom, &chromsize,
                          &binsize, &maxread, &py_cols_order,
                          &min_coord, &max_coord)) {
        return nullptr;
    }

    try {
        // Validate parameters
        if (binsize <= 0) {
            PyErr_Format(PyExc_ValueError, "Invalid binsize value %d", binsize);
            return nullptr;
        }
        if (maxread <= 0) {
            PyErr_Format(PyExc_ValueError, "Invalid maxread value %d", maxread);
            return nullptr;
        }
        if (chromsize <= 0) {
            PyErr_Format(PyExc_ValueError, "Invalid chromsize value %lld", chromsize);
            return nullptr;
        }

        // Parse cols_order (1-based column indices)
        enum { SEQ_COL, CHROM_COL, COORD_COL, STRAND_COL, NUM_COLS };
        const char *COL_NAMES[NUM_COLS] = {"sequence", "chromosome", "coordinate", "strand"};
        int cols_order[NUM_COLS] = {9, 11, 13, 14}; // defaults

        if (py_cols_order && py_cols_order != Py_None) {
            if (!PyTuple_Check(py_cols_order) && !PyList_Check(py_cols_order)) {
                PyErr_SetString(PyExc_TypeError, "cols_order must be a tuple or list");
                return nullptr;
            }
            Py_ssize_t n = PySequence_Size(py_cols_order);
            if (n != NUM_COLS) {
                PyErr_Format(PyExc_ValueError, "cols_order must have exactly %d elements", NUM_COLS);
                return nullptr;
            }
            for (int i = 0; i < NUM_COLS; i++) {
                PMPY item(PySequence_GetItem(py_cols_order, i), true);
                if (!item) return nullptr;
                long val = PyLong_AsLong(*item);
                if (val == -1 && PyErr_Occurred()) return nullptr;
                if (val <= 0) {
                    PyErr_Format(PyExc_ValueError, "Invalid columns order: %s column's order is %ld",
                                 COL_NAMES[i], val);
                    return nullptr;
                }
                cols_order[i] = (int)val;
            }
            // Check for duplicates
            for (int i = 0; i < NUM_COLS; i++) {
                for (int j = i + 1; j < NUM_COLS; j++) {
                    if (cols_order[i] == cols_order[j]) {
                        PyErr_Format(PyExc_ValueError,
                                     "Invalid columns order: %s column has the same order as %s column",
                                     COL_NAMES[i], COL_NAMES[j]);
                        return nullptr;
                    }
                }
            }
        }

        // Clamp coordinates
        if (min_coord < 0) min_coord = 0;
        if (max_coord < 0 || max_coord > chromsize) max_coord = chromsize;

        const int MAX_COV = 10;
        int64_t n_bins_cov = (int64_t)std::ceil((double)chromsize / binsize);
        std::vector<int> forward(n_bins_cov, 0);
        std::vector<int> reverse(n_bins_cov, 0);

        // Open file with BufferedFile (same as R misha)
        BufferedFile infile;
        infile.open(filename, "r");
        if (infile.error()) {
            PyErr_Format(PyExc_FileNotFoundError, "Failed to open file %s", filename);
            return nullptr;
        }

        // Parse file character by character (matching R misha exactly)
        int col = 1;
        int active_col_idx = -1;
        int c;
        std::string str[NUM_COLS];

        // Find which column index is first (col == 1)
        for (int i = 0; i < NUM_COLS; i++) {
            if (cols_order[i] == 1) {
                active_col_idx = i;
                break;
            }
        }

        while (true) {
            c = infile.getc();
            if (c == '\n' || c == EOF || c == '\t') {
                if (c == '\n' || c == EOF) {
                    int num_nonempty = 0;
                    for (int i = 0; i < NUM_COLS; i++) {
                        if (!str[i].empty()) num_nonempty++;
                    }

                    // Process line if all columns present
                    if (num_nonempty == NUM_COLS) {
                        // Check chromosome
                        if (str[CHROM_COL] == chrom) {
                            // Parse coordinate
                            char *endptr;
                            int64_t coord = strtoll(str[COORD_COL].c_str(), &endptr, 10);
                            if (*endptr == '\0' && coord >= 0 && coord < chromsize &&
                                coord >= min_coord && coord <= max_coord) {
                                if (str[STRAND_COL] == "+" || str[STRAND_COL] == "F") {
                                    int64_t idx = coord / binsize;
                                    if (idx >= 0 && idx < n_bins_cov) {
                                        forward[idx] = std::min(MAX_COV, forward[idx] + 1);
                                    }
                                } else if (str[STRAND_COL] == "-" || str[STRAND_COL] == "R") {
                                    // A minus-strand read is projected to its 3' end
                                    // (coord + read length), which can land at/after
                                    // chromsize for a read near the contig end; clamp to
                                    // the last bin so the write stays in bounds (R-parity).
                                    int64_t idx = (coord + (int64_t)str[SEQ_COL].size()) / binsize;
                                    if (idx >= n_bins_cov)
                                        idx = n_bins_cov - 1;
                                    reverse[idx] = std::min(MAX_COV, reverse[idx] + 1);
                                }
                            }
                        }
                    }

                    if (c == EOF) break;

                    if (num_nonempty > 0) {
                        for (int i = 0; i < NUM_COLS; i++) str[i].clear();
                    }
                    col = 1;
                } else {
                    col++;
                }

                active_col_idx = -1;
                for (int i = 0; i < NUM_COLS; i++) {
                    if (cols_order[i] == col) {
                        active_col_idx = i;
                        break;
                    }
                }
            } else if (active_col_idx >= 0) {
                str[active_col_idx].push_back((char)c);
            }
        }

        if (infile.error()) {
            PyErr_Format(PyExc_IOError, "Error while reading file %s", filename);
            return nullptr;
        }

        // Compute autocorrelation (matching R misha exactly)
        int min_off = (int)(-maxread / binsize);
        int max_off = (int)(maxread / binsize);
        int64_t min_idx = (int64_t)(max_off + min_coord / binsize);
        int64_t max_idx = (int64_t)(max_coord / binsize - max_off - 1);

        if (min_idx >= (int64_t)forward.size() || max_idx < 0) {
            PyErr_SetString(PyExc_ValueError, "Not enough data to calculate auto correlation.");
            return nullptr;
        }

        // Clamp
        if (min_idx < 0) min_idx = 0;
        if (max_idx > (int64_t)forward.size()) max_idx = (int64_t)forward.size();

        int64_t count = 0;
        int64_t tot_f = 0;
        int64_t tot_r = 0;
        int64_t tot_ff = 0;
        int64_t tot_rr = 0;
        int n_offsets = max_off - min_off;
        std::vector<double> tot_fr(n_offsets, 0.0);

        for (int64_t i = min_idx; i < max_idx; i++) {
            int cur_fr = forward[i];
            int cur_rv = reverse[i];
            tot_f += cur_fr;
            tot_r += cur_rv;
            count++;
            tot_rr += cur_rv * cur_rv;
            tot_ff += cur_fr * cur_fr;
            for (int off = min_off; off < max_off; off++) {
                tot_fr[off - min_off] += cur_fr * reverse[i + off];
            }
        }

        if (count == 0) {
            PyErr_SetString(PyExc_ValueError, "Not enough data to calculate auto correlation.");
            return nullptr;
        }

        double mean_f = tot_f / (double)count;
        double mean_r = tot_r / (double)count;
        double std_f = std::sqrt(std::max(0.0, tot_ff / (double)count - mean_f * mean_f));
        double std_r = std::sqrt(std::max(0.0, tot_rr / (double)count - mean_r * mean_r));

        // Build result: tuple(stats_dict, (bin_array, corr_array))
        // Stats dict
        PMPY py_stats(PyDict_New(), true);
        if (!py_stats) return nullptr;
        // PyDict_SetItemString does NOT steal a reference, so each freshly
        // created float must be DECREF'd after insertion to avoid leaking it.
        auto set_float = [](PyObject *d, const char *key, double v) {
            PyObject *f = PyFloat_FromDouble(v);
            PyDict_SetItemString(d, key, f);
            Py_DECREF(f);
        };
        set_float(*py_stats, "forward_mean", mean_f);
        set_float(*py_stats, "forward_stdev", std_f);
        set_float(*py_stats, "reverse_mean", mean_r);
        set_float(*py_stats, "reverse_stdev", std_r);

        // Bin and correlation arrays
        npy_intp dims[1] = {(npy_intp)n_offsets};
        PMPY py_bins(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        PMPY py_corr(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        if (!py_bins || !py_corr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate result arrays");
            return nullptr;
        }
        double *bins_data = (double *)PyArray_DATA((PyArrayObject *)*py_bins);
        double *corr_data = (double *)PyArray_DATA((PyArrayObject *)*py_corr);

        double denom = std_f * std_r;
        for (int off = min_off; off < max_off; off++) {
            int k = off - min_off;
            bins_data[k] = (double)off;
            if (denom > 0) {
                corr_data[k] = (tot_fr[k] / (double)count - mean_f * mean_r) / denom;
            } else {
                corr_data[k] = 0.0;
            }
        }

        // Return (stats_dict, (bins_array, corr_array))
        // Python side converts the arrays to a DataFrame
        // Py_BuildValue with "N" steals references, so we Py_INCREF first
        // since PMPY will Py_DECREF on destruction.
        Py_INCREF(*py_stats);
        Py_INCREF(*py_bins);
        Py_INCREF(*py_corr);
        return Py_BuildValue("(N(NN))", *py_stats, *py_bins, *py_corr);

    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
