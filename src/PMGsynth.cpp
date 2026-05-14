/*
 * PMGsynth.cpp
 *
 * Python C API wrappers for genome synthesis functions:
 *   pm_gsynth_train       - Train stratified Markov model (variable order k)
 *   pm_gsynth_sample      - Sample synthetic genome from trained model
 *   pm_gsynth_replace_kmer - Iterative k-mer replacement
 */

#include "pymisha.h"
#include "PMDb.h"
#include "GenomeSeqFetch.h"
#include "GenomeChromKey.h"
#include "GInterval.h"
#include "StratifiedMarkovModel.h"
#include "MaskUtils.h"
#include "BufferedFile.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
#include <string>

// Defined in PMStubs.cpp
extern void convert_py_intervals(PyObject *py_intervals,
                                 std::vector<GInterval> &intervals);

// ============================================================================
// FASTA helper + .fai support
// ============================================================================

struct FaiEntry {
    std::string name;
    long long   length;
    long long   offset;
    int         linebases;
    int         linewidth;
};

// Write a single FASTA record and record a FaiEntry describing it.
// header_byte_offset is the byte position of the start of the header line
// (taken from fasta_ofs.tellp() BEFORE writing the header). offset stored in
// `out` is the position of the first sequence base, matching samtools faidx.
static void write_fasta_record(std::ofstream &ofs,
                               const std::string &name,
                               const std::vector<char> &seq,
                               int line_width,
                               long long header_byte_offset,
                               FaiEntry *out) {
    ofs << ">" << name << "\n";
    for (size_t i = 0; i < seq.size(); i += line_width) {
        size_t len = std::min(static_cast<size_t>(line_width), seq.size() - i);
        ofs.write(&seq[i], len);
        ofs << "\n";
    }
    if (out) {
        out->name   = name;
        out->length = static_cast<long long>(seq.size());
        out->offset = header_byte_offset + 1 +
                      static_cast<long long>(name.size()) + 1;
        out->linebases = (out->length > 0)
                            ? std::min<int>(line_width, static_cast<int>(out->length))
                            : 0;
        out->linewidth = (out->length > 0) ? out->linebases + 1 : 0;
    }
}

// Flush a collected vector of FaiEntry to <fasta_path>.fai (samtools format).
// On failure to open the file, sets a Python RuntimeError and returns; the
// caller should check PyErr_Occurred().
static void flush_fai(const std::string &fasta_path,
                      const std::vector<FaiEntry> &entries) {
    std::string fai_path = fasta_path + ".fai";
    FILE *ffai = std::fopen(fai_path.c_str(), "w");
    if (!ffai) {
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to open .fai file for writing: %s",
                     fai_path.c_str());
        return;
    }
    for (const auto &e : entries) {
        std::fprintf(ffai, "%s\t%lld\t%lld\t%d\t%d\n",
                     e.name.c_str(), e.length, e.offset,
                     e.linebases, e.linewidth);
    }
    std::fclose(ffai);
}

// ============================================================================
// pm_gsynth_train
// ============================================================================

/*
 * Train a stratified Markov model from genome sequence data.
 *
 * Python args:
 *   intervals      - DataFrame (internal list format) of genomic intervals
 *   bin_indices    - numpy int32 array, flat bin index per iterator position
 *   iter_starts    - numpy int64 array, start position of each iterator interval
 *   iter_chroms    - numpy int32 array, chromid of each iterator interval
 *   breaks         - Python list of float, bin boundaries (num_bins+1 elements)
 *   bin_map        - numpy int32 array or None, bin mapping for merging sparse bins
 *   mask           - DataFrame or None, intervals to exclude
 *   pseudocount    - float, pseudocount for CDF normalization
 *   k              - int, Markov order (1..10, default 5)
 *
 * Returns:
 *   dict with keys: 'counts' (list of 2D numpy arrays), 'cdf' (list of 2D numpy),
 *   'per_bin_kmers' (numpy int64), 'total_kmers' (int), 'total_masked' (int),
 *   'total_n' (int), 'k' (int), 'num_kmers' (int)
 */
PyObject *pm_gsynth_train(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_intervals = NULL;
        PyObject *py_bin_indices = NULL;
        PyObject *py_iter_starts = NULL;
        PyObject *py_iter_chroms = NULL;
        PyObject *py_breaks = NULL;
        PyObject *py_bin_map = NULL;
        PyObject *py_mask = NULL;
        double pseudocount = 1.0;
        int k = 5;
        long long iter_size_arg = 0;
        const char *prior_mode_arg = "uniform";
        PyObject *py_prior_matrix = NULL;

        if (!PyArg_ParseTuple(args, "OOOOOOOdiLsO",
                              &py_intervals, &py_bin_indices,
                              &py_iter_starts, &py_iter_chroms,
                              &py_breaks, &py_bin_map, &py_mask,
                              &pseudocount, &k, &iter_size_arg,
                              &prior_mode_arg, &py_prior_matrix)) {
            verror("Invalid arguments to pm_gsynth_train");
        }
        std::string prior_mode = prior_mode_arg ? prior_mode_arg : "uniform";

        // Validate k
        if (k < 1 || k > StratifiedMarkovModel::MAX_K) {
            verror("Markov order k must be in [1, %d], got %d",
                   StratifiedMarkovModel::MAX_K, k);
        }

        // Validate iterator: callers must pass the iterator value used during
        // bin extraction. Inferring it from iter_starts diffs silently breaks
        // when intervals are not aligned to the iterator bin boundary (the
        // first diff equals the partial first bin width, not the iterator).
        if (iter_size_arg <= 0) {
            verror("iterator must be a positive integer, got %lld", iter_size_arg);
        }
        int64_t iter_size = (int64_t)iter_size_arg;

        int num_kmers = 1 << (2 * k);  // 4^k

        // --- Parse breaks ---
        if (!PyList_Check(py_breaks)) {
            verror("breaks must be a list");
        }
        Py_ssize_t num_breaks = PyList_Size(py_breaks);
        int num_bins = (int)(num_breaks - 1);
        if (num_bins <= 0) {
            verror("breaks must have at least 2 elements");
        }
        std::vector<double> breaks_vec(num_breaks);
        for (Py_ssize_t i = 0; i < num_breaks; ++i) {
            PyObject *item = PyList_GetItem(py_breaks, i);
            breaks_vec[i] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                verror("breaks must be numeric");
            }
        }

        // --- Parse bin indices ---
        PMPY arr_bins(PyArray_FROM_OTF(py_bin_indices, NPY_INT32,
                                        NPY_ARRAY_IN_ARRAY), true);
        if (!arr_bins) {
            PyErr_Clear();
            verror("bin_indices must be convertible to int32 array");
        }
        int num_iter_positions = (int)PyArray_SIZE((PyArrayObject *)(PyObject *)arr_bins);
        int32_t *bin_indices = (int32_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_bins);

        // --- Parse iter_starts ---
        PMPY arr_starts(PyArray_FROM_OTF(py_iter_starts, NPY_INT64,
                                          NPY_ARRAY_IN_ARRAY), true);
        if (!arr_starts) {
            PyErr_Clear();
            verror("iter_starts must be convertible to int64 array");
        }
        int64_t *iter_starts = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_starts);

        // --- Parse iter_chroms ---
        PMPY arr_chroms(PyArray_FROM_OTF(py_iter_chroms, NPY_INT32,
                                          NPY_ARRAY_IN_ARRAY), true);
        if (!arr_chroms) {
            PyErr_Clear();
            verror("iter_chroms must be convertible to int32 array");
        }
        int32_t *iter_chroms = (int32_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_chroms);

        // --- Parse bin_map (optional) ---
        std::vector<int> bin_map_vec;
        bool has_bin_map = (py_bin_map && py_bin_map != Py_None);
        if (has_bin_map) {
            PMPY arr_bm(PyArray_FROM_OTF(py_bin_map, NPY_INT32,
                                          NPY_ARRAY_IN_ARRAY), true);
            if (!arr_bm) {
                PyErr_Clear();
                verror("bin_map must be convertible to int32 array");
            }
            Py_ssize_t bm_len = PyArray_SIZE((PyArrayObject *)(PyObject *)arr_bm);
            int32_t *bm_data = (int32_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_bm);
            bin_map_vec.assign(bm_data, bm_data + bm_len);
        }

        // --- Parse mask intervals ---
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        int num_chroms = (int)chromkey.get_num_chroms();

        std::vector<std::vector<GInterval>> mask_per_chrom(num_chroms);
        if (py_mask && py_mask != Py_None) {
            std::vector<GInterval> mask_intervals;
            convert_py_intervals(py_mask, mask_intervals);
            // Sort and distribute by chromosome
            std::sort(mask_intervals.begin(), mask_intervals.end(),
                      [](const GInterval &a, const GInterval &b) {
                          return a.chromid < b.chromid ||
                                 (a.chromid == b.chromid && a.start < b.start);
                      });
            for (const auto &iv : mask_intervals) {
                if (iv.chromid >= 0 && iv.chromid < num_chroms) {
                    mask_per_chrom[iv.chromid].push_back(iv);
                }
            }
        }

        // --- Initialize model ---
        StratifiedMarkovModel model;
        model.init(num_bins, breaks_vec, k);

        // --- Set up sequence fetcher ---
        GenomeSeqFetch seqfetch;
        seqfetch.set_seqdir(g_pmdb->groot() + "/seq");

        // --- Build per-chromosome bin lookup ---
        std::vector<std::vector<std::pair<int64_t, int>>> chrom_bins(num_chroms);
        for (int i = 0; i < num_iter_positions; ++i) {
            int chromid = iter_chroms[i];
            if (chromid >= 0 && chromid < num_chroms) {
                chrom_bins[chromid].push_back({iter_starts[i], bin_indices[i]});
            }
        }
        for (int c = 0; c < num_chroms; ++c) {
            std::sort(chrom_bins[c].begin(), chrom_bins[c].end());
        }

        // --- Parse sample intervals ---
        std::vector<std::vector<GInterval>> intervals_per_chrom(num_chroms);
        {
            std::vector<GInterval> all_intervals;
            convert_py_intervals(py_intervals, all_intervals);
            for (auto &iv : all_intervals) {
                if (iv.chromid >= 0 && iv.chromid < num_chroms) {
                    intervals_per_chrom[iv.chromid].push_back(iv);
                }
            }
        }

        // --- Train model ---
        // Window size is k+1: k bases for context + 1 for next base
        int window_size = k + 1;
        uint64_t total_masked = 0;
        uint64_t total_n = 0;
        std::vector<char> seq_buf;

        for (int chromid = 0; chromid < num_chroms; ++chromid) {
            const auto &intervals = intervals_per_chrom[chromid];
            if (intervals.empty()) continue;

            int64_t chrom_size = chromkey.get_chrom_size(chromid);
            if (chrom_size <= 0) continue;

            const auto &mask_ivs = mask_per_chrom[chromid];
            const auto &bins = chrom_bins[chromid];

            for (const auto &iv : intervals) {
                int64_t interval_start = std::max<int64_t>(0, iv.start);
                int64_t interval_end = std::min<int64_t>(chrom_size, iv.end);
                if (interval_end <= interval_start) continue;

                // Read sequence for this interval
                GInterval read_iv(chromid, interval_start, interval_end, 0);
                seqfetch.read_interval(read_iv, chromkey, seq_buf);

                size_t mask_cursor = 0;
                // Advance mask cursor past intervals before us
                while (mask_cursor < mask_ivs.size() &&
                       mask_ivs[mask_cursor].end <= interval_start) {
                    ++mask_cursor;
                }

                size_t bin_cursor = 0;
                while (bin_cursor + 1 < bins.size() &&
                       interval_start >= bins[bin_cursor + 1].first) {
                    ++bin_cursor;
                }

                // Process each position in the interval
                for (int64_t pos = interval_start; pos + k < interval_end; ++pos) {
                    int64_t rel = pos - interval_start;

                    // Check mask
                    if (is_position_masked(pos, mask_ivs, mask_cursor)) {
                        total_masked++;
                        continue;
                    }

                    // Check for N in the (k+1)-mer window
                    bool has_n = false;
                    for (int w = 0; w < window_size; ++w) {
                        char c = seq_buf[rel + w];
                        if (c != 'A' && c != 'C' && c != 'G' && c != 'T' &&
                            c != 'a' && c != 'c' && c != 'g' && c != 't') {
                            has_n = true;
                            break;
                        }
                    }
                    if (has_n) {
                        total_n++;
                        continue;
                    }

                    // Determine bin for this position
                    while (bin_cursor + 1 < bins.size() &&
                           pos >= bins[bin_cursor + 1].first) {
                        ++bin_cursor;
                    }

                    int bin_idx = -1;
                    if (!bins.empty() &&
                        pos >= bins[bin_cursor].first &&
                        pos < bins[bin_cursor].first + iter_size) {
                        bin_idx = bins[bin_cursor].second;
                    }

                    if (bin_idx < 0 || bin_idx >= num_bins) continue;

                    // Encode forward k-mer context and next base
                    int context_idx = StratifiedMarkovModel::encode_kmer(&seq_buf[rel], k);
                    int next_base = StratifiedMarkovModel::encode_base(seq_buf[rel + k]);

                    if (context_idx < 0 || next_base < 0) continue;

                    // Count forward strand
                    model.increment_count(bin_idx, context_idx, next_base);

                    // Count reverse complement
                    int rc_context, rc_next;
                    StratifiedMarkovModel::revcomp_kmer(context_idx, next_base, k,
                                                        rc_context, rc_next);
                    model.increment_count(bin_idx, rc_context, rc_next);
                }
            }

            check_interrupt();
        }

        // --- Apply bin mapping ---
        if (has_bin_map && !bin_map_vec.empty()) {
            model.apply_bin_mapping(bin_map_vec);
        }

        // --- Resolve Dirichlet prior pi(b) -----------------------------------
        // 'marginal' uses post-merge counts so it must run after apply_bin_mapping.
        int marginal_fallbacks = 0;
        if (prior_mode == "uniform") {
            model.set_prior_uniform();
        } else if (prior_mode == "marginal") {
            marginal_fallbacks = model.set_prior_from_marginal();
        } else if (prior_mode == "global") {
            model.set_prior_from_global_marginal();
        } else if (prior_mode == "explicit") {
            if (!py_prior_matrix || py_prior_matrix == Py_None) {
                verror("prior_mode='explicit' requires a non-None prior_matrix");
            }
            PMPY arr_prior(PyArray_FROM_OTF(py_prior_matrix, NPY_DOUBLE,
                                             NPY_ARRAY_IN_ARRAY), true);
            if (!arr_prior) {
                PyErr_Clear();
                verror("prior_matrix must be a numeric 2D array");
            }
            PyArrayObject *mat = (PyArrayObject *)(PyObject *)arr_prior;
            if (PyArray_NDIM(mat) != 2 ||
                PyArray_DIM(mat, 0) != num_bins ||
                PyArray_DIM(mat, 1) != NUM_BASES) {
                verror("prior_matrix must be %d x %d (got %d x %d)",
                       num_bins, NUM_BASES,
                       (int)PyArray_DIM(mat, 0), (int)PyArray_DIM(mat, 1));
            }
            double *mat_data = (double *)PyArray_DATA(mat);
            std::vector<std::array<double, NUM_BASES>> pi_rows(num_bins);
            // numpy is row-major by default for new arrays, so [b][a] = mat[b*4 + a]
            for (int b = 0; b < num_bins; ++b) {
                for (int a = 0; a < NUM_BASES; ++a) {
                    pi_rows[b][a] = mat_data[b * NUM_BASES + a];
                }
            }
            model.set_prior_explicit(pi_rows);
        } else {
            verror("Unknown prior_mode: %s", prior_mode.c_str());
        }

        // --- Normalize and build CDF (uses prior set above) ---
        model.normalize_and_build_cdf(pseudocount);

        // --- Build result dict ---
        PMPY result(PyDict_New(), true);
        if (!result) verror("Failed to create result dict");

        // counts: list of 2D numpy arrays (num_bins x [num_kmers, 4])
        PMPY py_counts_list(PyList_New(num_bins), true);
        PMPY py_cdf_list(PyList_New(num_bins), true);

        for (int b = 0; b < num_bins; ++b) {
            // Counts array: num_kmers x 4 (uint64)
            npy_intp count_dims[2] = {num_kmers, NUM_BASES};
            PMPY py_count_mat(PyArray_SimpleNew(2, count_dims, NPY_UINT64), true);
            uint64_t *count_data = (uint64_t *)PyArray_DATA(
                (PyArrayObject *)(PyObject *)py_count_mat);
            for (int ctx = 0; ctx < num_kmers; ++ctx) {
                for (int base = 0; base < NUM_BASES; ++base) {
                    count_data[ctx * NUM_BASES + base] = model.get_count(b, ctx, base);
                }
            }
            py_count_mat.to_be_stolen();
            PyList_SET_ITEM((PyObject *)py_counts_list, b, (PyObject *)py_count_mat);

            // CDF array: num_kmers x 4 (float64)
            npy_intp cdf_dims[2] = {num_kmers, NUM_BASES};
            PMPY py_cdf_mat(PyArray_SimpleNew(2, cdf_dims, NPY_DOUBLE), true);
            double *cdf_data = (double *)PyArray_DATA(
                (PyArrayObject *)(PyObject *)py_cdf_mat);
            for (int ctx = 0; ctx < num_kmers; ++ctx) {
                for (int base = 0; base < NUM_BASES; ++base) {
                    cdf_data[ctx * NUM_BASES + base] = model.get_cdf(b, ctx, base);
                }
            }
            py_cdf_mat.to_be_stolen();
            PyList_SET_ITEM((PyObject *)py_cdf_list, b, (PyObject *)py_cdf_mat);
        }

        // per_bin_kmers: numpy int64 array
        npy_intp pbk_dims[1] = {num_bins};
        PMPY py_pbk(PyArray_SimpleNew(1, pbk_dims, NPY_UINT64), true);
        uint64_t *pbk_data = (uint64_t *)PyArray_DATA(
            (PyArrayObject *)(PyObject *)py_pbk);
        for (int b = 0; b < num_bins; ++b) {
            pbk_data[b] = model.get_bin_kmers(b);
        }

        // total_kmers
        PMPY py_total(PyLong_FromUnsignedLongLong(model.get_total_kmers()), true);
        PMPY py_masked(PyLong_FromUnsignedLongLong(total_masked), true);
        PMPY py_n(PyLong_FromUnsignedLongLong(total_n), true);
        PMPY py_k(PyLong_FromLong(k), true);
        PMPY py_num_kmers(PyLong_FromLong(num_kmers), true);

        // Resolved per-bin prior matrix (n_bins x 4, row-major for numpy).
        npy_intp prior_dims[2] = {num_bins, NUM_BASES};
        PMPY py_prior(PyArray_SimpleNew(2, prior_dims, NPY_DOUBLE), true);
        {
            const auto& prior_vec = model.get_prior();
            double *prior_data = (double *)PyArray_DATA(
                (PyArrayObject *)(PyObject *)py_prior);
            for (int b = 0; b < num_bins; ++b) {
                for (int a = 0; a < NUM_BASES; ++a) {
                    prior_data[b * NUM_BASES + a] = prior_vec[b][a];
                }
            }
        }
        PMPY py_marginal_fallbacks(PyLong_FromLong(marginal_fallbacks), true);

        py_counts_list.to_be_stolen();
        py_cdf_list.to_be_stolen();
        py_pbk.to_be_stolen();
        py_total.to_be_stolen();
        py_masked.to_be_stolen();
        py_n.to_be_stolen();
        py_k.to_be_stolen();
        py_num_kmers.to_be_stolen();
        py_prior.to_be_stolen();
        py_marginal_fallbacks.to_be_stolen();

        PyDict_SetItemString(result, "counts", py_counts_list);
        PyDict_SetItemString(result, "cdf", py_cdf_list);
        PyDict_SetItemString(result, "per_bin_kmers", py_pbk);
        PyDict_SetItemString(result, "total_kmers", py_total);
        PyDict_SetItemString(result, "total_masked", py_masked);
        PyDict_SetItemString(result, "total_n", py_n);
        PyDict_SetItemString(result, "k", py_k);
        PyDict_SetItemString(result, "num_kmers", py_num_kmers);
        PyDict_SetItemString(result, "prior", py_prior);
        PyDict_SetItemString(result, "marginal_fallbacks",
                             py_marginal_fallbacks);

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    } catch (const std::exception &e) {
        PyMisha::handle_error(e.what());
        return NULL;
    }
}

// ============================================================================
// pm_gsynth_sample
// ============================================================================

/*
 * Sample one or more synthetic genome sequences from a trained model.
 *
 * Python args:
 *   cdf_list       - list of 2D numpy arrays (one per bin), each num_kmers x 4
 *   breaks         - list of float, bin boundaries
 *   bin_indices    - numpy int32 array, flat bin index per iterator position
 *   iter_starts    - numpy int64 array, start of each iter position
 *   iter_chroms    - numpy int32 array, chromid of each iter position
 *   intervals      - DataFrame, intervals to synthesize
 *   mask_copy      - DataFrame or None, intervals where original seq is copied
 *   output_path    - str, output file path (ignored if format="vector")
 *   output_format  - int: 0=seq, 1=fasta, 2=vector
 *   n_samples      - int, number of samples per interval
 *   seed           - int or None, random seed
 *   k              - int, Markov order (1..10, default 5)
 *
 * Returns: list of strings (format=2) or None (format=0,1)
 */
PyObject *pm_gsynth_sample(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_cdf_list = NULL;
        PyObject *py_breaks = NULL;
        PyObject *py_bin_indices = NULL;
        PyObject *py_iter_starts = NULL;
        PyObject *py_iter_chroms = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_mask_copy = NULL;
        const char *output_path = "";
        int output_format = 2;
        int n_samples = 1;
        PyObject *py_seed = NULL;
        int k = 5;
        long long iter_size_arg = 0;
        int preserve_n_arg = 1;

        if (!PyArg_ParseTuple(args, "OOOOOOOsiiOiLp",
                              &py_cdf_list, &py_breaks,
                              &py_bin_indices, &py_iter_starts, &py_iter_chroms,
                              &py_intervals, &py_mask_copy,
                              &output_path, &output_format, &n_samples,
                              &py_seed, &k, &iter_size_arg, &preserve_n_arg)) {
            verror("Invalid arguments to pm_gsynth_sample");
        }
        bool preserve_n = (preserve_n_arg != 0);

        // Validate k
        if (k < 1 || k > StratifiedMarkovModel::MAX_K) {
            verror("Markov order k must be in [1, %d], got %d",
                   StratifiedMarkovModel::MAX_K, k);
        }

        // Validate iterator: caller must pass the iterator value used during
        // bin extraction (see pm_gsynth_train for rationale).
        if (iter_size_arg <= 0) {
            verror("iterator must be a positive integer, got %lld", iter_size_arg);
        }
        int64_t iter_size = (int64_t)iter_size_arg;

        int num_kmers = 1 << (2 * k);  // 4^k

        // Set random seed if provided
        if (py_seed && py_seed != Py_None) {
            long seed = PyLong_AsLong(py_seed);
            if (!PyErr_Occurred()) {
                srand48(seed);
            }
            PyErr_Clear();
        }

        // --- Parse breaks ---
        if (!PyList_Check(py_breaks)) {
            verror("breaks must be a list");
        }
        Py_ssize_t num_breaks = PyList_Size(py_breaks);
        int num_bins = (int)(num_breaks - 1);
        if (num_bins <= 0) {
            verror("breaks must have at least 2 elements");
        }

        // 0D models have a single bin spanning the whole genome. The Python
        // side then passes iter_size = INT64_MAX as a "no constraint"
        // sentinel; the per-position bin-bounds check below would overflow
        // (bins[c].first + INT64_MAX wraps to a negative number) and
        // misroute unaligned-interval samples through the uniform-random
        // fallback. Treat num_bins == 1 as a fast path that skips the
        // bounds check entirely (R 5.6.x parity follow-up, roadmap #1).
        const bool one_bin = (num_bins == 1);

        // --- Build CDF data from Python list ---
        // Use flat vector layout: cdf_data[bin][kmer_idx * NUM_BASES + base_idx]
        std::vector<std::vector<float>> cdf_data(num_bins);
        if (!PyList_Check(py_cdf_list) || PyList_Size(py_cdf_list) != num_bins) {
            verror("cdf_list must be a list with %d elements", num_bins);
        }
        for (int b = 0; b < num_bins; ++b) {
            PyObject *py_mat = PyList_GetItem(py_cdf_list, b);
            PMPY arr(PyArray_FROM_OTF(py_mat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY), true);
            if (!arr) {
                PyErr_Clear();
                verror("CDF matrix %d is not a valid numeric array", b);
            }
            PyArrayObject *mat = (PyArrayObject *)(PyObject *)arr;
            if (PyArray_NDIM(mat) != 2 ||
                PyArray_DIM(mat, 0) != num_kmers ||
                PyArray_DIM(mat, 1) != NUM_BASES) {
                verror("CDF matrix %d must be %d x %d", b, num_kmers, NUM_BASES);
            }
            double *data = (double *)PyArray_DATA(mat);
            cdf_data[b].resize(num_kmers * NUM_BASES);
            for (int i = 0; i < num_kmers * NUM_BASES; ++i) {
                cdf_data[b][i] = static_cast<float>(data[i]);
            }
        }

        // --- Parse bin indices ---
        PMPY arr_bins(PyArray_FROM_OTF(py_bin_indices, NPY_INT32,
                                        NPY_ARRAY_IN_ARRAY), true);
        if (!arr_bins) {
            PyErr_Clear();
            verror("bin_indices must be convertible to int32 array");
        }
        int num_iter_positions = (int)PyArray_SIZE((PyArrayObject *)(PyObject *)arr_bins);
        int32_t *bin_indices = (int32_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_bins);

        // --- Parse iter_starts ---
        PMPY arr_starts(PyArray_FROM_OTF(py_iter_starts, NPY_INT64,
                                          NPY_ARRAY_IN_ARRAY), true);
        if (!arr_starts) {
            PyErr_Clear();
            verror("iter_starts must be convertible to int64 array");
        }
        int64_t *iter_starts = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_starts);

        // --- Parse iter_chroms ---
        PMPY arr_chroms(PyArray_FROM_OTF(py_iter_chroms, NPY_INT32,
                                          NPY_ARRAY_IN_ARRAY), true);
        if (!arr_chroms) {
            PyErr_Clear();
            verror("iter_chroms must be convertible to int32 array");
        }
        int32_t *iter_chroms = (int32_t *)PyArray_DATA((PyArrayObject *)(PyObject *)arr_chroms);

        // --- Parse intervals ---
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        int num_chroms = (int)chromkey.get_num_chroms();

        std::vector<std::vector<GInterval>> sample_per_chrom(num_chroms);
        {
            std::vector<GInterval> all_intervals;
            convert_py_intervals(py_intervals, all_intervals);
            for (auto &iv : all_intervals) {
                if (iv.chromid >= 0 && iv.chromid < num_chroms) {
                    sample_per_chrom[iv.chromid].push_back(iv);
                }
            }
        }

        // --- Parse mask_copy intervals ---
        std::vector<std::vector<GInterval>> mask_copy_per_chrom(num_chroms);
        if (py_mask_copy && py_mask_copy != Py_None) {
            std::vector<GInterval> mask_intervals;
            convert_py_intervals(py_mask_copy, mask_intervals);
            std::sort(mask_intervals.begin(), mask_intervals.end(),
                      [](const GInterval &a, const GInterval &b) {
                          return a.chromid < b.chromid ||
                                 (a.chromid == b.chromid && a.start < b.start);
                      });
            for (const auto &iv : mask_intervals) {
                if (iv.chromid >= 0 && iv.chromid < num_chroms) {
                    mask_copy_per_chrom[iv.chromid].push_back(iv);
                }
            }
        }

        if (n_samples < 1) n_samples = 1;

        // --- Build per-chromosome bin lookup ---
        std::vector<std::vector<std::pair<int64_t, int>>> chrom_bins(num_chroms);
        for (int i = 0; i < num_iter_positions; ++i) {
            int chromid = iter_chroms[i];
            if (chromid >= 0 && chromid < num_chroms) {
                chrom_bins[chromid].push_back({iter_starts[i], bin_indices[i]});
            }
        }
        for (int c = 0; c < num_chroms; ++c) {
            std::sort(chrom_bins[c].begin(), chrom_bins[c].end());
        }

        // --- Set up sequence fetcher (for mask_copy) ---
        GenomeSeqFetch seqfetch;
        seqfetch.set_seqdir(g_pmdb->groot() + "/seq");

        // --- Open output ---
        std::ofstream fasta_ofs;
        BufferedFile seq_bfile;
        if (output_format == 1) {
            fasta_ofs.open(output_path);
            if (!fasta_ofs) verror("Failed to open output file: %s", output_path);
        } else if (output_format == 0) {
            seq_bfile.open(output_path, "wb");
            if (seq_bfile.error()) verror("Failed to open output file: %s", output_path);
        }

        std::vector<std::string> collected_seqs;
        std::vector<FaiEntry> fai_entries;

        // --- Sample per chromosome ---
        for (int chromid = 0; chromid < num_chroms; ++chromid) {
            const auto &sample_ivs = sample_per_chrom[chromid];
            if (sample_ivs.empty()) continue;

            int64_t chrom_size = chromkey.get_chrom_size(chromid);
            if (chrom_size <= 0) continue;

            const std::string &chrom_name = chromkey.id2chrom(chromid);
            const auto &mask_copy_ivs = mask_copy_per_chrom[chromid];
            const auto &bins = chrom_bins[chromid];

            for (size_t iv_idx = 0; iv_idx < sample_ivs.size(); ++iv_idx) {
                const GInterval &iv = sample_ivs[iv_idx];
                int64_t interval_start = std::max<int64_t>(0, iv.start);
                int64_t interval_end = std::min<int64_t>(chrom_size, iv.end);
                if (interval_end <= interval_start) continue;

                int64_t interval_len = interval_end - interval_start;

                // Load original sequence whenever we need to consult it:
                // for mask_copy regions, or for preserve_n's per-position
                // N check.
                std::vector<char> original_seq;
                if (!mask_copy_ivs.empty() || preserve_n) {
                    GInterval read_iv(chromid, interval_start, interval_end, 0);
                    seqfetch.read_interval(read_iv, chromkey, original_seq);
                }

                for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
                    // Bin queries use pos - k (the leftmost base of the
                    // (k+1)-mer context), matching the convention used by
                    // training. The first sampled position is
                    // interval_start + k, so the first query is at
                    // interval_start.
                    size_t bin_cursor = 0;
                    int64_t first_query = interval_start;
                    if (!bins.empty()) {
                        while (bin_cursor + 1 < bins.size() &&
                               first_query >= bins[bin_cursor + 1].first) {
                            ++bin_cursor;
                        }
                    }

                    std::vector<char> synth_seq(interval_len);

                    size_t mask_cursor = 0;
                    while (mask_cursor < mask_copy_ivs.size() &&
                           mask_copy_ivs[mask_cursor].end <= interval_start) {
                        ++mask_cursor;
                    }

                    // Initialize first k bases
                    int64_t init_len = std::min<int64_t>(k, interval_len);
                    for (int64_t i = 0; i < init_len; ++i) {
                        int64_t pos = interval_start + i;
                        if (is_position_masked(pos, mask_copy_ivs, mask_cursor) &&
                            i < (int64_t)original_seq.size()) {
                            synth_seq[i] = original_seq[i];
                            continue;
                        }
                        // preserve_n: keep N (or n) from the reference rather
                        // than fabricating an ACGT base at a gap position.
                        if (preserve_n && i < (int64_t)original_seq.size()) {
                            char orig = original_seq[i];
                            if (orig == 'N' || orig == 'n') {
                                synth_seq[i] = orig;
                                continue;
                            }
                        }
                        synth_seq[i] = StratifiedMarkovModel::decode_base(
                            static_cast<int>(drand48() * NUM_BASES));
                    }

                    // Sample remaining using Markov chain
                    for (int64_t pos = interval_start + init_len; pos < interval_end; ++pos) {
                        int64_t rel_pos = pos - interval_start;

                        // Check mask_copy
                        if (is_position_masked(pos, mask_copy_ivs, mask_cursor)) {
                            if (rel_pos < (int64_t)original_seq.size()) {
                                synth_seq[rel_pos] = original_seq[rel_pos];
                            } else {
                                synth_seq[rel_pos] = StratifiedMarkovModel::decode_base(
                                    static_cast<int>(drand48() * NUM_BASES));
                            }
                            continue;
                        }

                        // preserve_n: keep N (or n) from the reference rather
                        // than fabricating an ACGT base at a gap position.
                        if (preserve_n && rel_pos < (int64_t)original_seq.size()) {
                            char orig = original_seq[rel_pos];
                            if (orig == 'N' || orig == 'n') {
                                synth_seq[rel_pos] = orig;
                                continue;
                            }
                        }

                        // Find bin for this position. Use pos - k (the
                        // context-leftmost base) to match training, which
                        // attributes each (k+1)-mer event to bin_at(pos - k).
                        int64_t bin_query_pos = pos - k;
                        int bin_idx = -1;
                        if (one_bin) {
                            // 0D model: single bin spans everything. Skip
                            // the bounds check that would overflow when
                            // iter_size == INT64_MAX.
                            if (!bins.empty()) bin_idx = bins[0].second;
                        } else if (!bins.empty()) {
                            while (bin_cursor + 1 < bins.size() &&
                                   bin_query_pos >= bins[bin_cursor + 1].first) {
                                ++bin_cursor;
                            }
                            int64_t bin_first = bins[bin_cursor].first;
                            // Saturate: bin_first + iter_size can overflow
                            // when callers pass the INT64_MAX no-constraint
                            // sentinel on a > 0 bin start.
                            int64_t bin_end_excl =
                                (iter_size > std::numeric_limits<int64_t>::max() - bin_first)
                                    ? std::numeric_limits<int64_t>::max()
                                    : bin_first + iter_size;
                            if (bin_query_pos >= bin_first &&
                                bin_query_pos < bin_end_excl) {
                                bin_idx = bins[bin_cursor].second;
                            }
                        }

                        // Get k-mer context
                        int context_idx = StratifiedMarkovModel::encode_kmer(
                            &synth_seq[rel_pos - k], k);

                        int next_base;
                        if (context_idx < 0 || bin_idx < 0 || bin_idx >= num_bins) {
                            next_base = static_cast<int>(drand48() * NUM_BASES);
                        } else {
                            float r = static_cast<float>(drand48());
                            int base_offset = context_idx * NUM_BASES;
                            next_base = NUM_BASES - 1;
                            for (int b = 0; b < NUM_BASES; ++b) {
                                if (r < cdf_data[bin_idx][base_offset + b]) {
                                    next_base = b;
                                    break;
                                }
                            }
                        }

                        synth_seq[rel_pos] =
                            StratifiedMarkovModel::decode_base(next_base);
                    }

                    // Write output
                    if (output_format == 2) {
                        collected_seqs.push_back(
                            std::string(synth_seq.begin(), synth_seq.end()));
                    } else if (output_format == 1) {
                        std::string header = chrom_name;
                        if (!(interval_start == 0 && interval_end == chrom_size)) {
                            header = chrom_name + ":" +
                                     std::to_string(interval_start) + "-" +
                                     std::to_string(interval_end);
                        }
                        if (n_samples > 1) {
                            header += "_sample" + std::to_string(sample_idx + 1);
                        }
                        long long header_offset = static_cast<long long>(fasta_ofs.tellp());
                        FaiEntry entry;
                        write_fasta_record(fasta_ofs, header, synth_seq, 60,
                                           header_offset, &entry);
                        fai_entries.push_back(entry);
                    } else {
                        seq_bfile.write(&synth_seq[0], synth_seq.size());
                    }
                }
            }
            check_interrupt();
        }

        // Close files
        if (output_format == 1) fasta_ofs.close();
        else if (output_format == 0) seq_bfile.close();

        // Write samtools-compatible .fai index alongside the FASTA.
        if (output_format == 1) {
            flush_fai(output_path, fai_entries);
            if (PyErr_Occurred()) return NULL;
        }

        // Return result
        if (output_format == 2) {
            PMPY result(PyList_New(collected_seqs.size()), true);
            for (size_t i = 0; i < collected_seqs.size(); ++i) {
                PyObject *py_str = PyUnicode_FromStringAndSize(
                    collected_seqs[i].c_str(), collected_seqs[i].size());
                PyList_SET_ITEM((PyObject *)result, i, py_str);
            }
            result.to_be_stolen();
            return (PyObject *)result;
        }

        Py_RETURN_NONE;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    } catch (const std::exception &e) {
        PyMisha::handle_error(e.what());
        return NULL;
    }
}

// ============================================================================
// pm_gsynth_replace_kmer
// ============================================================================

/*
 * Iteratively replace a k-mer in genome sequences.
 *
 * Python args:
 *   target         - str, target k-mer
 *   replacement    - str, replacement sequence
 *   intervals      - DataFrame, intervals to process
 *   output_path    - str, output file path
 *   output_format  - int: 0=seq, 1=fasta, 2=vector
 *
 * Returns: list of strings (format=2) or None
 */
PyObject *pm_gsynth_replace_kmer(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *target_str = NULL;
        const char *replacement_str = NULL;
        PyObject *py_intervals = NULL;
        const char *output_path = "";
        int output_format = 2;

        if (!PyArg_ParseTuple(args, "ssOsi",
                              &target_str, &replacement_str,
                              &py_intervals, &output_path, &output_format)) {
            verror("Invalid arguments to pm_gsynth_replace_kmer");
        }

        std::string target(target_str);
        std::string replacement(replacement_str);

        if (target.empty() || replacement.empty()) {
            verror("target and replacement cannot be empty");
        }
        if (target.length() != replacement.length()) {
            verror("target and replacement must have the same length");
        }

        // Convert to uppercase
        for (auto &c : target) c = toupper(c);
        for (auto &c : replacement) c = toupper(c);

        // --- Parse intervals ---
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        int num_chroms = (int)chromkey.get_num_chroms();

        std::vector<std::vector<GInterval>> intervals_per_chrom(num_chroms);
        {
            std::vector<GInterval> all_intervals;
            convert_py_intervals(py_intervals, all_intervals);
            for (auto &iv : all_intervals) {
                if (iv.chromid >= 0 && iv.chromid < num_chroms) {
                    intervals_per_chrom[iv.chromid].push_back(iv);
                }
            }
        }

        // --- Set up sequence fetcher ---
        GenomeSeqFetch seqfetch;
        seqfetch.set_seqdir(g_pmdb->groot() + "/seq");

        // --- Open output ---
        std::ofstream fasta_ofs;
        BufferedFile seq_bfile;
        if (output_format == 1) {
            fasta_ofs.open(output_path);
            if (!fasta_ofs) verror("Failed to open output file: %s", output_path);
        } else if (output_format == 0) {
            seq_bfile.open(output_path, "wb");
            if (seq_bfile.error()) verror("Failed to open output file: %s", output_path);
        }

        std::vector<std::string> collected_seqs;
        size_t kmer_len = target.length();

        // --- Process intervals ---
        for (int chromid = 0; chromid < num_chroms; ++chromid) {
            const auto &intervals = intervals_per_chrom[chromid];
            if (intervals.empty()) continue;

            int64_t chrom_size = chromkey.get_chrom_size(chromid);
            if (chrom_size <= 0) continue;
            const std::string &chrom_name = chromkey.id2chrom(chromid);

            for (const auto &iv : intervals) {
                int64_t interval_start = std::max<int64_t>(0, iv.start);
                int64_t interval_end = std::min<int64_t>(chrom_size, iv.end);
                if (interval_end <= interval_start) continue;

                // Read original sequence
                std::vector<char> seq;
                GInterval read_iv(chromid, interval_start, interval_end, 0);
                seqfetch.read_interval(read_iv, chromkey, seq);

                // Convert to uppercase for matching
                for (auto &c : seq) c = toupper(c);

                // Iteratively replace
                bool found_any = true;
                while (found_any) {
                    found_any = false;
                    for (size_t i = 0; i + kmer_len <= seq.size(); ++i) {
                        bool match = true;
                        for (size_t j = 0; j < kmer_len; ++j) {
                            if (seq[i + j] != target[j]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            for (size_t j = 0; j < kmer_len; ++j) {
                                seq[i + j] = replacement[j];
                            }
                            found_any = true;
                        }
                    }
                }

                // Write output
                if (output_format == 2) {
                    collected_seqs.push_back(std::string(seq.begin(), seq.end()));
                } else if (output_format == 1) {
                    std::string header = chrom_name;
                    if (intervals.size() > 1) {
                        header += "_" + std::to_string(interval_start) + "_" +
                                  std::to_string(interval_end);
                    }
                    fasta_ofs << ">" << header << "\n";
                    for (size_t i = 0; i < seq.size(); i += 60) {
                        size_t len = std::min<size_t>(60, seq.size() - i);
                        fasta_ofs.write(&seq[i], len);
                        fasta_ofs << "\n";
                    }
                } else {
                    seq_bfile.write(&seq[0], seq.size());
                }
            }
            check_interrupt();
        }

        // Close files
        if (output_format == 1) fasta_ofs.close();
        else if (output_format == 0) seq_bfile.close();

        // Return result
        if (output_format == 2) {
            PMPY result(PyList_New(collected_seqs.size()), true);
            for (size_t i = 0; i < collected_seqs.size(); ++i) {
                PyObject *py_str = PyUnicode_FromStringAndSize(
                    collected_seqs[i].c_str(), collected_seqs[i].size());
                PyList_SET_ITEM((PyObject *)result, i, py_str);
            }
            result.to_be_stolen();
            return (PyObject *)result;
        }

        Py_RETURN_NONE;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    } catch (const std::exception &e) {
        PyMisha::handle_error(e.what());
        return NULL;
    }
}
