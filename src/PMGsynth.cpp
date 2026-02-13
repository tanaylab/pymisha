/*
 * PMGsynth.cpp
 *
 * Python C API wrappers for genome synthesis functions:
 *   pm_gsynth_train       - Train stratified Markov-5 model
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
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

// Defined in PMStubs.cpp
extern void convert_py_intervals(PyObject *py_intervals,
                                 std::vector<GInterval> &intervals);

// ============================================================================
// pm_gsynth_train
// ============================================================================

/*
 * Train a stratified Markov-5 model from genome sequence data.
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
 *
 * Returns:
 *   dict with keys: 'counts' (list of 2D numpy arrays), 'cdf' (list of 2D numpy),
 *   'per_bin_kmers' (numpy int64), 'total_kmers' (int), 'total_masked' (int),
 *   'total_n' (int)
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

        if (!PyArg_ParseTuple(args, "OOOOOOOd",
                              &py_intervals, &py_bin_indices,
                              &py_iter_starts, &py_iter_chroms,
                              &py_breaks, &py_bin_map, &py_mask,
                              &pseudocount)) {
            verror("Invalid arguments to pm_gsynth_train");
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

        // --- Compute iterator bin size ---
        int64_t iter_size = 0;
        if (num_iter_positions > 0) {
            for (int i = 1; i < num_iter_positions; ++i) {
                if (iter_chroms[i] == iter_chroms[i - 1]) {
                    iter_size = iter_starts[i] - iter_starts[i - 1];
                    break;
                }
            }
            // If only one position per chrom, cover the entire chromosome
            if (iter_size <= 0) iter_size = INT64_MAX;
        }

        // --- Initialize model ---
        StratifiedMarkovModel model;
        model.init(num_bins, breaks_vec);

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
                for (int64_t pos = interval_start; pos + 5 < interval_end; ++pos) {
                    int64_t rel = pos - interval_start;

                    // Check mask
                    if (is_position_masked(pos, mask_ivs, mask_cursor)) {
                        total_masked++;
                        continue;
                    }

                    // Check for N in the 6-mer window
                    bool has_n = false;
                    for (int k = 0; k < 6; ++k) {
                        char c = seq_buf[rel + k];
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

                    // Encode forward 5-mer context and next base
                    int context_idx = StratifiedMarkovModel::encode_5mer(&seq_buf[rel]);
                    int next_base = StratifiedMarkovModel::encode_base(seq_buf[rel + 5]);

                    if (context_idx < 0 || next_base < 0) continue;

                    // Count forward strand
                    model.increment_count(bin_idx, context_idx, next_base);

                    // Count reverse complement
                    int rc_context, rc_next;
                    StratifiedMarkovModel::revcomp_6mer(context_idx, next_base,
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

        // --- Normalize and build CDF ---
        model.normalize_and_build_cdf(pseudocount);

        // --- Build result dict ---
        PMPY result(PyDict_New(), true);
        if (!result) verror("Failed to create result dict");

        // counts: list of 2D numpy arrays (num_bins x [1024, 4])
        PMPY py_counts_list(PyList_New(num_bins), true);
        PMPY py_cdf_list(PyList_New(num_bins), true);

        for (int b = 0; b < num_bins; ++b) {
            // Counts array: 1024 x 4 (uint64)
            npy_intp count_dims[2] = {NUM_5MERS, NUM_BASES};
            PMPY py_count_mat(PyArray_SimpleNew(2, count_dims, NPY_UINT64), true);
            uint64_t *count_data = (uint64_t *)PyArray_DATA(
                (PyArrayObject *)(PyObject *)py_count_mat);
            for (int ctx = 0; ctx < NUM_5MERS; ++ctx) {
                for (int base = 0; base < NUM_BASES; ++base) {
                    count_data[ctx * NUM_BASES + base] = model.get_count(b, ctx, base);
                }
            }
            py_count_mat.to_be_stolen();
            PyList_SET_ITEM((PyObject *)py_counts_list, b, (PyObject *)py_count_mat);

            // CDF array: 1024 x 4 (float64)
            npy_intp cdf_dims[2] = {NUM_5MERS, NUM_BASES};
            PMPY py_cdf_mat(PyArray_SimpleNew(2, cdf_dims, NPY_DOUBLE), true);
            double *cdf_data = (double *)PyArray_DATA(
                (PyArrayObject *)(PyObject *)py_cdf_mat);
            for (int ctx = 0; ctx < NUM_5MERS; ++ctx) {
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

        py_counts_list.to_be_stolen();
        py_cdf_list.to_be_stolen();
        py_pbk.to_be_stolen();
        py_total.to_be_stolen();
        py_masked.to_be_stolen();
        py_n.to_be_stolen();

        PyDict_SetItemString(result, "counts", py_counts_list);
        PyDict_SetItemString(result, "cdf", py_cdf_list);
        PyDict_SetItemString(result, "per_bin_kmers", py_pbk);
        PyDict_SetItemString(result, "total_kmers", py_total);
        PyDict_SetItemString(result, "total_masked", py_masked);
        PyDict_SetItemString(result, "total_n", py_n);

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
 *   cdf_list       - list of 2D numpy arrays (one per bin), each 1024 x 4
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

        if (!PyArg_ParseTuple(args, "OOOOOOOsiiO",
                              &py_cdf_list, &py_breaks,
                              &py_bin_indices, &py_iter_starts, &py_iter_chroms,
                              &py_intervals, &py_mask_copy,
                              &output_path, &output_format, &n_samples,
                              &py_seed)) {
            verror("Invalid arguments to pm_gsynth_sample");
        }

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

        // --- Build CDF data from Python list ---
        std::vector<std::vector<std::array<float, NUM_BASES>>> cdf_data(num_bins);
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
                PyArray_DIM(mat, 0) != NUM_5MERS ||
                PyArray_DIM(mat, 1) != NUM_BASES) {
                verror("CDF matrix %d must be %d x %d", b, NUM_5MERS, NUM_BASES);
            }
            double *data = (double *)PyArray_DATA(mat);
            cdf_data[b].resize(NUM_5MERS);
            for (int ctx = 0; ctx < NUM_5MERS; ++ctx) {
                for (int base = 0; base < NUM_BASES; ++base) {
                    cdf_data[b][ctx][base] = static_cast<float>(
                        data[ctx * NUM_BASES + base]);
                }
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

        // --- Compute iterator bin size ---
        int64_t iter_size = 0;
        if (num_iter_positions > 0) {
            for (int i = 1; i < num_iter_positions; ++i) {
                if (iter_chroms[i] == iter_chroms[i - 1]) {
                    iter_size = iter_starts[i] - iter_starts[i - 1];
                    break;
                }
            }
            // If only one position per chrom, cover the entire chromosome
            if (iter_size <= 0) iter_size = INT64_MAX;
        }

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

                // Load original sequence for mask_copy
                std::vector<char> original_seq;
                if (!mask_copy_ivs.empty()) {
                    GInterval read_iv(chromid, interval_start, interval_end, 0);
                    seqfetch.read_interval(read_iv, chromkey, original_seq);
                }

                for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
                    size_t bin_cursor = 0;
                    if (!bins.empty()) {
                        while (bin_cursor + 1 < bins.size() &&
                               interval_start >= bins[bin_cursor + 1].first) {
                            ++bin_cursor;
                        }
                    }

                    std::vector<char> synth_seq(interval_len);

                    size_t mask_cursor = 0;
                    while (mask_cursor < mask_copy_ivs.size() &&
                           mask_copy_ivs[mask_cursor].end <= interval_start) {
                        ++mask_cursor;
                    }

                    // Initialize first 5 bases
                    int64_t init_len = std::min<int64_t>(5, interval_len);
                    for (int64_t i = 0; i < init_len; ++i) {
                        int64_t pos = interval_start + i;
                        if (is_position_masked(pos, mask_copy_ivs, mask_cursor) &&
                            i < (int64_t)original_seq.size()) {
                            synth_seq[i] = original_seq[i];
                        } else {
                            synth_seq[i] = StratifiedMarkovModel::decode_base(
                                static_cast<int>(drand48() * NUM_BASES));
                        }
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

                        // Find bin for this position
                        int bin_idx = -1;
                        if (!bins.empty()) {
                            while (bin_cursor + 1 < bins.size() &&
                                   pos >= bins[bin_cursor + 1].first) {
                                ++bin_cursor;
                            }
                            if (pos >= bins[bin_cursor].first &&
                                pos < bins[bin_cursor].first + iter_size) {
                                bin_idx = bins[bin_cursor].second;
                            }
                        }

                        // Get 5-mer context
                        int context_idx = StratifiedMarkovModel::encode_5mer(
                            &synth_seq[rel_pos - 5]);

                        int next_base;
                        if (context_idx < 0 || bin_idx < 0 || bin_idx >= num_bins) {
                            next_base = static_cast<int>(drand48() * NUM_BASES);
                        } else {
                            float r = static_cast<float>(drand48());
                            const auto &cdf = cdf_data[bin_idx][context_idx];
                            next_base = NUM_BASES - 1;
                            for (int b = 0; b < NUM_BASES; ++b) {
                                if (r < cdf[b]) {
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
                        fasta_ofs << ">" << header << "\n";
                        for (int64_t i = 0; i < interval_len; i += 60) {
                            int64_t len = std::min<int64_t>(60, interval_len - i);
                            fasta_ofs.write(&synth_seq[i], len);
                            fasta_ofs << "\n";
                        }
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
