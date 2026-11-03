#ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE 199309
    #include <time.h>
    #undef _POSIX_C_SOURCE
#endif

// This must be undefined before "#include <numpy/arrayobject.h>" in the file that calls import_array
#ifdef NO_IMPORT_ARRAY
    #undef NO_IMPORT_ARRAY
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
    #define PY_ARRAY_UNIQUE_SYMBOL pymisha_ARRAY_API
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <stdlib.h>
#include <string>
#include "BufferedFile.h"
#include "PMDb.h"

PyObject *g_module;
PyObject *s_pm_err;

// Every entry point runs through gcall(), as every misha .Call runs through .gcall() in R. It
// raises a file the call failed to finish writing (see BufferedFile::write_error): the last
// buffer is written when C++ closes the file, often in a destructor that runs after the entry
// point has chosen its return value. Taken on every exit, so a failed call does not leave it
// behind for the next one; raised only when the call had otherwise succeeded, so that a writer
// staged by the caller (_atomic_track_create) is discarded. The outer call's error is put back,
// so a call nested in a writer (a Python function in a track expression) neither raises nor
// drops it.
template <PyCFunction F>
static PyObject *gcall(PyObject *self, PyObject *args)
{
    const std::string outer = BufferedFile::write_error();
    BufferedFile::set_write_error("");

    PyObject *res = F(self, args);

    const std::string mine = BufferedFile::write_error();
    BufferedFile::set_write_error(outer.c_str());
    if (res && !mine.empty()) {
        Py_DECREF(res);
        PyErr_SetString(s_pm_err, mine.c_str());
        return NULL;
    }
    return res;
}

static void pymisha_module_free(void *m)
{
    if (g_pmdb) {
        delete g_pmdb;
        g_pmdb = nullptr;
    }
    s_pm_err = nullptr;
}

// Forward declarations of functions to be implemented
PyObject *pm_dbinit(PyObject *self, PyObject *args);
PyObject *pm_gdb_chrom_order(PyObject *self, PyObject *args);
PyObject *pm_gdb_check_genome_idx(PyObject *self, PyObject *args);
PyObject *pm_chrom_files(PyObject *self, PyObject *args);
PyObject *pm_chrom_ids(PyObject *self, PyObject *args);
PyObject *pm_dbreload(PyObject *self, PyObject *args);
PyObject *pm_dbunload(PyObject *self, PyObject *args);
PyObject *pm_dbsetdatasets(PyObject *self, PyObject *args);
PyObject *pm_dbgetdatasets(PyObject *self, PyObject *args);
PyObject *pm_dbsetgwd(PyObject *self, PyObject *args);
PyObject *pm_gwd_is_subdir(PyObject *self, PyObject *args);
PyObject *pm_new_track_path(PyObject *self, PyObject *args);
PyObject *pm_extract(PyObject *self, PyObject *args);
PyObject *pm_screen(PyObject *self, PyObject *args);
PyObject *pm_summary(PyObject *self, PyObject *args);
PyObject *pm_quantiles(PyObject *self, PyObject *args);
PyObject *pm_intervals_summary(PyObject *self, PyObject *args);
PyObject *pm_intervals_quantiles(PyObject *self, PyObject *args);
PyObject *pm_track_names(PyObject *self, PyObject *args);
PyObject *pm_interv_names(PyObject *self, PyObject *args);
PyObject *pm_interv_register(PyObject *self, PyObject *args);
PyObject *pm_interv_unregister(PyObject *self, PyObject *args);
PyObject *pm_track_info(PyObject *self, PyObject *args);
PyObject *pm_track_path(PyObject *self, PyObject *args);
PyObject *pm_track_dataset(PyObject *self, PyObject *args);
PyObject *pm_normalize_chroms(PyObject *self, PyObject *args);
PyObject *pm_track_convert_to_indexed(PyObject *self, PyObject *args);
PyObject *pm_track_split_indexed_to_per_chrom(PyObject *self, PyObject *args);
PyObject *pm_track_pack_per_chrom_to_indexed(PyObject *self, PyObject *args);
PyObject *pm_track_create_empty_indexed(PyObject *self, PyObject *args);
PyObject *pm_track_create_sparse(PyObject *self, PyObject *args);
PyObject *pm_track_create_dense(PyObject *self, PyObject *args);
PyObject *pm_track_create_expr(PyObject *self, PyObject *args);
PyObject *pm_set_create_dir_override(PyObject *self, PyObject *args);
PyObject *pm_invalidate_dir_cache(PyObject *self, PyObject *args);
PyObject *pm_clear_dir_caches(PyObject *self, PyObject *args);
PyObject *pm_clear_create_dir_override(PyObject *self, PyObject *args);
PyObject *pm_set_create_parallel_writers(PyObject *self, PyObject *args);
PyObject *pm_intervals_all(PyObject *self, PyObject *args);
PyObject *pm_iterate(PyObject *self, PyObject *args);
PyObject *pm_seed(PyObject *self, PyObject *args);
PyObject *pm_test_df(PyObject *self, PyObject *args);
PyObject *pm_read_df(PyObject *self, PyObject *args);
PyObject *pm_vtrack_compute(PyObject *self, PyObject *args);
PyObject *pm_find_neighbors(PyObject *self, PyObject *args);
PyObject *pm_seq_extract(PyObject *self, PyObject *args);
PyObject *pm_partition(PyObject *self, PyObject *args);
PyObject *pm_dist(PyObject *self, PyObject *args);
PyObject *pm_intervals_union(PyObject *self, PyObject *args);
PyObject *pm_intervals_intersect(PyObject *self, PyObject *args);
PyObject *pm_intervals_diff(PyObject *self, PyObject *args);
PyObject *pm_intervals_canonic(PyObject *self, PyObject *args);
PyObject *pm_intervals_covered_bp(PyObject *self, PyObject *args);
PyObject *pm_sample(PyObject *self, PyObject *args);
PyObject *pm_cor(PyObject *self, PyObject *args);
PyObject *pm_lookup(PyObject *self, PyObject *args);
PyObject *pm_segment(PyObject *self, PyObject *args);
PyObject *pm_wilcox(PyObject *self, PyObject *args);
PyObject *pm_modify(PyObject *self, PyObject *args);
PyObject *pm_smooth(PyObject *self, PyObject *args);
PyObject *pm_gsynth_train(PyObject *self, PyObject *args);
PyObject *pm_gsynth_sample(PyObject *self, PyObject *args);
PyObject *pm_gsynth_replace_kmer(PyObject *self, PyObject *args);
PyObject *pm_quadtree_query_stats(PyObject *self, PyObject *args);
PyObject *pm_quadtree_query_objects(PyObject *self, PyObject *args);
PyObject *pm_quadtree_query_stats_batch(PyObject *self, PyObject *args);
PyObject *pm_neighbors_2d(PyObject *self, PyObject *args);
PyObject *pm_track2d_convert_to_indexed(PyObject *self, PyObject *args);
PyObject *pm_track2d_index_info(PyObject *self, PyObject *args);
PyObject *pm_gseq_pwm_edits(PyObject *self, PyObject *args);
PyObject *pm_pwm_score_strings(PyObject *self, PyObject *args);
PyObject *pm_gseq_potts(PyObject *self, PyObject *args);
PyObject *pm_gvariants_reach(PyObject *self, PyObject *args);
PyObject *pm_potts_score_codes_cmp(PyObject *self, PyObject *args);
PyObject *pm_gextract_variants(PyObject *self, PyObject *args);
PyObject *pm_kmer_count_strings(PyObject *self, PyObject *args);
PyObject *pm_crc64_update(PyObject *self, PyObject *args);
PyObject *pm_crc64_finalize(PyObject *self, PyObject *args);
PyObject *pm_compute_strands_autocorr(PyObject *self, PyObject *args);
PyObject *pm_ggenome_implant(PyObject *self, PyObject *args);
PyObject *pm_intervals_random(PyObject *self, PyObject *args);
PyObject *pm_parse_wig_or_bedgraph(PyObject *self, PyObject *args);
PyObject *pm_is_bigwig(PyObject *self, PyObject *args);
PyObject *pm_parse_bigwig(PyObject *self, PyObject *args);
PyObject *pm_parquet_columns(PyObject *self, PyObject *args);
PyObject *pm_parquet_import(PyObject *self, PyObject *args);
PyObject *pm_parquet_write(PyObject *self, PyObject *args);
PyObject *pm_test_2d_iterator(PyObject *self, PyObject *args);
PyObject *pm_test_fixed_rect_iterator(PyObject *self, PyObject *args);
PyObject *pm_test_fixed_rect_scanner(PyObject *self, PyObject *args);
PyObject *pm_test_scanner_reuse(PyObject *self, PyObject *args);
PyObject *pm_test_2d_scanner(PyObject *self, PyObject *args);
PyObject *pm_test_track_rects_iterator(PyObject *self, PyObject *args);
PyObject *pm_test_cartesian_grid_iterator(PyObject *self, PyObject *args);
PyObject *pm_extract_2d(PyObject *self, PyObject *args);
PyObject *pm_extract_2d_objects(PyObject *self, PyObject *args);
PyObject *pm_extract_2d_scanner(PyObject *self, PyObject *args);
PyObject *pm_liftover_aggregate(PyObject *self, PyObject *args);
PyObject *pm_parse_chain_file(PyObject *self, PyObject *args);
PyObject *pm_read_source_track_1d(PyObject *self, PyObject *args);
PyObject *pm_chain_intervals_resolve(PyObject *self, PyObject *args);
PyObject *pm_map_intervals(PyObject *self, PyObject *args);
PyObject *pm_liftover_track(PyObject *self, PyObject *args);
PyObject *pm_liftover_track_2d(PyObject *self, PyObject *args);
PyObject *pm_track_2d_create(PyObject *self, PyObject *args);
PyObject *pm_track_2d_import_files(PyObject *self, PyObject *args);
PyObject *pm_track_2d_import_contacts_files(PyObject *self, PyObject *args);
PyObject *pm_import_mappedseq(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"pm_dbinit", gcall<pm_dbinit>, METH_VARARGS, "Initialize database connection"},
    {"pm_gdb_chrom_order", gcall<pm_gdb_chrom_order>, METH_VARARGS, "Chromosomes of a database by chrom id"},
    {"pm_gdb_check_genome_idx", gcall<pm_gdb_check_genome_idx>, METH_VARARGS, "Check a database's genome.idx against its chromosomes"},
    {"pm_chrom_files", gcall<pm_chrom_files>, METH_VARARGS, "Per-chromosome files of a track or intervals set directory"},
    {"pm_chrom_ids", gcall<pm_chrom_ids>, METH_VARARGS, "Chrom ids of chromosome names"},
    {"pm_dbreload", gcall<pm_dbreload>, METH_VARARGS, "Reload database"},
    {"pm_dbunload", gcall<pm_dbunload>, METH_VARARGS, "Unload database"},
    {"pm_dbsetdatasets", gcall<pm_dbsetdatasets>, METH_VARARGS, "Set loaded dataset roots"},
    {"pm_dbgetdatasets", gcall<pm_dbgetdatasets>, METH_VARARGS, "Get loaded dataset roots"},
    {"pm_dbsetgwd", gcall<pm_dbsetgwd>, METH_VARARGS, "Set the working directory (takes effect at the next pm_dbreload)"},
    {"pm_gwd_is_subdir", gcall<pm_gwd_is_subdir>, METH_VARARGS, "Whether the working directory is a subdirectory"},
    {"pm_new_track_path", gcall<pm_new_track_path>, METH_VARARGS, "Directory a new track of this name is created in"},
    {"pm_extract", gcall<pm_extract>, METH_VARARGS, "Extract track values"},
    {"pm_screen", gcall<pm_screen>, METH_VARARGS, "Screen intervals by expression"},
    {"pm_summary", gcall<pm_summary>, METH_VARARGS, "Summarize expression values"},
    {"pm_quantiles", gcall<pm_quantiles>, METH_VARARGS, "Compute expression quantiles"},
    {"pm_intervals_summary", gcall<pm_intervals_summary>, METH_VARARGS, "Summarize expression values per interval"},
    {"pm_intervals_quantiles", gcall<pm_intervals_quantiles>, METH_VARARGS, "Compute expression quantiles per interval"},
    {"pm_track_names", gcall<pm_track_names>, METH_VARARGS, "Get track names"},
    {"pm_interv_names", gcall<pm_interv_names>, METH_VARARGS, "Get interval-set names"},
    {"pm_interv_register", gcall<pm_interv_register>, METH_VARARGS, "Register an interval-set name in the cache"},
    {"pm_interv_unregister", gcall<pm_interv_unregister>, METH_VARARGS, "Remove an interval-set name from the cache"},
    {"pm_track_info", gcall<pm_track_info>, METH_VARARGS, "Get track information"},
    {"pm_track_path", gcall<pm_track_path>, METH_VARARGS, "Get track path on disk"},
    {"pm_track_dataset", gcall<pm_track_dataset>, METH_VARARGS, "Get track dataset root"},
    {"pm_normalize_chroms", gcall<pm_normalize_chroms>, METH_VARARGS, "Normalize chromosome names"},
    {"pm_track_convert_to_indexed", gcall<pm_track_convert_to_indexed>, METH_VARARGS, "Convert track to indexed format"},
    {"pm_track_split_indexed_to_per_chrom", gcall<pm_track_split_indexed_to_per_chrom>, METH_VARARGS, "Split indexed track back into per-chromosome files"},
    {"pm_track_pack_per_chrom_to_indexed", gcall<pm_track_pack_per_chrom_to_indexed>, METH_VARARGS, "Pack per-chromosome files into indexed format (explicit args)"},
    {"pm_track_create_empty_indexed", gcall<pm_track_create_empty_indexed>, METH_VARARGS, "Create empty indexed track"},
    {"pm_track_create_sparse", gcall<pm_track_create_sparse>, METH_VARARGS, "Create sparse track from intervals+values"},
    {"pm_track_create_dense", gcall<pm_track_create_dense>, METH_VARARGS, "Create dense track from intervals+values"},
    {"pm_track_create_expr", gcall<pm_track_create_expr>, METH_VARARGS, "Create track from expression in streaming mode"},
    {"pm_set_create_dir_override", gcall<pm_set_create_dir_override>, METH_VARARGS, "Set thread-local override path for next pm_track_create_*"},
    {"pm_invalidate_dir_cache", gcall<pm_invalidate_dir_cache>, METH_VARARGS, "Drop the memoised track index for one or more track directories"},
    {"pm_clear_dir_caches", gcall<pm_clear_dir_caches>, METH_VARARGS, "Drop every memoised track index"},
    {"pm_clear_create_dir_override", gcall<pm_clear_create_dir_override>, METH_VARARGS, "Clear thread-local create_dir_override"},
    {"pm_set_create_parallel_writers", gcall<pm_set_create_parallel_writers>, METH_VARARGS, "Set thread-local worker count for empty-chrom file dispatch in pm_track_create_sparse"},
    {"pm_intervals_all", gcall<pm_intervals_all>, METH_VARARGS, "Get all genome intervals"},
    {"pm_iterate", gcall<pm_iterate>, METH_VARARGS, "Iterate intervals with iterator policy"},
    {"pm_seed", gcall<pm_seed>, METH_VARARGS, "Set random seed"},
    {"pm_vtrack_compute", gcall<pm_vtrack_compute>, METH_VARARGS, "Compute virtual track values"},
    {"pm_find_neighbors", gcall<pm_find_neighbors>, METH_VARARGS, "Find nearest neighbor intervals"},
    {"pm_seq_extract", gcall<pm_seq_extract>, METH_VARARGS, "Extract DNA sequences for intervals"},
    {"pm_partition", gcall<pm_partition>, METH_VARARGS, "Partition track values into bins"},
    {"pm_dist", gcall<pm_dist>, METH_VARARGS, "Calculate distribution of track values over bins"},
    {"pm_lookup", gcall<pm_lookup>, METH_VARARGS, "Lookup table transform on binned track values"},
    {"pm_intervals_union", gcall<pm_intervals_union>, METH_VARARGS, "Union of two interval sets"},
    {"pm_intervals_intersect", gcall<pm_intervals_intersect>, METH_VARARGS, "Intersection of two interval sets"},
    {"pm_intervals_diff", gcall<pm_intervals_diff>, METH_VARARGS, "Difference of two interval sets"},
    {"pm_intervals_canonic", gcall<pm_intervals_canonic>, METH_VARARGS, "Canonicalize intervals"},
    {"pm_intervals_covered_bp", gcall<pm_intervals_covered_bp>, METH_VARARGS, "Count total covered basepairs"},
    {"pm_sample", gcall<pm_sample>, METH_VARARGS, "Sample values from track expression"},
    {"pm_cor", gcall<pm_cor>, METH_VARARGS, "Compute correlation between expression pairs"},
    {"pm_segment", gcall<pm_segment>, METH_VARARGS, "Segment track expression using Wilcoxon test"},
    {"pm_wilcox", gcall<pm_wilcox>, METH_VARARGS, "Sliding-window Wilcoxon test on track expression"},
    {"pm_modify", gcall<pm_modify>, METH_VARARGS, "Modify dense track values in-place"},
    {"pm_smooth", gcall<pm_smooth>, METH_VARARGS, "Create smoothed track from expression"},
    {"__pm_test_df", gcall<pm_test_df>, METH_VARARGS, "Test DataFrame conversion"},
    {"__read_df", gcall<pm_read_df>, METH_VARARGS, "Read DataFrame from internal format"},
    {"pm_gsynth_train", gcall<pm_gsynth_train>, METH_VARARGS, "Train stratified Markov-5 model"},
    {"pm_gsynth_sample", gcall<pm_gsynth_sample>, METH_VARARGS, "Sample synthetic genome"},
    {"pm_gsynth_replace_kmer", gcall<pm_gsynth_replace_kmer>, METH_VARARGS, "Replace k-mers iteratively"},
    {"pm_quadtree_query_stats", gcall<pm_quadtree_query_stats>, METH_VARARGS, "Query quad-tree for aggregated stats"},
    {"pm_quadtree_query_objects", gcall<pm_quadtree_query_objects>, METH_VARARGS, "Query quad-tree for matching objects"},
    {"pm_quadtree_query_stats_batch", gcall<pm_quadtree_query_stats_batch>, METH_VARARGS, "Batch query quad-tree stats for N rectangles"},
    {"pm_neighbors_2d", gcall<pm_neighbors_2d>, METH_VARARGS, "2D NN iteration on one chrom-pair via in-memory quadtree (Phase NN)"},
    {"pm_track2d_convert_to_indexed", gcall<pm_track2d_convert_to_indexed>, METH_VARARGS, "Convert 2D track per-pair files to indexed format"},
    {"pm_track2d_index_info", gcall<pm_track2d_index_info>, METH_VARARGS, "Get 2D track index info"},
    {"pm_gseq_pwm_edits", gcall<pm_gseq_pwm_edits>, METH_VARARGS, "Get detailed PWM edit distance information"},
    {"pm_pwm_score_strings", gcall<pm_pwm_score_strings>, METH_VARARGS, "Score sequences with PWM scorer"},
    {"pm_gseq_potts", gcall<pm_gseq_potts>, METH_VARARGS, "Score sequences under a Potts (pairwise energy) model"},
    {"pm_potts_score_codes_cmp", gcall<pm_potts_score_codes_cmp>, METH_VARARGS, "Test only: score code windows with both Potts kernels"},
    {"pm_gvariants_reach", gcall<pm_gvariants_reach>, METH_VARARGS, "Sequence reach of each sequence vtrack in an expression"},
    {"pm_gextract_variants", gcall<pm_gextract_variants>, METH_VARARGS, "Evaluate an expression with one base substituted per row"},
    {"pm_kmer_count_strings", gcall<pm_kmer_count_strings>, METH_VARARGS, "Count k-mer occurrences in sequences"},
    {"pm_crc64_update", gcall<pm_crc64_update>, METH_VARARGS, "CRC64-ECMA incremental update"},
    {"pm_crc64_finalize", gcall<pm_crc64_finalize>, METH_VARARGS, "CRC64-ECMA finalize"},
    {"pm_compute_strands_autocorr", gcall<pm_compute_strands_autocorr>, METH_VARARGS, "Compute strand cross-correlation"},
    {"pm_ggenome_implant", gcall<pm_ggenome_implant>, METH_VARARGS, "Implant donor sequences into reference FASTA (C++ fast path)"},
    {"pm_intervals_random", gcall<pm_intervals_random>, METH_VARARGS, "Generate random non-overlapping genomic intervals (C++ fast path)"},
    {"pm_parse_wig_or_bedgraph", gcall<pm_parse_wig_or_bedgraph>, METH_VARARGS, "Parse a WIG/BedGraph file into chrom/start/end/value arrays (C++ fast path)"},
    {"pm_is_bigwig", gcall<pm_is_bigwig>, METH_VARARGS, "True if a file is a bigWig: its magic number or a .bw/.bigWig name"},
    {"pm_parse_bigwig", gcall<pm_parse_bigwig>, METH_VARARGS, "Read the full-resolution records of a bigWig into chrom/start/end/value arrays"},
    {"pm_parquet_columns", gcall<pm_parquet_columns>, METH_VARARGS, "Names and types of the columns of a Parquet file"},
    {"pm_parquet_import", gcall<pm_parquet_import>, METH_VARARGS, "Import value columns of Parquet files as dense track files"},
    {"pm_parquet_write", gcall<pm_parquet_write>, METH_VARARGS, "Write intervals and values to a Parquet file"},
    {"pm_liftover_aggregate", gcall<pm_liftover_aggregate>, METH_VARARGS, "Aggregate overlapping intervals per chrom (C++ fast path for liftover)"},
    {"pm_parse_chain_file", gcall<pm_parse_chain_file>, METH_VARARGS, "Parse a UCSC chain file into 10 columns (C++ fast path for gintervals_load_chain)"},
    {"pm_read_source_track_1d", gcall<pm_read_source_track_1d>, METH_VARARGS, "Read a 1D source-track directory into a (type, df_dict) tuple (C++ fast path for gtrack_liftover)"},
    {"pm_chain_intervals_resolve", gcall<pm_chain_intervals_resolve>, METH_VARARGS, "Resolve src+tgt overlap policies on a chain DataFrame (C++ fast path for gintervals_load_chain)"},
    {"pm_map_intervals", gcall<pm_map_intervals>, METH_VARARGS, "Map source intervals through a resolved chain (G1.P3.B.2 fast path)"},
    {"pm_liftover_track", gcall<pm_liftover_track>, METH_VARARGS, "Lift + aggregate a 1D source track to the current target DB (G1.P3.C fast path for gtrack_liftover)"},
    {"pm_liftover_track_2d", gcall<pm_liftover_track_2d>, METH_VARARGS, "Lift a 2D source track into the directory of a new track (gtrack_liftover)"},
    {"pm_track_2d_create", gcall<pm_track_2d_create>, METH_VARARGS, "Write the pair files of a 2D track from intervals (gtrack_2d_create)"},
    {"pm_track_2d_import_files", gcall<pm_track_2d_import_files>, METH_VARARGS, "Read 2D intervals files and write the pair files of a 2D track (gtrack_2d_import)"},
    {"pm_track_2d_import_contacts_files", gcall<pm_track_2d_import_contacts_files>, METH_VARARGS, "Read contacts files and write the pair files of a points track (gtrack_2d_import_contacts)"},
    {"pm_test_2d_iterator", gcall<pm_test_2d_iterator>, METH_VARARGS, "Test-only: drive PMTrackExpressionIntervals2DIterator and return emissions"},
    {"pm_test_fixed_rect_iterator", gcall<pm_test_fixed_rect_iterator>, METH_VARARGS, "Test-only: walk a FixedRect 2D iterator. Args: width, height, intervals_dict, band|None."},
    {"pm_test_fixed_rect_scanner", gcall<pm_test_fixed_rect_scanner>, METH_VARARGS, "Test-only: FixedRect iterator + 2D scanner end-to-end. Args: width, height, track, func, intervals_dict, band|None."},
    {"pm_test_scanner_reuse", gcall<pm_test_scanner_reuse>, METH_VARARGS, "Test-only: call run() twice on the same scanner to verify state reset. Args: width, height, track, func, intervals1_dict, intervals2_dict."},
    {"pm_test_2d_scanner", gcall<pm_test_2d_scanner>, METH_VARARGS, "Test-only: drive PMTrackExpr2DScanner over a 2D track + intervals"},
    {"pm_test_track_rects_iterator", gcall<pm_test_track_rects_iterator>, METH_VARARGS, "Test-only: walk a TrackRects 2D iterator. Args: track_name, intervals_dict, band|None."},
    {"pm_test_cartesian_grid_iterator", gcall<pm_test_cartesian_grid_iterator>, METH_VARARGS, "Walk a CartesianGrid 2D iterator. Args: intervals1, expansion1, intervals2|None, expansion2|None, band_idx|None, scope, band|None."},
    {"pm_cartesian_grid_intervals", gcall<pm_test_cartesian_grid_iterator>, METH_VARARGS, "Enumerate the cells of a CartesianGrid 2D iterator over a 2D scope (used by giterator_intervals). Same args as pm_test_cartesian_grid_iterator."},
    {"pm_extract_2d", gcall<pm_extract_2d>, METH_VARARGS, "Extract objects from a 2D RECTS/POINTS track for 2D intervals"},
    {"pm_extract_2d_objects", gcall<pm_extract_2d_objects>, METH_VARARGS, "Reduce a 2D RECTS/POINTS track to a per-interval scalar via exists/size/first/last/sample"},
    {"pm_extract_2d_scanner", gcall<pm_extract_2d_scanner>, METH_VARARGS,
     "Run the 2D scanner with an iterator policy. Args: policy_dict, intervals_dict, vars_list, colnames_list, band|None. Returns dict of per-colname value arrays plus _chrom1/_start1/_end1/_chrom2/_start2/_end2 coord arrays."},
    {"pm_import_mappedseq", gcall<pm_import_mappedseq>, METH_VARARGS, "Import mapped sequences (SAM/tab) into a sparse or dense track"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__pymisha(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pymisha",
        "PyMisha genomics toolkit - C++ extension module",
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        pymisha_module_free
    };

    g_module = PyModule_Create(&moduledef);

    if (!g_module)
        return NULL;

    s_pm_err = PyErr_NewException("pymisha.error", NULL, NULL);
    if (!s_pm_err) {
        Py_DECREF(g_module);
        return NULL;
    }
    if (PyModule_AddObject(g_module, "error", s_pm_err) < 0) {
        Py_DECREF(s_pm_err);
        s_pm_err = NULL;
        Py_DECREF(g_module);
        return NULL;
    }

    import_array();

    struct timespec tm;
    clock_gettime(CLOCK_MONOTONIC, &tm);
    srand48(tm.tv_sec ^ tm.tv_nsec);

    return g_module;
}
