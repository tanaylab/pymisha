// pm_liftover_track: C++ port of liftover.gtrack_liftover orchestrator.
// G1.P3.C: strings P3.A read_source_track_1d + P3.B.2 map_intervals + P1
// liftover_aggregate (SPARSE) / new aggregate_per_bin_cpp (FIXED_BIN) inline,
// avoiding the DataFrame round-trips that the Python orchestrator pays
// between each step. Output dict carries track_type + bin_size so Python
// dispatches to gtrack_create_dense vs gtrack_create_sparse.

#include "pymisha.h"
#include "PMSourceTrack.h"
#include "PMChainIntervals.h"
#include "PMLiftoverAggregate.h"
#include "TGLException.h"
#include "GenomeChromKey.h"

#include <algorithm>
#include <sys/stat.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Holds extracted chain POD column buffers (owned vectors).
struct ChainColumnBuffers {
    std::vector<std::string> chrom;
    std::vector<int64_t> start;
    std::vector<int64_t> end;
    std::vector<int64_t> strand;
    std::vector<std::string> chromsrc;
    std::vector<int64_t> startsrc;
    std::vector<int64_t> endsrc;
    std::vector<int64_t> strandsrc;
    std::vector<int64_t> chain_id;
    std::vector<double>  score;
    int64_t n = 0;
};

// Pull a key from df_dict and coerce to a contiguous 1-D numpy array of the
// requested dtype. Returns nullptr on failure (Python error set).
static PyObject *get_array_local(PyObject *df_dict, const char *key, int dtype)
{
    PyObject *item = PyDict_GetItemString(df_dict, key);
    if (!item) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict missing required key '%s'", key);
        return nullptr;
    }
    PyObject *arr = PyArray_FROM_OTF(item, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return nullptr;
    if (PyArray_NDIM((PyArrayObject *)arr) != 1) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict['%s'] must be 1-D", key);
        Py_DECREF(arr);
        return nullptr;
    }
    return arr;
}

// Extract a Python dict {str: int} -> vector<pair<string, int64_t>> preserving
// insertion order. Sets PyErr and returns false on failure.
bool extract_tgt_chrom_sizes(
    PyObject *py_dict, std::vector<std::pair<std::string, int64_t>> &out)
{
    out.clear();
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(py_dict, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "tgt_chrom_sizes keys must be strings");
            return false;
        }
        Py_ssize_t klen = 0;
        const char *kstr = PyUnicode_AsUTF8AndSize(key, &klen);
        if (!kstr) return false;
        long long sz = PyLong_AsLongLong(value);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "tgt_chrom_sizes values must be int");
            return false;
        }
        out.emplace_back(std::string(kstr, (size_t)klen), (int64_t)sz);
    }
    return true;
}

// Extract a chain Python dict -> ChainColumnBuffers. Sets PyErr and returns
// false on failure.
bool extract_chain_dict(PyObject *chain_dict, ChainColumnBuffers &out)
{
    auto extract_int64_col = [&](const char *key, std::vector<int64_t> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_INT64), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        const int64_t *p = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr);
        dest.assign(p, p + n);
        return true;
    };
    auto extract_double_col = [&](const char *key, std::vector<double> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_DOUBLE), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        const double *p = (const double *)PyArray_DATA((PyArrayObject *)*arr);
        dest.assign(p, p + n);
        return true;
    };
    auto extract_string_col = [&](const char *key, std::vector<std::string> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_OBJECT), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        PyObject **pp = (PyObject **)PyArray_DATA((PyArrayObject *)*arr);
        dest.clear();
        dest.reserve((size_t)n);
        for (npy_intp i = 0; i < n; ++i) {
            if (!pp[i] || !PyUnicode_Check(pp[i])) {
                PyErr_Format(PyExc_TypeError,
                             "chain_df_dict['%s'][%lld] must be a string",
                             key, (long long)i);
                return false;
            }
            Py_ssize_t slen = 0;
            const char *sstr = PyUnicode_AsUTF8AndSize(pp[i], &slen);
            if (!sstr) return false;
            dest.emplace_back(sstr, (size_t)slen);
        }
        return true;
    };

    if (!extract_string_col("chrom", out.chrom)) return false;
    if (!extract_int64_col("start", out.start)) return false;
    if (!extract_int64_col("end", out.end)) return false;
    if (!extract_int64_col("strand", out.strand)) return false;
    if (!extract_string_col("chromsrc", out.chromsrc)) return false;
    if (!extract_int64_col("startsrc", out.startsrc)) return false;
    if (!extract_int64_col("endsrc", out.endsrc)) return false;
    if (!extract_int64_col("strandsrc", out.strandsrc)) return false;
    if (!extract_int64_col("chain_id", out.chain_id)) return false;
    if (!extract_double_col("score", out.score)) return false;

    out.n = (int64_t)out.chrom.size();
    // Length-consistency check
    if ((int64_t)out.start.size() != out.n ||
        (int64_t)out.end.size() != out.n ||
        (int64_t)out.strand.size() != out.n ||
        (int64_t)out.chromsrc.size() != out.n ||
        (int64_t)out.startsrc.size() != out.n ||
        (int64_t)out.endsrc.size() != out.n ||
        (int64_t)out.strandsrc.size() != out.n ||
        (int64_t)out.chain_id.size() != out.n ||
        (int64_t)out.score.size() != out.n) {
        PyErr_SetString(PyExc_ValueError,
                        "chain_df_dict columns have mismatched lengths");
        return false;
    }
    return true;
}

// Build the result dict.
PyObject *build_result_dict(
    const std::vector<std::string> &out_chrom,
    const std::vector<int64_t> &out_start,
    const std::vector<int64_t> &out_end,
    const std::vector<double>  &out_value,
    const std::string &track_type,
    int64_t bin_size)
{
    npy_intp n = (npy_intp)out_chrom.size();

    PMPY py_chrom(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n, NPY_INT64), true);
    PMPY py_value(PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
    if (!py_chrom || !py_start || !py_end || !py_value) return nullptr;

    // Zero-initialize the object array so Py_XDECREF on partial fill is safe.
    PyObject **chrom_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    for (npy_intp i = 0; i < n; ++i) {
        Py_INCREF(Py_None);
        chrom_out[i] = Py_None;
    }
    int64_t *start_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out   = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    double  *value_out = (double *) PyArray_DATA((PyArrayObject *)*py_value);

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(out_chrom[i].data(), out_chrom[i].size());
        if (!s) return nullptr;
        Py_DECREF(chrom_out[i]);  // release the None placeholder
        chrom_out[i] = s;
        start_out[i] = out_start[i];
        end_out[i]   = out_end[i];
        value_out[i] = out_value[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return nullptr;
    if (PyDict_SetItemString(*result, "chrom", *py_chrom) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "start", *py_start) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "end",   *py_end)   < 0) return nullptr;
    if (PyDict_SetItemString(*result, "value", *py_value) < 0) return nullptr;
    PMPY tt(PyUnicode_FromString(track_type.c_str()), true);
    if (!tt || PyDict_SetItemString(*result, "track_type", *tt) < 0) return nullptr;
    PMPY bs(PyLong_FromLongLong(bin_size), true);
    if (!bs || PyDict_SetItemString(*result, "bin_size", *bs) < 0) return nullptr;
    result.to_be_stolen();
    return (PyObject *)*result;
}

PyObject *build_empty_result_dict(const std::string &track_type, int64_t bin_size)
{
    std::vector<std::string> empty_s;
    std::vector<int64_t> empty_i;
    std::vector<double> empty_d;
    return build_result_dict(empty_s, empty_i, empty_i, empty_d, track_type, bin_size);
}

}  // namespace

// Renames each source chromosome that is not a chain source chromosome to its name without
// or with the "chr" prefix, if the chain has that one (R's map_src_chainid2genomeid).
static void match_chain_src_chroms(const std::vector<std::string> &chain_chromsrc,
                                   std::vector<std::string> &src_chrom)
{
    std::unordered_set<std::string> chain_names(chain_chromsrc.begin(), chain_chromsrc.end());
    std::unordered_map<std::string, std::string> renamed;
    for (std::string &chrom : src_chrom) {
        if (chain_names.count(chrom))
            continue;
        auto it = renamed.find(chrom);
        if (it == renamed.end()) {
            std::string alt = chrom.compare(0, 3, "chr") ? "chr" + chrom : chrom.substr(3);
            it = renamed.emplace(chrom, chain_names.count(alt) ? alt : chrom).first;
        }
        chrom = it->second;
    }
}

/*
 * pm_liftover_track(src_track_dir, chain_df_dict, tgt_chrom_sizes,
 *                   cluster_strategy, agg_type, na_rm, min_n, nth_index)
 *   -> dict
 *
 * Full orchestration (G1.P3.C Task 6).
 */
PyObject *pm_liftover_track(PyObject *self, PyObject *args)
{
    const char *src_track_dir;
    PyObject *chain_df_dict;
    PyObject *tgt_chrom_sizes_py;
    const char *cluster_strategy_str;
    const char *agg_type_str;
    int na_rm_int;
    long long min_n_ll;
    long long nth_index_ll;
    PyObject *src_db_chroms_py = nullptr;
    if (!PyArg_ParseTuple(args, "sOOsspLL|O",
                          &src_track_dir, &chain_df_dict, &tgt_chrom_sizes_py,
                          &cluster_strategy_str, &agg_type_str,
                          &na_rm_int, &min_n_ll, &nth_index_ll, &src_db_chroms_py)) {
        return nullptr;
    }
    if (!PyDict_Check(chain_df_dict)) {
        PyErr_SetString(PyExc_TypeError, "chain_df_dict must be a dict");
        return nullptr;
    }
    if (!PyDict_Check(tgt_chrom_sizes_py)) {
        PyErr_SetString(PyExc_TypeError, "tgt_chrom_sizes must be a dict");
        return nullptr;
    }

    try {
        // 1) Read source track. Reports src_type + bin_size.
        std::string src_type;
        int64_t src_bin_size = 0;
        std::vector<std::string> src_chrom;
        std::vector<int64_t>     src_start, src_end;
        std::vector<double>      src_value;

        // 3) Extract chain dict into POD column buffers.
        ChainColumnBuffers chain_buf;
        if (!extract_chain_dict(chain_df_dict, chain_buf)) return nullptr;

        // the chromosomes an indexed source is keyed by, as the caller read them for its
        // database (R's .gdb.chrom_names_at), and the chain's source chromosomes, which
        // name the files of a per-chromosome source by name or by "chr" toggled
        std::vector<std::string> src_db_chroms;
        if (src_db_chroms_py && src_db_chroms_py != Py_None) {
            PMPY seq(PySequence_Fast(src_db_chroms_py, "src_db_chroms must be a sequence"), true);
            if (!seq)
                return nullptr;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE((PyObject *)seq); ++i) {
                const char *name = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM((PyObject *)seq, i));
                if (!name)
                    return nullptr;
                src_db_chroms.emplace_back(name);
            }
        }
        std::vector<std::string> chain_src_names;
        {
            std::unordered_set<std::string> seen;
            for (const std::string &name : chain_buf.chromsrc) {
                if (seen.insert(name).second)
                    chain_src_names.push_back(name);
            }
        }
        GenomeChromKey src_chromkey;
        chain_src_chromkey(chain_src_names, src_chromkey);
        read_source_track_1d_cpp(src_track_dir, src_type, src_bin_size,
                                  src_chrom, src_start, src_end, src_value, &src_chromkey,
                                  src_db_chroms_py && src_db_chroms_py != Py_None ? &src_db_chroms : nullptr);

        // 2) Extract tgt_chrom_sizes Python dict -> vector<pair> preserving
        //    insertion order (Task 4 aggregate_per_bin_cpp API).
        std::vector<std::pair<std::string, int64_t>> tgt_sizes;
        if (!extract_tgt_chrom_sizes(tgt_chrom_sizes_py, tgt_sizes))
            return nullptr;

        // Early exit if source empty OR chain empty.
        if (src_chrom.empty() || chain_buf.n == 0) {
            return build_empty_result_dict(src_type, src_bin_size);
        }

        // An indexed source is read by its database's chromosomes, which match a chain
        // chromosome by name, as is, then without or with the "chr" prefix; the files of a
        // per-chromosome source are found by the chain's names, as is or "chr" toggled
        // (R's gtrack.liftover).
        struct stat idx_st;
        if (stat((std::string(src_track_dir) + "/track.idx").c_str(), &idx_st) == 0)
            match_chain_src_chroms(chain_buf.chromsrc, src_chrom);

        // 4) Map intervals.
        SrcRowInput src_in{
            src_chrom.data(), src_start.data(), src_end.data(),
            src_value.data(), (int64_t)src_chrom.size(),
        };
        ChainRowInput chain_in{
            chain_buf.chrom.data(), chain_buf.start.data(),
            chain_buf.end.data(), chain_buf.strand.data(),
            chain_buf.chromsrc.data(), chain_buf.startsrc.data(),
            chain_buf.endsrc.data(), chain_buf.strandsrc.data(),
            chain_buf.chain_id.data(), chain_buf.score.data(),
            chain_buf.n,
        };
        ClusterStrat strat = parse_cluster_strategy_str(cluster_strategy_str);
        MappedOutput mapped;
        map_intervals_cpp(src_in, chain_in,
                          /*include_metadata=*/false, strat, mapped);

        if (mapped.chrom.empty()) {
            return build_empty_result_dict(src_type, src_bin_size);
        }

        // 5) Aggregate. Dispatch on src_type.
        // Dedup key = (chain id, source interval index), as R's. Pieces of the SAME
        // source interval split by "agg" target segmentation share the key (counted
        // once); DIFFERENT source intervals of the same chain landing in one target
        // locus stay distinct and are aggregated. Keying by chain_id alone collapsed
        // them to the first one's value (R 5.11.6, mirroring IntervalsLiftover's key).
        std::vector<int64_t> agg_key(mapped.chain_id.size());
        for (size_t k = 0; k < agg_key.size(); ++k)
            agg_key[k] = (mapped.chain_id[k] << 32) ^ mapped.intervalID[k];
        AggType agg = parse_agg_type_str(agg_type_str);
        std::vector<std::string> out_chrom;
        std::vector<int64_t>     out_start, out_end;
        std::vector<double>      out_value;
        if (src_type == "dense") {
            // FIXED_BIN preservation per R-parity.
            aggregate_per_bin_cpp(
                mapped.chrom, mapped.start, mapped.end, mapped.value,
                agg_key,
                tgt_sizes, src_bin_size,
                agg, na_rm_int != 0, (int64_t)min_n_ll, (int64_t)nth_index_ll,
                out_chrom, out_start, out_end, out_value);
        } else {
            // SPARSE path.
            aggregate_overlapping_cpp(
                mapped.chrom, mapped.start, mapped.end, mapped.value, agg_key,
                agg, na_rm_int != 0, (int64_t)min_n_ll, (int64_t)nth_index_ll,
                out_chrom, out_start, out_end, out_value);
        }

        // 6) Marshal output -> dict.
        return build_result_dict(out_chrom, out_start, out_end, out_value,
                                  src_type, src_bin_size);

    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (TGLException &e) {
        // the source's chromosome files (GenomeTrack::ChromFiles): two files of a chromosome
        PyErr_SetString(PyExc_ValueError, e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    }
}
