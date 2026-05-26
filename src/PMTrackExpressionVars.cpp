/*
 * PMTrackExpressionVars.cpp
 *
 * Manages track variables in expressions for pymisha
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <unistd.h>

#include "PMTrackExpressionVars.h"
#include "TGLException.h"
#include "PWMScorer.h"
#include "PWMEditDistanceScorer.h"
#include "PWMLseEditDistanceScorer.h"
#include "KmerCounter.h"
#include "MaskedBpCounter.h"
#include "GenomeSeqFetch.h"

// ---- helpers for reading Python spec dicts (same logic as PMVTrack.cpp) ----
namespace {

bool vt_str_to_bool(PyObject *obj, bool default_val) {
    if (!obj) return default_val;
    if (obj == Py_True) return true;
    if (obj == Py_False) return false;
    if (PyBool_Check(obj)) return obj == Py_True;
    if (PyNumber_Check(obj)) {
        PMPY tmp(PyNumber_Long(obj), true);
        if (!tmp) return default_val;
        long v = PyLong_AsLong(tmp);
        return v != 0;
    }
    return default_val;
}

int64_t vt_obj_to_int64(PyObject *obj, int64_t default_val, bool *found = nullptr) {
    if (!obj || obj == Py_None) {
        if (found) *found = false;
        return default_val;
    }
    if (found) *found = true;
    PMPY tmp(PyNumber_Long(obj), true);
    if (!tmp) {
        PyErr_Clear();
        return default_val;
    }
    return PyLong_AsLongLong(tmp);
}

double vt_obj_to_double(PyObject *obj, double default_val, bool *found = nullptr) {
    if (!obj || obj == Py_None) {
        if (found) *found = false;
        return default_val;
    }
    if (found) *found = true;
    PMPY tmp(PyNumber_Float(obj), true);
    if (!tmp) {
        PyErr_Clear();
        return default_val;
    }
    return PyFloat_AsDouble(tmp);
}

std::string vt_obj_to_string(PyObject *obj, const std::string &default_val) {
    if (!obj || obj == Py_None) return default_val;
    if (!PyUnicode_Check(obj)) return default_val;
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) return default_val;
    return std::string(s);
}

PyObject *vt_dict_get(PyObject *dict, const char *key) {
    if (!dict || !PyDict_Check(dict)) return nullptr;
    return PyDict_GetItemString(dict, key);
}

bool vt_parse_pssm(PyObject *obj, DnaPSSM &pssm, double prior) {
    if (!obj || obj == Py_None) return false;

    PMPY arr(PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
    if (!arr || !PyArray_Check((PyArrayObject *)*arr) || PyArray_NDIM((PyArrayObject *)*arr) != 2) {
        PyErr_Clear();
        return false;
    }

    npy_intp rows = PyArray_DIM((PyArrayObject *)*arr, 0);
    npy_intp cols = PyArray_DIM((PyArrayObject *)*arr, 1);
    if (rows <= 0 || cols <= 0) return false;

    npy_intp len = rows;
    bool transposed = false;
    if (cols == 4) {
        len = rows;
        transposed = false;
    } else if (rows == 4) {
        len = cols;
        transposed = true;
    } else {
        return false;
    }

    pssm.resize((int)len);
    for (npy_intp i = 0; i < len; ++i) {
        double pa, pc, pg, pt;
        if (!transposed) {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 0);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 1);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 2);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 3);
        } else {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 0, i);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 1, i);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 2, i);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 3, i);
        }
        pssm[i] = DnaProbVec((float)pa, (float)pc, (float)pg, (float)pt);
    }

    if (prior > 0) {
        pssm.add_dirichlet_prior((float)prior);
    }
    return true;
}

} // anonymous namespace

PMTrackExpressionVars::PMTrackExpressionVars()
    : m_bin_size(0),
      m_common_track_type(GenomeTrack::NUM_TYPES),
      m_common_track_type_valid(false)
{
}

PMTrackExpressionVars::~PMTrackExpressionVars()
{
}

void PMTrackExpressionVars::name2var(const std::string &name, std::string &var_name) const
{
    var_name = name;
    // Replace dots with underscores for Python variable name
    std::replace(var_name.begin(), var_name.end(), '.', '_');
}

PMTrackExpressionVars::TrackVar &PMTrackExpressionVars::add_track_var(const std::string &track_name)
{
    // Check if already exists
    auto it = m_var_map.find(track_name);
    if (it != m_var_map.end()) {
        return m_track_vars[it->second];
    }

    // Create new track variable
    m_track_vars.emplace_back();
    TrackVar &var = m_track_vars.back();
    var.name = track_name;
    name2var(track_name, var.var_name);
    var.values = nullptr;
    var.bin_size = 0;
    var.cur_chromid = -1;
    var.cur_chromid_valid = false;
    var.track_type = GenomeTrack::NUM_TYPES;

    // Check for variable name collision (e.g., "a.b" and "a_b" both normalize to "a_b")
    auto collision_it = m_varname_to_track.find(var.var_name);
    if (collision_it != m_varname_to_track.end() && collision_it->second != track_name) {
        TGLError("Track name collision: '%s' and '%s' both normalize to Python variable '%s'. "
                 "Rename one of the tracks to avoid ambiguity.",
                 track_name.c_str(), collision_it->second.c_str(), var.var_name.c_str());
    }
    m_varname_to_track[var.var_name] = track_name;

    // Get track path
    if (!g_pmdb || !g_pmdb->is_initialized()) {
        TGLError("Database not initialized");
    }

    var.track_path = g_pmdb->track_path(track_name);

    // Check track type
    GenomeTrack::Type track_type = GenomeTrack::get_type(var.track_path.c_str(), g_pmdb->chromkey(), false);
    var.track_type = track_type;

    if (!m_common_track_type_valid) {
        m_common_track_type = track_type;
        m_common_track_type_valid = true;
    } else if (track_type != m_common_track_type) {
        // R misha allows mixing the two 1D scalar formats - dense (FIXED_BIN)
        // and sparse (SPARSE) - in a single track expression, but only when an
        // explicit iterator is supplied (each track is read per iterator bin via
        // read_interval). Permit that combination and flag the expression as
        // mixed-format so implicit iterator inference refuses it (matching R's
        // "cannot implicitly determine iterator policy"). Any other type
        // mismatch (e.g. 1D vs 2D, arrays) remains unsupported.
        const bool a_scalar_1d = track_type == GenomeTrack::FIXED_BIN ||
                                 track_type == GenomeTrack::SPARSE;
        const bool b_scalar_1d = m_common_track_type == GenomeTrack::FIXED_BIN ||
                                 m_common_track_type == GenomeTrack::SPARSE;
        if (a_scalar_1d && b_scalar_1d) {
            // Keep m_common_track_type as the first track's type so any later
            // genuinely-incompatible track (2D/array) still trips this guard;
            // the sticky m_mixed_track_types flag is what callers consult.
            m_mixed_track_types = true;
        } else {
            TGLError("Mixed track types in expression are not supported: '%s' is %s, expected %s",
                     track_name.c_str(), GenomeTrack::TYPE_NAMES[track_type],
                     GenomeTrack::TYPE_NAMES[m_common_track_type]);
        }
    }

    if (track_type == GenomeTrack::FIXED_BIN) {
        // Create fixed bin track object
        var.track = std::make_unique<GenomeTrackFixedBin>();
        auto *fixed_bin = static_cast<GenomeTrackFixedBin *>(var.track.get());

        // Pre-load bin size from first available chromosome to enable proper iterator selection
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        for (unsigned i = 0; i < chromkey.get_num_chroms(); ++i) {
            std::string chrom_file = GenomeTrack::find_existing_1d_filename(chromkey, var.track_path, i);
            if (!chrom_file.empty()) {
                std::string full_path = var.track_path + "/" + chrom_file;
                fixed_bin->init_read(full_path.c_str(), i);
                var.bin_size = fixed_bin->get_bin_size();

                // Update global bin size and check for consistency
                if (m_bin_size == 0) {
                    m_bin_size = var.bin_size;
                } else if (m_bin_size != var.bin_size) {
                    TGLError("Mixed bin sizes detected: track '%s' has bin size %ld, "
                             "but previous tracks have bin size %ld. "
                             "Use explicit iterator policy to resolve.",
                             track_name.c_str(), var.bin_size, m_bin_size);
                }
                vdebug("Track '%s' bin size: %ld (from chrom %d)\n",
                       track_name.c_str(), var.bin_size, i);
                break;
            }
        }
    } else if (track_type == GenomeTrack::SPARSE) {
        var.track = std::make_unique<GenomeTrackSparse>();
    } else if (track_type == GenomeTrack::ARRAYS) {
        TGLError("gextract / scanner does not support array tracks yet. "
                 "Use gtrack_array_extract('%s', ...) to read the per-column "
                 "values, or gtrack_array_get_colnames() to inspect the "
                 "column names.",
                 track_name.c_str());
    } else {
        TGLError("Track type '%s' not yet supported for track: %s",
                 GenomeTrack::TYPE_NAMES[track_type], track_name.c_str());
    }

    m_var_map[track_name] = m_track_vars.size() - 1;
    return var;
}

void PMTrackExpressionVars::parse_exprs(const std::vector<std::string> &track_exprs,
                                        std::vector<std::string> &exprs4compile,
                                        PyObject *py_vtracks)
{
    m_track_vars.clear();
    m_var_map.clear();
    m_vtrack_vars.clear();
    m_vtrack_var_map.clear();
    m_varname_to_track.clear();
    m_bin_size = 0;
    m_common_track_type = GenomeTrack::NUM_TYPES;
    m_common_track_type_valid = false;
    m_mixed_track_types = false;

    // Collect vtrack names from the dict so we can recognize them in expressions
    std::unordered_map<std::string, PyObject *> vtrack_specs;
    if (py_vtracks && PyDict_Check(py_vtracks)) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(py_vtracks, &pos, &key, &value)) {
            if (PyUnicode_Check(key)) {
                vtrack_specs[PyUnicode_AsUTF8(key)] = value;
            }
        }
    }

    exprs4compile.resize(track_exprs.size());

    for (size_t iexpr = 0; iexpr < track_exprs.size(); ++iexpr) {
        const std::string &expr = track_exprs[iexpr];
        std::string &expr4compile = exprs4compile[iexpr];
        expr4compile = expr;

        // Scan for track/vtrack names in the expression
        // Track names contain letters, digits, underscores, and dots
        size_t pos = 0;
        while (pos < expr.size()) {
            // Skip non-identifier characters
            while (pos < expr.size() && !isalnum(expr[pos]) && expr[pos] != '_') {
                ++pos;
            }
            if (pos >= expr.size()) break;

            // Find end of potential identifier
            size_t start = pos;
            while (pos < expr.size() && (isalnum(expr[pos]) || expr[pos] == '_' || expr[pos] == '.')) {
                ++pos;
            }

            std::string name = expr.substr(start, pos - start);

            // Check if this is a vtrack name (check vtracks first, they shadow tracks)
            auto vit = vtrack_specs.find(name);
            if (vit != vtrack_specs.end()) {
                VTrackVar &vvar = add_vtrack_var(name, vit->second);
                // Replace vtrack name with variable name in expression
                size_t offset = expr4compile.size() - expr.size();
                expr4compile.replace(start + offset, name.size(), vvar.var_name);
                continue;
            }

            // Also check for dotted prefix matches against vtracks
            // (e.g., "my.vtrack" might appear as identifier "my.vtrack")
            if (name.find('.') != std::string::npos) {
                bool found_vtrack = false;
                // Try progressively shorter dotted prefixes
                size_t dot_pos = name.rfind('.');
                while (dot_pos != std::string::npos) {
                    std::string prefix = name.substr(0, dot_pos);
                    auto vit2 = vtrack_specs.find(prefix);
                    if (vit2 != vtrack_specs.end()) {
                        VTrackVar &vvar = add_vtrack_var(prefix, vit2->second);
                        size_t offset = expr4compile.size() - expr.size();
                        expr4compile.replace(start + offset, prefix.size(), vvar.var_name);
                        found_vtrack = true;
                        break;
                    }
                    if (dot_pos == 0) break;
                    dot_pos = name.rfind('.', dot_pos - 1);
                }
                if (found_vtrack) continue;
            }

            // Check if this is a physical track name
            if (g_pmdb->track_exists(name)) {
                TrackVar &var = add_track_var(name);

                // Replace track name with variable name in expression
                size_t offset = expr4compile.size() - expr.size();
                expr4compile.replace(start + offset, name.size(), var.var_name);
            }
        }
    }

    vdebug("Parsed expressions, found %lu track variables, %lu vtrack variables\n",
           m_track_vars.size(), m_vtrack_vars.size());
}

void PMTrackExpressionVars::define_py_vars(unsigned size, PMPY &ldict, bool use_python)
{
    npy_intp dims[1] = {(npy_intp)size};

    for (auto &var : m_track_vars) {
        if (use_python) {
            // Create NumPy array for this track variable
            var.py_var.assign(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
            if (!var.py_var) {
                TGLError("Failed to create NumPy array for track variable: %s", var.name.c_str());
            }

            var.values = (double *)PyArray_DATA((PyArrayObject *)*var.py_var);

            // Add to local dictionary
            PyDict_SetItemString(ldict, var.var_name.c_str(), var.py_var);

            vdebug("Defined Python variable '%s' for track '%s' (size=%u)\n",
                   var.var_name.c_str(), var.name.c_str(), size);
        } else {
            // Use C++ vector storage
            var.cpp_values.resize(size);
            var.values = var.cpp_values.data();

            vdebug("Allocated C++ storage for track '%s' (size=%u)\n",
                   var.name.c_str(), size);
        }

        // Initialize with NaN
        for (unsigned i = 0; i < size; ++i) {
            var.values[i] = std::nan("");
        }
    }

    // Allocate arrays for vtrack variables (same pattern)
    for (auto &vvar : m_vtrack_vars) {
        if (use_python) {
            vvar.py_var.assign(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
            if (!vvar.py_var) {
                TGLError("Failed to create NumPy array for vtrack variable: %s", vvar.name.c_str());
            }
            vvar.values = (double *)PyArray_DATA((PyArrayObject *)*vvar.py_var);
            PyDict_SetItemString(ldict, vvar.var_name.c_str(), vvar.py_var);
            vdebug("Defined Python variable '%s' for vtrack '%s' (size=%u)\n",
                   vvar.var_name.c_str(), vvar.name.c_str(), size);
        } else {
            vvar.cpp_values.resize(size);
            vvar.values = vvar.cpp_values.data();
            vdebug("Allocated C++ storage for vtrack '%s' (size=%u)\n",
                   vvar.name.c_str(), size);
        }
        for (unsigned i = 0; i < size; ++i) {
            vvar.values[i] = std::nan("");
        }
    }
}

void PMTrackExpressionVars::set_vars(const GInterval &interval, unsigned idx)
{
    for (auto &var : m_track_vars) {
        if (var.track_type == GenomeTrack::SPARSE) {
            GenomeTrackSparse *sparse = static_cast<GenomeTrackSparse *>(var.track.get());

            if (var.cur_chromid != interval.chromid) {
                std::string chrom_file = GenomeTrack::find_existing_1d_filename(
                    g_pmdb->chromkey(), var.track_path, interval.chromid);

                std::string full_path = var.track_path + "/" + chrom_file;
                // Indexed tracks have no per-chrom file (data is in track.dat via
                // track.idx); init_read reads through the index. Only gate on the
                // per-chrom file for the per-chromosome layout, else indexed
                // sparse tracks always read back NaN.
                const bool indexed =
                    access((var.track_path + "/track.idx").c_str(), F_OK) == 0;
                if (!indexed && access(full_path.c_str(), F_OK) != 0) {
                    var.cur_chromid = interval.chromid;
                    var.cur_chromid_valid = false;
                } else {
                    sparse->init_read(full_path.c_str(), interval.chromid);
                    var.cur_chromid = interval.chromid;
                    var.cur_chromid_valid = true;
                }
            }

            if (!var.cur_chromid_valid) {
                var.values[idx] = std::nan("");
                continue;
            }

            sparse->read_interval(interval);
            var.values[idx] = sparse->last_avg();
            continue;
        }

        GenomeTrackFixedBin *fixed_bin = static_cast<GenomeTrackFixedBin *>(var.track.get());

        // Load chromosome if needed
        if (var.cur_chromid != interval.chromid) {
            // Find the chromosome file in the track directory
            std::string chrom_file = GenomeTrack::find_existing_1d_filename(
                g_pmdb->chromkey(), var.track_path, interval.chromid);

            if (chrom_file.empty()) {
                // Chromosome not found in track
                var.cur_chromid = interval.chromid;
                var.cur_chromid_valid = false;
            } else {
                std::string full_path = var.track_path + "/" + chrom_file;
                fixed_bin->init_read(full_path.c_str(), interval.chromid);
                var.cur_chromid = interval.chromid;
                var.cur_chromid_valid = true;
                var.bin_size = fixed_bin->get_bin_size();

                // Update global bin size and check for mismatches
                if (m_bin_size == 0) {
                    m_bin_size = var.bin_size;
                } else if (m_bin_size != var.bin_size) {
                    TGLError("Mixed bin sizes detected: track '%s' has bin size %ld, "
                             "but previous tracks have bin size %ld. "
                             "Use explicit iterator policy to resolve.",
                             var.name.c_str(), var.bin_size, m_bin_size);
                }
            }
            var.last_bin = -1;  // Reset sequential tracking on chrom change
        }

        if (!var.cur_chromid_valid) {
            var.values[idx] = std::nan("");
            continue;
        }

        // Average the track over the WHOLE interval, matching R misha and the
        // sparse path above. The previous implementation point-sampled the
        // single native bin at the interval midpoint, which is only correct
        // when the iterator bin equals the native bin; for a coarsening
        // iterator (bin spans several native bins) it returned the midpoint
        // bin's value instead of the mean. read_interval() handles the cursor
        // and the common sequential single-bin case (its fast path) itself.
        fixed_bin->read_interval(interval);
        var.values[idx] = fixed_bin->last_avg();
    }

    // Evaluate virtual track variables
    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    for (auto &vvar : m_vtrack_vars) {
        if (vvar.scorer) {
            // Sequence-based vtrack: apply shift, call scorer
            GInterval eval;
            if (!apply_shift(interval, vvar.sshift, vvar.eshift, chromkey, eval)) {
                vvar.values[idx] = std::nan("");
            } else {
                vvar.values[idx] = vvar.scorer->score_interval(eval, chromkey);
            }
        } else if (!vvar.src_track_name.empty()) {
            // Value-based vtrack with physical track source
            GInterval eval;
            if (!apply_shift(interval, vvar.sshift, vvar.eshift, chromkey, eval)) {
                vvar.values[idx] = std::nan("");
            } else {
                vvar.values[idx] = eval_value_based_vtrack(vvar, eval);
            }
        } else {
            // Fallback: should not happen if add_vtrack_var was correct
            vvar.values[idx] = std::nan("");
        }
    }
}

const PMTrackExpressionVars::TrackVar *PMTrackExpressionVars::var(const char *name) const
{
    auto it = m_var_map.find(name);
    if (it != m_var_map.end()) {
        return &m_track_vars[it->second];
    }
    return nullptr;
}

const PMTrackExpressionVars::VTrackVar *PMTrackExpressionVars::vtrack_var(const char *name) const
{
    auto it = m_vtrack_var_map.find(name);
    if (it != m_vtrack_var_map.end()) {
        return &m_vtrack_vars[it->second];
    }
    return nullptr;
}

void PMTrackExpressionVars::pad_tail_with_nan(unsigned start_idx, unsigned end_idx)
{
    for (auto &var : m_track_vars) {
        if (var.values) {
            for (unsigned i = start_idx; i < end_idx; ++i) {
                var.values[i] = std::nan("");
            }
        }
    }
    for (auto &vvar : m_vtrack_vars) {
        if (vvar.values) {
            for (unsigned i = start_idx; i < end_idx; ++i) {
                vvar.values[i] = std::nan("");
            }
        }
    }
}

// ---- apply_shift: clamp interval to chromosome bounds ----

bool PMTrackExpressionVars::apply_shift(const GInterval &in, int64_t sshift, int64_t eshift,
                                         const GenomeChromKey &chromkey, GInterval &out)
{
    out = in;
    int64_t start = in.start + sshift;
    int64_t end = in.end + eshift;
    if (start < 0) start = 0;
    int64_t chrom_size = (int64_t)chromkey.get_chrom_size(in.chromid);
    if (end > chrom_size) end = chrom_size;
    out.start = start;
    out.end = end;
    return out.start < out.end;
}

// ---- add_vtrack_var: register a vtrack and build its scorer ----

PMTrackExpressionVars::VTrackVar &PMTrackExpressionVars::add_vtrack_var(
    const std::string &vtrack_name, PyObject *spec)
{
    // Check if already exists
    auto it = m_vtrack_var_map.find(vtrack_name);
    if (it != m_vtrack_var_map.end()) {
        return m_vtrack_vars[it->second];
    }

    m_vtrack_vars.emplace_back();
    VTrackVar &vvar = m_vtrack_vars.back();
    vvar.name = vtrack_name;
    // Use a distinct prefix to avoid collisions with physical track variable names
    // (e.g., track "foo.bar" -> "foo_bar" but vtrack "foo_bar" -> "__pmvt_foo_bar")
    vvar.var_name = "__pmvt_";
    for (char c : vtrack_name) {
        vvar.var_name += (c == '.') ? '_' : c;
    }
    vvar.py_spec = spec;  // borrowed reference

    // Parse common fields
    vvar.sshift = vt_obj_to_int64(vt_dict_get(spec, "sshift"), 0);
    vvar.eshift = vt_obj_to_int64(vt_dict_get(spec, "eshift"), 0);

    std::string func = vt_obj_to_string(vt_dict_get(spec, "func"), "avg");
    std::transform(func.begin(), func.end(), func.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    vvar.func = func;

    // Parse param
    PyObject *params_obj = vt_dict_get(spec, "params");
    if (params_obj && PyList_Check(params_obj) && PyList_Size(params_obj) > 0) {
        params_obj = PyList_GetItem(params_obj, 0);
    }
    vvar.param = vt_obj_to_double(params_obj, std::numeric_limits<double>::quiet_NaN());

    // Determine vtrack type from func and src
    PyObject *py_src = vt_dict_get(spec, "src");
    bool is_seq_func = (func.rfind("pwm", 0) == 0 || func == "kmer.count" || func == "kmer.frac" ||
                        func == "masked.count" || func == "masked.frac");

    if (is_seq_func && (!py_src || py_src == Py_None)) {
        // Sequence-based vtrack
        build_vtrack_scorer(vvar, spec);
    } else if (py_src && py_src != Py_None && PyUnicode_Check(py_src)) {
        // Physical track source — value-based vtrack
        setup_value_based_vtrack(vvar, spec, func);
    } else {
        // Interval-based or DataFrame source — delegate to pm_vtrack_compute
        // This should not happen on the C++ path (Python filters these out)
        TGLError("Virtual track '%s' has unsupported source type for C++ inline evaluation",
                 vtrack_name.c_str());
    }

    m_vtrack_var_map[vtrack_name] = m_vtrack_vars.size() - 1;
    return vvar;
}

// ---- build_vtrack_scorer: construct sequence scorer from spec ----

void PMTrackExpressionVars::build_vtrack_scorer(VTrackVar &vvar, PyObject *spec)
{
    const std::string &func = vvar.func;

    // Create per-vtrack sequence fetcher
    vvar.seqfetch = std::make_unique<GenomeSeqFetch>();
    vvar.seqfetch->set_seqdir(g_pmdb->groot() + "/seq");

    if (func.rfind("pwm", 0) == 0) {
        PyObject *py_pssm = vt_dict_get(spec, "pssm");
        double prior = vt_obj_to_double(vt_dict_get(spec, "prior"), 0.01);
        bool bidirect = vt_str_to_bool(vt_dict_get(spec, "bidirect"), true);
        bool extend = vt_str_to_bool(vt_dict_get(spec, "extend"), true);
        int strand_mode = (int)vt_obj_to_int64(vt_dict_get(spec, "strand"), 0);
        double score_thresh = vt_obj_to_double(vt_dict_get(spec, "score_thresh"), 0.0);
        if (PyObject *alt = vt_dict_get(spec, "score.thresh")) {
            score_thresh = vt_obj_to_double(alt, score_thresh);
        }

        DnaPSSM pssm;
        if (!vt_parse_pssm(py_pssm, pssm, prior)) {
            TGLError("pwm vtrack '%s' requires a numeric pssm matrix (shape Lx4)", vvar.name.c_str());
        }
        pssm.set_bidirect(bidirect);
        char strand = bidirect ? 0 : (char)strand_mode;

        // PWM edit distance functions
        bool is_edit_dist = (func == "pwm.edit_distance" || func == "pwm.edit_distance.pos" ||
                             func == "pwm.max.edit_distance");
        bool is_lse_edit_dist = (func == "pwm.edit_distance.lse" || func == "pwm.edit_distance.lse.pos");

        if (is_edit_dist) {
            int max_edits = (int)vt_obj_to_int64(vt_dict_get(spec, "max_edits"), -1);
            int max_indels = (int)vt_obj_to_int64(vt_dict_get(spec, "max_indels"), 0);
            float score_min_f = std::numeric_limits<float>::quiet_NaN();
            float score_max_f = std::numeric_limits<float>::quiet_NaN();
            bool found_smin = false, found_smax = false;
            double smin_d = vt_obj_to_double(vt_dict_get(spec, "score_min"), 0.0, &found_smin);
            double smax_d = vt_obj_to_double(vt_dict_get(spec, "score_max"), 0.0, &found_smax);
            if (found_smin) score_min_f = (float)smin_d;
            if (found_smax) score_max_f = (float)smax_d;

            // Parse direction parameter
            std::string dir_str = vt_obj_to_string(vt_dict_get(spec, "direction"), "above");
            PWMEditDistanceScorer::Direction direction = PWMEditDistanceScorer::Direction::ABOVE;
            if (dir_str == "below") {
                direction = PWMEditDistanceScorer::Direction::BELOW;
            } else if (dir_str != "above") {
                TGLError("direction parameter must be \"above\" or \"below\"");
            }

            PWMEditDistanceScorer::Mode mode = PWMEditDistanceScorer::Mode::MIN_EDITS;
            if (func == "pwm.edit_distance.pos") mode = PWMEditDistanceScorer::Mode::MIN_EDITS_POSITION;
            else if (func == "pwm.max.edit_distance") mode = PWMEditDistanceScorer::Mode::PWM_MAX_EDITS;

            vvar.scorer = std::make_unique<PWMEditDistanceScorer>(
                pssm, vvar.seqfetch.get(), (float)score_thresh,
                max_edits, extend, strand, mode,
                score_min_f, max_indels, score_max_f, direction);
        } else if (is_lse_edit_dist) {
            int max_edits = (int)vt_obj_to_int64(vt_dict_get(spec, "max_edits"), -1);
            float score_min_f = std::numeric_limits<float>::quiet_NaN();
            float score_max_f = std::numeric_limits<float>::quiet_NaN();
            bool found_smin = false, found_smax = false;
            double smin_d = vt_obj_to_double(vt_dict_get(spec, "score_min"), 0.0, &found_smin);
            double smax_d = vt_obj_to_double(vt_dict_get(spec, "score_max"), 0.0, &found_smax);
            if (found_smin) score_min_f = (float)smin_d;
            if (found_smax) score_max_f = (float)smax_d;

            // Parse direction parameter
            std::string lse_dir_str = vt_obj_to_string(vt_dict_get(spec, "direction"), "above");
            PWMLseEditDistanceScorer::Direction lse_direction = PWMLseEditDistanceScorer::Direction::ABOVE;
            if (lse_dir_str == "below") {
                lse_direction = PWMLseEditDistanceScorer::Direction::BELOW;
            } else if (lse_dir_str != "above") {
                TGLError("direction parameter must be \"above\" or \"below\"");
            }

            PWMLseEditDistanceScorer::Mode mode = PWMLseEditDistanceScorer::Mode::LSE_EDIT_DISTANCE;
            if (func == "pwm.edit_distance.lse.pos") mode = PWMLseEditDistanceScorer::Mode::LSE_EDIT_DISTANCE_POS;

            vvar.scorer = std::make_unique<PWMLseEditDistanceScorer>(
                pssm, vvar.seqfetch.get(), (float)score_thresh,
                max_edits, extend, strand, mode,
                score_min_f, score_max_f, lse_direction);
        } else {
            // Standard PWM scoring: pwm, pwm.max, pwm.max.pos, pwm.count
            std::vector<float> spat_factor;
            int spat_bin = (int)vt_obj_to_int64(vt_dict_get(spec, "spat_bin"), 1);
            bool has_spat = false;
            if (PyObject *spat = vt_dict_get(spec, "spat_factor")) {
                PMPY arr(PyArray_FROM_OTF(spat, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
                if (arr && PyArray_NDIM((PyArrayObject *)*arr) == 1) {
                    npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
                    spat_factor.resize(n);
                    for (npy_intp i = 0; i < n; ++i) {
                        spat_factor[i] = (float)(*(double *)PyArray_GETPTR1((PyArrayObject *)*arr, i));
                    }
                    has_spat = true;
                }
            }

            int64_t spat_min = vt_obj_to_int64(vt_dict_get(spec, "spat_min"), 0);
            int64_t spat_max = vt_obj_to_int64(vt_dict_get(spec, "spat_max"), 0);
            if (has_spat && spat_max > spat_min) {
                pssm.set_range((int)spat_min, (int)spat_max);
            }

            PWMScorer::ScoringMode pwm_mode = PWMScorer::TOTAL_LIKELIHOOD;
            if (func == "pwm.max") pwm_mode = PWMScorer::MAX_LIKELIHOOD;
            else if (func == "pwm.max.pos") pwm_mode = PWMScorer::MAX_LIKELIHOOD_POS;
            else if (func == "pwm.count") pwm_mode = PWMScorer::MOTIF_COUNT;

            vvar.scorer = std::make_unique<PWMScorer>(
                pssm, vvar.seqfetch.get(), extend, pwm_mode, strand,
                spat_factor, spat_bin, (float)score_thresh);
        }
    } else if (func == "kmer.count" || func == "kmer.frac") {
        std::string kmer = vt_obj_to_string(vt_dict_get(spec, "kmer"), "");
        if (kmer.empty()) {
            PyObject *params = vt_dict_get(spec, "params");
            kmer = vt_obj_to_string(params, "");
        }
        if (kmer.empty()) {
            TGLError("kmer vtrack '%s' requires 'kmer' parameter", vvar.name.c_str());
        }
        bool extend = vt_str_to_bool(vt_dict_get(spec, "extend"), true);
        char strand = (char)vt_obj_to_int64(vt_dict_get(spec, "strand"), 0);
        KmerCounter::CountMode mode = (func == "kmer.frac") ? KmerCounter::FRACTION : KmerCounter::SUM;

        vvar.scorer = std::make_unique<KmerCounter>(
            kmer, vvar.seqfetch.get(), mode, extend, strand);
    } else if (func == "masked.count" || func == "masked.frac") {
        MaskedBpCounter::CountMode mode = (func == "masked.frac") ? MaskedBpCounter::FRACTION : MaskedBpCounter::COUNT;

        vvar.scorer = std::make_unique<MaskedBpCounter>(
            vvar.seqfetch.get(), mode);
    } else {
        TGLError("Unsupported sequence-based vtrack function '%s' for vtrack '%s'",
                 func.c_str(), vvar.name.c_str());
    }
}

// ---- setup_value_based_vtrack: prepare for track-source value aggregation ----

void PMTrackExpressionVars::setup_value_based_vtrack(VTrackVar &vvar, PyObject *spec,
                                                      const std::string &func)
{
    PyObject *py_src = vt_dict_get(spec, "src");
    vvar.src_track_name = vt_obj_to_string(py_src, "");
    if (vvar.src_track_name.empty()) {
        TGLError("Value-based vtrack '%s' requires a track name as source", vvar.name.c_str());
    }

    if (!g_pmdb->track_exists(vvar.src_track_name)) {
        TGLError("Source track '%s' for vtrack '%s' does not exist",
                 vvar.src_track_name.c_str(), vvar.name.c_str());
    }

    vvar.src_track_path = g_pmdb->track_path(vvar.src_track_name);
    vvar.src_track_type = GenomeTrack::get_type(vvar.src_track_path.c_str(), g_pmdb->chromkey(), false);

    if (vvar.src_track_type == GenomeTrack::FIXED_BIN) {
        vvar.src_track = std::make_unique<GenomeTrackFixedBin>();
    } else if (vvar.src_track_type == GenomeTrack::SPARSE) {
        vvar.src_track = std::make_unique<GenomeTrackSparse>();
    } else {
        TGLError("Track type not supported for value-based vtrack '%s': source track '%s'",
                 vvar.name.c_str(), vvar.src_track_name.c_str());
    }

    // Register special functions on 1D tracks
    GenomeTrack1D *track1d = dynamic_cast<GenomeTrack1D *>(vvar.src_track.get());
    if (track1d) {
        if (func == "stddev" || func == "std") track1d->register_function(GenomeTrack1D::STDDEV);
        if (func == "quantile") track1d->register_quantile(10000, 1000, 1000);
        if (func == "exists") track1d->register_function(GenomeTrack1D::EXISTS);
        if (func == "size") track1d->register_function(GenomeTrack1D::SIZE);
        if (func == "sample") track1d->register_function(GenomeTrack1D::SAMPLE);
        if (func == "sample.pos.abs" || func == "sample.pos.relative") track1d->register_function(GenomeTrack1D::SAMPLE_POS);
        if (func == "first") track1d->register_function(GenomeTrack1D::FIRST);
        if (func == "first.pos.abs" || func == "first.pos.relative") track1d->register_function(GenomeTrack1D::FIRST_POS);
        if (func == "last") track1d->register_function(GenomeTrack1D::LAST);
        if (func == "last.pos.abs" || func == "last.pos.relative") track1d->register_function(GenomeTrack1D::LAST_POS);
        if (func == "max.pos.abs" || func == "max.pos.relative") track1d->register_function(GenomeTrack1D::MAX_POS);
        if (func == "min.pos.abs" || func == "min.pos.relative") track1d->register_function(GenomeTrack1D::MIN_POS);
    }

    vvar.src_cur_chromid = -1;
    vvar.src_cur_chromid_valid = false;
}

// ---- eval_value_based_vtrack: compute value for a single shifted interval ----

double PMTrackExpressionVars::eval_value_based_vtrack(VTrackVar &vvar, const GInterval &eval)
{
    const std::string &func = vvar.func;
    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    // Load chromosome if needed
    if (vvar.src_cur_chromid != eval.chromid) {
        std::string chrom_file = GenomeTrack::find_existing_1d_filename(
            chromkey, vvar.src_track_path, eval.chromid);
        std::string full_path = vvar.src_track_path + "/" + chrom_file;
        // Indexed tracks have no per-chrom file; init_read reads through
        // track.idx. Gating on the per-chrom file made value-based vtracks over
        // indexed source tracks always return NaN.
        const bool indexed =
            access((vvar.src_track_path + "/track.idx").c_str(), F_OK) == 0;
        if (!indexed && access(full_path.c_str(), F_OK) != 0) {
            vvar.src_cur_chromid = eval.chromid;
            vvar.src_cur_chromid_valid = false;
        } else {
            if (vvar.src_track_type == GenomeTrack::FIXED_BIN) {
                auto *fb = static_cast<GenomeTrackFixedBin *>(vvar.src_track.get());
                fb->init_read(full_path.c_str(), eval.chromid);
            } else if (vvar.src_track_type == GenomeTrack::SPARSE) {
                auto *sp = static_cast<GenomeTrackSparse *>(vvar.src_track.get());
                sp->init_read(full_path.c_str(), eval.chromid);
                vvar.sp_intervals_ptr = &sp->get_intervals();
                vvar.sp_vals_ptr = &sp->get_vals();
                vvar.sp_scan_idx = 0;
                vvar.sp_last_start = std::numeric_limits<int64_t>::min();
                vvar.sp_scan_ready = false;
            }
            vvar.src_cur_chromid = eval.chromid;
            vvar.src_cur_chromid_valid = true;
        }
    }

    if (!vvar.src_cur_chromid_valid) {
        if (func == "exists" || func == "size") return 0.0;
        return std::numeric_limits<double>::quiet_NaN();
    }

    // For sparse tracks with fast-reduce functions, use the direct scan approach
    GenomeTrackSparse *sparse = (vvar.src_track_type == GenomeTrack::SPARSE)
        ? static_cast<GenomeTrackSparse *>(vvar.src_track.get()) : nullptr;
    GenomeTrackFixedBin *fixed_bin = (vvar.src_track_type == GenomeTrack::FIXED_BIN)
        ? static_cast<GenomeTrackFixedBin *>(vvar.src_track.get()) : nullptr;

    bool sparse_fast_reduce = sparse &&
        (func == "avg" || func == "mean" || func == "sum" ||
         func == "min" || func == "max" || func == "size" || func == "exists");

    if (sparse_fast_reduce && vvar.sp_intervals_ptr) {
        const auto &sp_intervals = *vvar.sp_intervals_ptr;
        const auto &sp_vals = *vvar.sp_vals_ptr;
        if (sp_intervals.empty()) {
            if (func == "size" || func == "exists") return 0.0;
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Hybrid sequential/binary scan
        if (!vvar.sp_scan_ready || eval.start < vvar.sp_last_start) {
            size_t lo = 0, hi = sp_intervals.size();
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (sp_intervals[mid].end <= eval.start) lo = mid + 1;
                else hi = mid;
            }
            vvar.sp_scan_idx = lo;
            vvar.sp_scan_ready = true;
        } else {
            while (vvar.sp_scan_idx < sp_intervals.size() &&
                   sp_intervals[vvar.sp_scan_idx].end <= eval.start) {
                ++vvar.sp_scan_idx;
            }
        }
        vvar.sp_last_start = eval.start;

        // Collect matching values
        double result_sum = 0.0;
        double result_min = std::numeric_limits<double>::max();
        double result_max = -std::numeric_limits<double>::max();
        size_t count = 0;

        for (size_t j = vvar.sp_scan_idx; j < sp_intervals.size(); ++j) {
            if (sp_intervals[j].start >= eval.end) break;
            if (sp_intervals[j].end > eval.start) {
                float v = sp_vals[j];
                if (!std::isnan(v)) {
                    result_sum += v;
                    if (v < result_min) result_min = v;
                    if (v > result_max) result_max = v;
                    ++count;
                }
            }
        }

        if (count == 0) {
            if (func == "size" || func == "exists") return 0.0;
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (func == "avg" || func == "mean") return result_sum / count;
        if (func == "sum") return result_sum;
        if (func == "min") return result_min;
        if (func == "max") return result_max;
        if (func == "size") return (double)count;
        if (func == "exists") return 1.0;
        return std::numeric_limits<double>::quiet_NaN();
    }

    // General path: use GenomeTrack1D read_interval
    GenomeTrack1D *track1d = dynamic_cast<GenomeTrack1D *>(vvar.src_track.get());
    if (!track1d) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    track1d->read_interval(eval);

    if (func == "avg" || func == "mean") return track1d->last_avg();
    if (func == "sum") return track1d->last_sum();
    if (func == "min") return track1d->last_min();
    if (func == "max") return track1d->last_max();
    if (func == "stddev" || func == "std") return track1d->last_stddev();
    if (func == "quantile") return track1d->last_quantile(vvar.param);
    if (func == "exists") return track1d->last_exists();
    if (func == "size") return track1d->last_size();
    if (func == "first") return track1d->last_first();
    if (func == "last") return track1d->last_last();
    if (func == "sample") return track1d->last_sample();
    if (func == "nearest") return track1d->last_nearest();

    // Position-based funcs
    if (func == "first.pos.abs") return track1d->last_first_pos();
    if (func == "first.pos.relative") {
        double pos = track1d->last_first_pos();
        return std::isnan(pos) ? pos : pos - eval.start;
    }
    if (func == "last.pos.abs") return track1d->last_last_pos();
    if (func == "last.pos.relative") {
        double pos = track1d->last_last_pos();
        return std::isnan(pos) ? pos : pos - eval.start;
    }
    if (func == "max.pos.abs") return track1d->last_max_pos();
    if (func == "max.pos.relative") {
        double pos = track1d->last_max_pos();
        return std::isnan(pos) ? pos : pos - eval.start;
    }
    if (func == "min.pos.abs") return track1d->last_min_pos();
    if (func == "min.pos.relative") {
        double pos = track1d->last_min_pos();
        return std::isnan(pos) ? pos : pos - eval.start;
    }
    if (func == "sample.pos.abs") return track1d->last_sample_pos();
    if (func == "sample.pos.relative") {
        double pos = track1d->last_sample_pos();
        return std::isnan(pos) ? pos : pos - eval.start;
    }

    // LSE (log-sum-exp)
    if (func == "lse") {
        // Need to collect raw values for LSE
        if (fixed_bin) {
            unsigned bin_size = fixed_bin->get_bin_size();
            int64_t sbin = eval.start / bin_size;
            int64_t ebin = (int64_t)std::ceil(eval.end / (double)bin_size);
            std::vector<float> bin_vals;
            int64_t bins_read = fixed_bin->read_bins_bulk(sbin, ebin - sbin, bin_vals);
            std::vector<double> vals;
            vals.reserve(bins_read);
            for (int64_t j = 0; j < bins_read; ++j) {
                if (!std::isnan(bin_vals[j])) vals.push_back((double)bin_vals[j]);
            }
            if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
            double m = *std::max_element(vals.begin(), vals.end());
            if (std::isinf(m) && m < 0) return m;
            double sum_exp = 0.0;
            for (double v : vals) sum_exp += std::exp(v - m);
            return m + std::log(sum_exp);
        }
        // Sparse LSE: collect values
        if (sparse && vvar.sp_intervals_ptr) {
            const auto &sp_ints = *vvar.sp_intervals_ptr;
            const auto &sp_v = *vvar.sp_vals_ptr;
            std::vector<double> vals;
            // Use scan state
            if (!vvar.sp_scan_ready || eval.start < vvar.sp_last_start) {
                size_t lo = 0, hi = sp_ints.size();
                while (lo < hi) {
                    size_t mid = lo + (hi - lo) / 2;
                    if (sp_ints[mid].end <= eval.start) lo = mid + 1;
                    else hi = mid;
                }
                vvar.sp_scan_idx = lo;
                vvar.sp_scan_ready = true;
            } else {
                while (vvar.sp_scan_idx < sp_ints.size() &&
                       sp_ints[vvar.sp_scan_idx].end <= eval.start) {
                    ++vvar.sp_scan_idx;
                }
            }
            vvar.sp_last_start = eval.start;

            for (size_t j = vvar.sp_scan_idx; j < sp_ints.size(); ++j) {
                if (sp_ints[j].start >= eval.end) break;
                if (sp_ints[j].end > eval.start && !std::isnan(sp_v[j])) {
                    vals.push_back((double)sp_v[j]);
                }
            }
            if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
            double m = *std::max_element(vals.begin(), vals.end());
            if (std::isinf(m) && m < 0) return m;
            double sum_exp = 0.0;
            for (double v : vals) sum_exp += std::exp(v - m);
            return m + std::log(sum_exp);
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    return std::numeric_limits<double>::quiet_NaN();
}
