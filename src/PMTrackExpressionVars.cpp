/*
 * PMTrackExpressionVars.cpp
 *
 * Manages track variables in expressions for pymisha
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unistd.h>

#include "PMTrackExpressionVars.h"
#include "TGLException.h"

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
        TGLError("Mixed track types in expression are not supported: '%s' is %s, expected %s",
                 track_name.c_str(), GenomeTrack::TYPE_NAMES[track_type],
                 GenomeTrack::TYPE_NAMES[m_common_track_type]);
    }

    if (track_type == GenomeTrack::FIXED_BIN) {
        // Create fixed bin track object
        auto *fixed_bin = new GenomeTrackFixedBin();
        var.track = std::unique_ptr<GenomeTrack>(fixed_bin);

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
    } else {
        TGLError("Track type '%s' not yet supported for track: %s",
                 GenomeTrack::TYPE_NAMES[track_type], track_name.c_str());
    }

    m_var_map[track_name] = m_track_vars.size() - 1;
    return var;
}

void PMTrackExpressionVars::parse_exprs(const std::vector<std::string> &track_exprs,
                                        std::vector<std::string> &exprs4compile)
{
    m_track_vars.clear();
    m_var_map.clear();
    m_varname_to_track.clear();
    m_bin_size = 0;
    m_common_track_type = GenomeTrack::NUM_TYPES;
    m_common_track_type_valid = false;

    exprs4compile.resize(track_exprs.size());

    for (size_t iexpr = 0; iexpr < track_exprs.size(); ++iexpr) {
        const std::string &expr = track_exprs[iexpr];
        std::string &expr4compile = exprs4compile[iexpr];
        expr4compile = expr;

        // Scan for track names in the expression
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

            // Check if this is a track name
            if (g_pmdb->track_exists(name)) {
                TrackVar &var = add_track_var(name);

                // Replace track name with variable name in expression
                size_t offset = expr4compile.size() - expr.size();
                expr4compile.replace(start + offset, name.size(), var.var_name);
            }
        }
    }

    vdebug("Parsed expressions, found %lu track variables\n", m_track_vars.size());
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
}

void PMTrackExpressionVars::set_vars(const GInterval &interval, unsigned idx)
{
    for (auto &var : m_track_vars) {
        if (var.track_type == GenomeTrack::SPARSE) {
            GenomeTrackSparse *sparse = dynamic_cast<GenomeTrackSparse *>(var.track.get());
            if (!sparse) {
                var.values[idx] = std::nan("");
                continue;
            }

            if (var.cur_chromid != interval.chromid) {
                std::string chrom_file = GenomeTrack::find_existing_1d_filename(
                    g_pmdb->chromkey(), var.track_path, interval.chromid);

                std::string full_path = var.track_path + "/" + chrom_file;
                if (access(full_path.c_str(), F_OK) != 0) {
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

        GenomeTrackFixedBin *fixed_bin = dynamic_cast<GenomeTrackFixedBin *>(var.track.get());
        if (!fixed_bin) {
            var.values[idx] = std::nan("");
            continue;
        }

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
        }

        if (!var.cur_chromid_valid) {
            var.values[idx] = std::nan("");
            continue;
        }

        // Read track value at interval midpoint
        int64_t pos = (interval.start + interval.end) / 2;
        int64_t bin = pos / var.bin_size;

        fixed_bin->goto_bin(bin);
        float val;
        if (fixed_bin->read_next_bin(val)) {
            var.values[idx] = val;
        } else {
            var.values[idx] = std::nan("");
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

void PMTrackExpressionVars::pad_tail_with_nan(unsigned start_idx, unsigned end_idx)
{
    for (auto &var : m_track_vars) {
        if (var.values) {
            for (unsigned i = start_idx; i < end_idx; ++i) {
                var.values[i] = std::nan("");
            }
        }
    }
}
