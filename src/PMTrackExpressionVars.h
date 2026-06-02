/*
 * PMTrackExpressionVars.h
 *
 * Manages track variables in expressions for pymisha
 * Creates and populates NumPy arrays for track values during iteration
 */

#ifndef PMTRACKEXPRESSIONVARS_H_
#define PMTRACKEXPRESSIONVARS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "pymisha.h"
#include "PMDb.h"
#include "GInterval.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "GenomeTrackArray.h"
#include "GenomeSeqScorer.h"

class PMTrackExpressionVars {
public:
    struct TrackVar {
        std::string name;           // Track name (e.g., "test.fixedbin")
        std::string var_name;       // Python variable name (e.g., "test_fixedbin")
        std::string track_path;     // Full path to track directory
        PMPY py_var;                // NumPy array for values
        std::vector<double> cpp_values; // Storage for values when Python is not used
        double *values;             // Raw pointer to array data
        std::unique_ptr<GenomeTrack> track;  // Track reader
        GenomeTrack::Type track_type;        // Track type
        int64_t bin_size;           // Bin size (for fixed bin tracks)
        int64_t last_bin{-1};       // Last bin read (for sequential seek skip)
        int cur_chromid;            // Currently loaded chromosome
        bool cur_chromid_valid;     // Whether current chromosome data is valid
    };

    // Virtual track variable — evaluated inline during scanning
    struct VTrackVar {
        std::string name;           // original vtrack name
        std::string var_name;       // Python-safe variable name

        // Value buffer (same pattern as TrackVar)
        PMPY py_var;                // Python array ref
        std::vector<double> cpp_values;
        double *values{nullptr};    // pointer into array data

        // Borrowed reference to the Python spec dict (kept alive by caller)
        PyObject *py_spec{nullptr};

        // Iterator shift modifiers
        int64_t sshift{0};
        int64_t eshift{0};

        // Scorer (for sequence-based vtracks) — polymorphic
        std::unique_ptr<GenomeSeqScorer> scorer;

        // Per-vtrack sequence fetcher (fork-safe: each vtrack owns its own)
        std::unique_ptr<GenomeSeqFetch> seqfetch;

        // For value-based vtracks with a physical track source
        std::string src_track_name;
        std::string src_track_path;
        GenomeTrack::Type src_track_type{GenomeTrack::NUM_TYPES};
        std::unique_ptr<GenomeTrack> src_track;
        std::string func;
        double param{0.0};  // function-specific parameter (quantile, distance)
        int src_cur_chromid{-1};
        bool src_cur_chromid_valid{false};

        // Vtrack-instance dedup: when several dense (FIXED_BIN) vtracks share the
        // same source track + sshift/eshift, only the first ("primary") owns a
        // reader (with the union of all their funcs registered) and does the
        // read_interval; "followers" set this to the primary's index and just
        // extract their own last_<func>() from the primary's reader.
        // -1 = primary or standalone. Relies on set_vars evaluating vvars in
        // registration order (primary before its followers) per interval.
        int shared_primary_idx{-1};

        // Sparse fast-reduce state
        const std::vector<GInterval> *sp_intervals_ptr{nullptr};
        const std::vector<float> *sp_vals_ptr{nullptr};
        size_t sp_scan_idx{0};
        int64_t sp_last_start{0};
        bool sp_scan_ready{false};

        // For value-based vtracks with interval/DataFrame source
        // (These are computed by pm_vtrack_compute via Python, so we keep the spec)
        bool use_pm_vtrack_compute{false};
    };

    PMTrackExpressionVars();
    ~PMTrackExpressionVars();

    // Parse expressions to find track names and prepare variable mappings
    // py_vtracks: optional Python dict mapping vtrack_name -> spec dict
    void parse_exprs(const std::vector<std::string> &track_exprs,
                     std::vector<std::string> &exprs4compile,
                     PyObject *py_vtracks = nullptr);

    // Define Python variables in the local dictionary
    void define_py_vars(unsigned size, PMPY &ldict, bool use_python);

    // Set variable values for current batch of intervals
    void set_vars(const GInterval &interval, unsigned idx);

    // Get number of track variables
    unsigned get_num_track_vars() const { return m_track_vars.size(); }

    // Get number of vtrack variables
    unsigned get_num_vtrack_vars() const { return m_vtrack_vars.size(); }

    // Get track name for a variable
    const std::string &get_track_name(unsigned ivar) const {
        return m_track_vars[ivar].name;
    }

    // Get track for a variable
    GenomeTrack *get_track(unsigned ivar) const {
        return m_track_vars[ivar].track.get();
    }

    // Look up a variable by name (checks both track vars and vtrack vars)
    const TrackVar *var(const char *name) const;

    // Look up a vtrack variable by name
    const VTrackVar *vtrack_var(const char *name) const;

    // Get the bin size used by tracks (0 if not uniform)
    int64_t get_bin_size() const { return m_bin_size; }

    // Get common track type (NUM_TYPES if none or mixed)
    GenomeTrack::Type get_common_track_type() const { return m_common_track_type; }

    // Whether a common track type was detected
    bool has_common_track_type() const { return m_common_track_type_valid; }

    // Whether the expression mixes 1D scalar formats (dense + sparse). R allows
    // this only with an explicit iterator; implicit iterator inference must
    // refuse it.
    bool has_mixed_track_types() const { return m_mixed_track_types; }

    // Track path for the first track variable (for iterator selection)
    const std::string &first_track_path() const { return m_track_vars.front().track_path; }

    // Pad tail slots with NaN to prevent stale data in partial batches
    void pad_tail_with_nan(unsigned start_idx, unsigned end_idx);

private:
    std::vector<TrackVar> m_track_vars;
    std::vector<VTrackVar> m_vtrack_vars;
    std::unordered_map<std::string, size_t> m_var_map;  // track_name -> index in m_track_vars
    std::unordered_map<std::string, size_t> m_vtrack_var_map;  // vtrack_name -> index in m_vtrack_vars
    std::unordered_map<std::string, size_t> m_value_vtrack_groups;  // dense (src+sshift+eshift) -> primary vvar index
    std::unordered_map<std::string, std::string> m_varname_to_track;  // var_name -> track_name (for collision detection)
    int64_t m_bin_size;  // Uniform bin size (0 if mixed)
    GenomeTrack::Type m_common_track_type{GenomeTrack::NUM_TYPES};
    bool m_common_track_type_valid{false};
    bool m_mixed_track_types{false};  // dense + sparse mixed in one expression

    // Convert track name to valid Python variable name
    void name2var(const std::string &name, std::string &var_name) const;

    // Check if character position is a valid variable boundary
    bool is_var(const std::string &str, size_t start, size_t end) const {
        return (!start || !is_py_var_char(str[start - 1])) &&
               (end == str.size() || !is_py_var_char(str[end]));
    }

    // Add a track variable
    TrackVar &add_track_var(const std::string &track_name);

    // Add a virtual track variable from spec dict
    VTrackVar &add_vtrack_var(const std::string &vtrack_name, PyObject *spec);

    // Build scorer for sequence-based vtrack
    void build_vtrack_scorer(VTrackVar &vvar, PyObject *spec);

    // Set up value-based vtrack from physical track source
    void setup_value_based_vtrack(VTrackVar &vvar, PyObject *spec, const std::string &func);

    // Apply an array track's column slice + reduction from the vtrack spec
    // (gvtrack.array.slice). No-op when the spec carries no slice keys, leaving
    // the default avg-over-all-columns reduction.
    void configure_array_slice(GenomeTrackArray &track, PyObject *spec);

    // Evaluate a value-based vtrack for a single interval
    double eval_value_based_vtrack(VTrackVar &vvar, const GInterval &interval);

    // Register the reducer(s) a value-based func needs on a 1D reader.
    static void register_value_funcs(GenomeTrack1D *t, const std::string &func,
                                     GenomeTrack::Type type);

    // Extract a dense reader's already-computed last_<func>() value (used by a
    // dedup follower, whose primary already did the read for this interval).
    static double extract_dense_last(GenomeTrack1D *t, const std::string &func,
                                     int64_t eval_start, double param);

    // Apply shift to interval, clamping to chromosome bounds
    static bool apply_shift(const GInterval &in, int64_t sshift, int64_t eshift,
                            const GenomeChromKey &chromkey, GInterval &out);
};

#endif /* PMTRACKEXPRESSIONVARS_H_ */
