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

    PMTrackExpressionVars();
    ~PMTrackExpressionVars();

    // Parse expressions to find track names and prepare variable mappings
    void parse_exprs(const std::vector<std::string> &track_exprs,
                     std::vector<std::string> &exprs4compile);

    // Define Python variables in the local dictionary
    void define_py_vars(unsigned size, PMPY &ldict, bool use_python);

    // Set variable values for current batch of intervals
    void set_vars(const GInterval &interval, unsigned idx);

    // Get number of track variables
    unsigned get_num_track_vars() const { return m_track_vars.size(); }

    // Get track name for a variable
    const std::string &get_track_name(unsigned ivar) const {
        return m_track_vars[ivar].name;
    }

    // Get track for a variable
    GenomeTrack *get_track(unsigned ivar) const {
        return m_track_vars[ivar].track.get();
    }

    // Look up a variable by name
    const TrackVar *var(const char *name) const;

    // Get the bin size used by tracks (0 if not uniform)
    int64_t get_bin_size() const { return m_bin_size; }

    // Get common track type (NUM_TYPES if none or mixed)
    GenomeTrack::Type get_common_track_type() const { return m_common_track_type; }

    // Whether a common track type was detected
    bool has_common_track_type() const { return m_common_track_type_valid; }

    // Track path for the first track variable (for iterator selection)
    const std::string &first_track_path() const { return m_track_vars.front().track_path; }

    // Pad tail slots with NaN to prevent stale data in partial batches
    void pad_tail_with_nan(unsigned start_idx, unsigned end_idx);

private:
    std::vector<TrackVar> m_track_vars;
    std::unordered_map<std::string, size_t> m_var_map;  // track_name -> index in m_track_vars
    std::unordered_map<std::string, std::string> m_varname_to_track;  // var_name -> track_name (for collision detection)
    int64_t m_bin_size;  // Uniform bin size (0 if mixed)
    GenomeTrack::Type m_common_track_type{GenomeTrack::NUM_TYPES};
    bool m_common_track_type_valid{false};

    // Convert track name to valid Python variable name
    void name2var(const std::string &name, std::string &var_name) const;

    // Check if character position is a valid variable boundary
    bool is_var(const std::string &str, size_t start, size_t end) const {
        return (!start || !is_py_var_char(str[start - 1])) &&
               (end == str.size() || !is_py_var_char(str[end]));
    }

    // Add a track variable
    TrackVar &add_track_var(const std::string &track_name);
};

#endif /* PMTRACKEXPRESSIONVARS_H_ */
